# -*- coding: utf-8 -*-
"""cert_common.py — fusedjoin 证书 v2（0.1–0.5）的共享引擎。7 个 run_fusedjoin_<ds>.py 共用。

定位（见 POSITIONING.md）：保证是相对**部署判官**（teacher 门 prompt），不是相对 human GT；
human GT 只在最终评测。

五个修正：
  0.1 精度改行级 betting：审计行给 (TP_r, FP_r)，检验 E[FP_r − (1−p0)(TP_r+FP_r)] ≥ 0。
      FP 只来自 auto 且 auto ≤ 1/行 ⇒ FP_r ≤ 1 ⇒ −Z_r ≥ −p0（常数下界），无需 Nmax 归一。
  0.2 召回改 pair 级精确形式：Y_r = miss_r − ε·pos_r，检验均值 < 0 ⇒ pair 召回 ≥ 1−ε。
      payoff = ε·pos_r − miss_r ≥ −rej_r（rej_r = 该行被 τR 拒绝的候选数，先验无标签）。
  0.3 弃 A/B 切分：全审计行 + 事先排好阶梯（fixed-sequence fallback），α 不再过度分摊。
  0.4 精度/召回联合失败概率 P(P<T_P ∨ R<T_R) ≤ δ（union bound：α_P=α_R=δ/2）。
  0.5 oracle 定义统一：teacher = 部署门 prompt（门精度按定义 = 1）。

betting 用 Ville 上鞅：单元是**行**（不假设行内 pair 独立），有限样本有效。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import run_fusedjoin as RJ                      # noqa: E402
import run_hybrid_join_v10_6_old as M            # noqa: E402

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                                # pragma: no cover
    _ENC = None


def _ntok(s):
    return len(_ENC.encode(str(s))) if _ENC is not None else max(1, len(str(s)) // 3)


TAU_LADDER = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]   # τR 免费拒绝阈值阶梯（事先排好）


# ────────────────────────────────────────────────────────────────────────────
# betting：行级、有界、单边 e-process（Ville 上鞅，有限样本有效）
# ────────────────────────────────────────────────────────────────────────────
def bet_up_logwealth(payoff, lower, grid=None):
    """押注 E[payoff] > 0 的 log 资本，payoff_r ≥ −lower_r（lower_r 先验已知，>0）。

    归一 w_r = payoff_r / lower_r ∈ [−1, +∞)。对 λ∈(0,1] 构造乘性赌
    K_n(λ)=∏_t (1 + λ·w_t)。H0: E[payoff]≤0 下 E[w]≤0 ⇒ E[1+λw]≤1，K_n(λ) 是非负上鞅；
    λ 混合仍是上鞅。Ville ⇒ P(∃n: K_n ≥ 1/α) ≤ α。单元是**行**，不假设行内 pair 独立。
    """
    lower = np.asarray(lower, float)
    w = np.asarray(payoff, float) / np.maximum(lower, 1e-12)
    if grid is None:
        grid = np.linspace(0.05, 1.0, 20)
    logs = []
    for lam in grid:
        f = 1.0 + lam * w
        f = np.clip(f, 1e-12, None)
        logs.append(np.log(f).sum())
    logs = np.array(logs)
    m = logs.max()
    return m + np.log(np.exp(logs - m).mean())


def certify_precision(tp, fp, p0, alpha):
    """H0: precision ≤ p0 ⟺ E[FP_r − (1−p0)(TP_r+FP_r)] ≥ 0。

    reject（认证 precision ≥ p0）等价于押注 payoff = (1−p0)·TP_r − p0·FP_r > 0。
    下界：payoff ≥ −p0·FP_r ≥ −p0（auto ≤ 1/行 ⇒ FP_r ≤ 1），故 lower = p0。
    """
    payoff = (1.0 - p0) * np.asarray(tp, float) - p0 * np.asarray(fp, float)
    return bet_up_logwealth(payoff, p0) >= np.log(1.0 / alpha)


def certify_precision_general(tp, fp, p0, alpha, fp_ub, lower_mode="global"):
    """精度 betting。**默认全局常数**：L_P = p0·max_r E_r（整帧可算，不看标签）。

    对 best-result 方法（L1 等可一行出多个对）FP_r 可 >1，payoff ≥ −p0·E_r，
    故 L_P = p0·E_max。H0: precision ≤ p0 ⟺ E[FP_r − (1−p0)(TP_r+FP_r)] ≥ 0。

    lower_mode:
      global  —— 常数 p0·max(fp_ub)（有效；论文用这个）
      perrow  —— 逐行 p0·fp_ub_r（**无效**，只用于回归对照；定理要求 L 与行无关）
      shipped —— 常数 p0（旧 cert_common 原式；仅当 FP_r≤1 时有效）
    """
    payoff = (1.0 - p0) * np.asarray(tp, float) - p0 * np.asarray(fp, float)
    if payoff.size == 0:
        return False
    ub = np.asarray(fp_ub, float)
    if lower_mode == "perrow":
        lower = p0 * np.maximum(ub, 1.0)
    elif lower_mode == "shipped":
        lower = p0
    elif lower_mode == "global":
        lower = p0 * max(float(ub.max()), 1.0)
    else:
        raise ValueError(f"unknown lower_mode {lower_mode!r}")
    return bet_up_logwealth(payoff, lower) >= np.log(1.0 / alpha)


def certify_recall(miss, pos, rej_r, eps, alpha, lower_mode="global"):
    """H0: recall < 1−ε ⟺ E[miss_r − ε·pos_r] ≥ 0。

    reject（认证 pair 召回 ≥ 1−ε）等价于押注 payoff = ε·pos_r − miss_r > 0。
    下界：payoff ≥ −miss_r ≥ −NEM_max（NEM_r = 该行未 emit 候选数，整帧可算、不看标签）。

    **默认全局常数 L_R = max(rej_r)**。rej_r 必须覆盖整帧（部署时可在全帧算），
    不能只传审计行的逐行拒绝数——逐行归一化检验的是另一个（行加权）估计量，见 T1_theorem.md §5。
    lower_mode:
      global  —— 常数 max(rej_r)（有效；论文用这个）
      perrow  —— 逐行（**无效**，只用于回归对照）
      shipped —— 同 perrow（旧 cert_common 原式，**无效**）
    """
    payoff = eps * np.asarray(pos, float) - np.asarray(miss, float)
    if payoff.size == 0:
        return False
    r = np.asarray(rej_r, float)
    if lower_mode in ("perrow", "shipped"):
        lower = np.maximum(r, 1.0)
    elif lower_mode == "global":
        lower = max(float(r.max()), 1.0)
    else:
        raise ValueError(f"unknown lower_mode {lower_mode!r}")
    return bet_up_logwealth(payoff, lower) >= np.log(1.0 / alpha)


def certify_recall_global(miss, pos, nem_max, eps, alpha):
    """pair 级召回的全局常数形式（显式接口，避免把逐行数组误传成 global）。"""
    return certify_recall(miss, pos, np.full(len(miss), max(float(nem_max), 1.0)),
                          eps, alpha, lower_mode="global")


def certify_row_recall(miss_row, pos_row, eps, alpha):
    """行级 entity recall 证书。L=1（Pos_row, Miss_row ∈ {0,1}）。

    estimand: R_row = Σ 1{行至少 emit 一个真匹配} / Σ 1{行至少有一个真匹配}
    payoff = ε·Pos_row_r − Miss_row_r ∈ [−1, ε] ⇒ 全局常数 1。
    """
    payoff = eps * np.asarray(pos_row, float) - np.asarray(miss_row, float)
    if payoff.size == 0:
        return False
    return bet_up_logwealth(payoff, 1.0) >= np.log(1.0 / alpha)


def row_stats(fr, emitted, reject):
    """审计行上的 (pos, tp, fp, n, miss, rej_r)。只统计**已标注**的审计行候选对（complete-case）。"""
    ci = fr["ci"]; y = fr["y_t"]
    audit = sorted(fr["label_rows"])
    idx = {r: i for i, r in enumerate(audit)}
    is_audit = np.array([ci[k] in idx for k in range(len(ci))], dtype=bool)
    rej_r = np.array([int(reject[ci == r].sum()) for r in audit])
    lab = is_audit & ~np.isnan(y)
    pos = np.zeros(len(audit), dtype=int)
    tp = np.zeros(len(audit), dtype=int)
    fp = np.zeros(len(audit), dtype=int)
    for k in np.flatnonzero(lab):
        i = idx[int(ci[k])]
        if y[k] > 0.5:
            pos[i] += 1
            if emitted[k]:
                tp[i] += 1
        elif emitted[k]:
            fp[i] += 1
    n = tp + fp
    miss = pos - tp
    return pos, tp, fp, n, miss, rej_r


def rung(fr, tau, gate_labels=None):
    """一个阶梯档的 emitted 掩码与行级统计。

    gate_labels=None  —— 旧口径：门 = oracle = teacher（0.5，只适合做上界/自洽检查）。
    gate_labels 数组  —— **真实部署门**：emitted = auto ∪ (D & gate_yes)，oracle 仍是 teacher。
                         gate_labels ∈ {1,0,-1}（-1 = 缓存缺失，按“不 emit”回放）。

    全局常数在**整帧**上算（不看标签）：
      E_max = max_r |E_r|, NEM_max = max_r |C_r \\ E_r|
    """
    in_auto = fr["in_auto"]; fsc = fr["fsc"]; y = fr["y_t"]
    teacher_yes = np.nan_to_num(y, nan=0.0) > 0.5
    reject = (fsc <= tau) & ~in_auto
    D = ~in_auto & ~reject
    if gate_labels is None:
        gate_yes = teacher_yes
        gate_ok = np.ones(fr["nL"], bool)
    else:
        gl = np.asarray(gate_labels)
        gate_yes = gl == 1
        gate_ok = np.ones(fr["nL"], bool)
        # 该行在需要门判决的 D 对上有缓存缺失 ⇒ 该行的部署输出不可回放
        np.logical_and.at(gate_ok, fr["ci"][D & (gl < 0)], False)
    emitted = in_auto | (D & gate_yes)
    pos, tp, fp, n, miss, rej_r = row_stats(fr, emitted, reject)

    # 整帧全局常数（结构性；emitted 由部署门决定，不含 human GT）
    nL = fr["nL"]
    C_all = np.bincount(fr["ci"], minlength=nL)
    em_all = np.bincount(fr["ci"][emitted], minlength=nL)
    nem_all = C_all - em_all
    em_max = int(em_all.max())
    nem_max = int(nem_all.max())

    # 行级 entity recall 的 payoff 分量
    pos_row = (pos > 0).astype(int)
    tp_row = (tp > 0).astype(int)
    miss_row = pos_row - tp_row

    return dict(emitted=emitted, pos=pos, tp=tp, fp=fp, n=n, miss=miss, rej_r=rej_r,
                pos_row=pos_row, tp_row=tp_row, miss_row=miss_row,
                em_max=em_max, nem_max=nem_max, gate_ok=gate_ok,
                v_calls=int(D.sum()), gate_missing=int((~gate_ok).sum()))


# ────────────────────────────────────────────────────────────────────────────
# prompt 构建（按数据集；teacher 即部署门 prompt，0.5）
# ────────────────────────────────────────────────────────────────────────────
def _txt(n, d, cap):
    s = " ".join(str(x) for x in (n, d) if x and str(x) != "nan")
    return s[:cap].replace("\n", " ")


def build_prompt(kind, ln, ld, rn, rd, cap_l, cap_r):
    """kind: product / acm / categorize / scifact / biodex。返回 teacher（=门）prompt 原文。"""
    ln, rn = str(ln), str(rn)
    if kind == "product":
        l = f"{ln} — {str(ld)[:cap_l]}"
        r = f"{rn} — {str(rd)[:cap_r]}"
        return ("Are these the same product? Same brand and model, OR one is a fuller/"
                "terser listing of the same item = Yes. Different brand, model, or "
                "category = No.\n"
                f"Left: {l}\nRight: {r}\nAnswer Yes or No.")
    if kind == "acm":
        l = f"{ln} — {str(ld)[:cap_l]}"
        r = f"{rn} — {str(rd)[:cap_r]}"
        return ("Are these the same entity (the same publication/paper)?\n"
                "Two records that describe the same publication — even with a "
                "differently-worded title, an abbreviated author list, or different "
                "venue/year formatting — are the same entity = Yes. Different "
                "publications = No.\n"
                f"Left: {l}\nRight: {r}\nAnswer Yes or No.")
    if kind == "categorize":
        l = _txt(ln, ld, cap_l)
        r = str(rn)[:cap_r]
        return ("You are judging whether a product belongs to a category.\n"
                "Answer YES if the product described on the left is an instance of the "
                "category on the right — the category is a correct label for the product.\n"
                "Answer NO if the category is wrong, refers to a different kind of product, "
                "or is only loosely related. Sharing words is NOT enough.\n"
                f"Left: {l}\nRight: {r}\nAnswer Yes or No.")
    if kind == "scifact":
        l = _txt(ln, ld, cap_l)
        r = _txt(rn, rd, cap_r)
        return ("You are judging whether a scientific abstract supports a claim.\n"
                "Answer YES if the abstract provides evidence that the claim is true — "
                "even if it only supports part of the claim or uses different wording.\n"
                "Answer NO if the abstract contradicts the claim, is only topically "
                "related, or does not address the claim's specific assertion.\n"
                f"Left (claim): {l}\nRight (abstract): {r}\nAnswer Yes or No.")
    if kind == "biodex":
        l = _txt(ln, ld, cap_l)
        r = str(rn)[:cap_r]
        return ("Adverse drug event case. You are judging whether the drug on the right "
                "is reported as a suspected or interacting drug in the case on the left.\n"
                "Answer YES if the case reports this drug as suspected or interacting, "
                "even if the wording differs.\n"
                "Answer NO if the drug is not reported in the case, is only mentioned in "
                "passing, or is unrelated to the adverse event.\n"
                f"Left (case): {l}\nRight (drug): {r}\nAnswer Yes or No.")
    if kind == "company":
        ld, rd = str(ld), str(rd)      # v7：全文描述，无 name、无截断
        return ("Are these the same company? Same company (Wikipedia article vs. its own "
                "website, or two descriptions of the same business) = Yes. Different company, "
                "different industry, or unrelated business = No.\n"
                f"Left: {ld}\nRight: {rd}\nAnswer Yes or No.")
    raise ValueError(kind)


# ────────────────────────────────────────────────────────────────────────────
# 数据集规格（7 个）。cache 与 tag 沿用各数据集的运行脚本口径。
# ────────────────────────────────────────────────────────────────────────────
DS_SPECS = {
    "products":   dict(ds="products",          n_target=None, cache=".llm_decision_cache.json",
                       ds_prefix="products", teacher_tag="openai/qwen-plus|teacher|v6",
                       gate_tag="openai/qwen-plus|neutral|v6",
                       kind="product", cap_l=300, cap_r=300),
    "amazon":     dict(ds="amazon_google_1000", n_target=2000, cache=".llm_decision_cache_amazon_google_1000.json",
                       ds_prefix=None, teacher_tag="openai/qwen-plus|teacher|v1-amazon_google_1000",
                       gate_tag="openai/qwen-plus|neutral|v1-amazon_google_1000",
                       kind="product", cap_l=300, cap_r=300),
    "acm":        dict(ds="dblp_acm_sample",   n_target=1000, cache=".llm_decision_cache_dblp_acm_v6.json",
                       ds_prefix=None, teacher_tag="openai/qwen-plus|teacher|v6",
                       gate_tag="openai/qwen-plus|neutral|v6",
                       kind="acm", cap_l=300, cap_r=300),
    "company":     dict(ds="company",           n_target=200,  cache=".llm_decision_cache_company.json",
                       ds_prefix=None, teacher_tag="openai/qwen-plus|teacher|v7",
                       gate_tag="openai/qwen-plus|neutral|v7",
                       kind="company", cap_l=None, cap_r=None),
    "wdc":        dict(ds="wdc_computers_500", n_target=5000, cache=".llm_decision_cache_wdc_computers_500.json",
                       ds_prefix=None, teacher_tag="openai/qwen-plus|teacher|v1-wdc",
                       gate_tag="openai/qwen-plus|neutral|v1-wdc",
                       kind="product", cap_l=300, cap_r=300),
    "scifact":    dict(ds="scifact",           n_target=1000, cache=".llm_decision_cache_scifact.json",
                       ds_prefix=None, teacher_tag="openai/qwen-plus|teacher|v1-scifact",
                       gate_tag="openai/qwen-plus|neutral|v1-scifact",
                       kind="scifact", cap_l=1200, cap_r=1600),
    "biodex":     dict(ds="biodex_500",        n_target=5000, cache=".llm_decision_cache_biodex_500.json",
                       ds_prefix=None, teacher_tag="openai/qwen-plus|teacher|v1-biodex",
                       gate_tag="openai/qwen-plus|neutral|v1-biodex",
                       kind="biodex", cap_l=1200, cap_r=120),
}


def cache_get(cache, tag, li, rj, ds_prefix=None):
    """缓存键前缀兼容。

    products 的缓存键是 `products|{tag}|{lid}|{rid}`（`_dataset="products"`）；
    其余六数据集的键是 `{tag}|{lid}|{rid}`。先按前缀查，再回退到无前缀。
    """
    ds = ds_prefix if ds_prefix is not None else cache.get("_dataset")
    if ds:
        v = cache.get(f"{ds}|{tag}|{li}|{rj}")
        if v is not None:
            return v
    return cache.get(f"{tag}|{li}|{rj}")


def gate_labels_for(fr, tag):
    """从缓存读整帧 C 上的门标签：1/0/-1（-1 = 未缓存）。**不看 human GT**。"""
    lid = fr["left"]["left_id"].astype(str).values
    rid = fr["right"]["right_id"].astype(str).values
    cache = fr["cache"]
    ds_prefix = (fr.get("cfg") or {}).get("ds_prefix")
    out = np.full(len(fr["C"]), -1, dtype=int)
    for k, (i, j) in enumerate(fr["C"]):
        v = cache_get(cache, tag, lid[i], rid[j], ds_prefix)
        if v == "yes":
            out[k] = 1
        elif v == "no":
            out[k] = 0
    return out


def _configure(sp):
    """把规格写进 M.DATASET_CFG / M.GATE_CFG。

    **部署门 ≠ oracle**：neutral/strict/lenient 用真实门 prompt，teacher 用独立判官。
    旧版把 neutral/strict 设成 teacher，导致“门=oracle”，证书与报的系统两张皮。
    """
    M.DATASET_CFG[sp["ds"]] = dict(
        n_target=sp["n_target"], cache=sp["cache"], ds_prefix=sp["ds_prefix"],
        neutral=sp["gate_tag"], teacher=sp["teacher_tag"])
    M.GATE_CFG[sp["ds"]] = dict(model="openai/qwen-plus",
                                lenient=sp["gate_tag"], strict=sp["gate_tag"])


def token_bill(sp, pairs, left, right):
    """pairs 上 teacher（=门）prompt 的 prompt+completion token 估计（tiktoken，零 LLM）。"""
    ln = left["left_name"].astype(str).values
    ld = left["left_description"].astype(str).values
    rn = right["right_name"].astype(str).values
    rd = right["right_description"].astype(str).values
    p_tok = c_tok = 0
    for (i, j) in pairs:
        p_tok += _ntok(build_prompt(sp["kind"], ln[i], ld[i], rn[j], rd[j],
                                    sp["cap_l"], sp["cap_r"]))
        c_tok += 1     # 'yes' / 'no' 各 1 token
    return dict(calls=len(pairs), prompt=p_tok, completion=c_tok, total=p_tok + c_tok)


def run(sp_key, args):
    sp = DS_SPECS[sp_key]
    _configure(sp)
    # 缓存缺失（如 amazon 尚无 qwen teacher 缓存）时以空缓存起跑，证书自然报「无标注」，
    # 不因文件缺失崩溃 —— 语义与「零 LLM 回放」一致。
    cache_path = ROOT / sp["cache"]
    if not cache_path.exists():
        cache_path.write_text("{}", encoding="utf-8")
    aP = args.delta / 2.0
    aR = args.delta / 2.0

    t0 = time.time()
    pre = M.prepare(sp["ds"], args.seed, args.m)
    fr = RJ._frame(pre)
    C = fr["C"]; gt_set = fr["gt_set"]
    label_rows = fr["label_rows"]
    in_auto = fr["in_auto"]

    # 部署门标签（整帧 C 上；-1 = 缓存缺失，按“不 emit”回放）
    gate_tag = getattr(args, "gate_tag", None) or sp["gate_tag"]
    gate_labels = gate_labels_for(fr, gate_tag)

    # teacher 对（审计行 C 对，证书用）+ 升级带 D（部署门，全部行）
    teacher_pairs = sorted(p for p in C if p[0] in label_rows)
    D = sorted(p for p in C if not p in fr["auto_set"])
    nL, nR = fr["nL"], fr["nR"]

    # 一致性：teacher 缓存覆盖审计行 C 对的比例（证书只认已标注对）
    lab = ~np.isnan(fr["y_t"])
    y_t = fr["y_t"]
    n_audit_lab = int(lab.sum())

    print(f"[{sp_key}] {nL}×{nR}  |GT|={len(gt_set):,}  |C|={len(C):,}  auto={int(in_auto.sum()):,}  "
          f"|D|={len(D):,}  teacher对={len(teacher_pairs):,}  审计行={len(label_rows)} "
          f"标注候选对={n_audit_lab:,}", flush=True)
    print(f"    定位: 部署门 = {gate_tag}；oracle = {sp['teacher_tag']}（独立判官）；"
          f"human GT 只在最终评测", flush=True)
    print(f"    门缓存缺失行 = {int((gate_labels < 0).sum() > 0)} 行相关；"
          f"缺失对 = {int((gate_labels < 0).sum()):,}", flush=True)

    # ── 阶梯表（证书 + 部署 P/R/F1）──
    h0 = args.p0[0]; e0 = args.eps[0]
    audit = sorted(label_rows)
    print(f"\n    阶梯（固定顺序）  p0={h0}  ε={e0}  α_P=α_R={aP:.4f} (δ={args.delta})", flush=True)
    print(f"      {'τR':>6} {'|M|':>7} {'升级调用':>8} {'点P(teacher)':>12} {'点R(C内)':>9} "
          f"{'点R行':>8} {'精度证':>6} {'行召回证':>8} {'pair召回证':>9} {'humanF1':>8}", flush=True)
    rows = []
    for tau in TAU_LADDER:
        g = rung(fr, tau, gate_labels=gate_labels)
        keep = g["gate_ok"][audit]
        tp_k = g["tp"][keep]; fp_k = g["fp"][keep]; n_k = g["n"][keep]
        pos_k = g["pos"][keep]; miss_k = g["miss"][keep]
        pr_k = g["pos_row"][keep]; mr_k = g["miss_row"][keep]
        stp, s_n = int(tp_k.sum()), int(n_k.sum())
        s_pos, s_miss = int(pos_k.sum()), int(miss_k.sum())
        s_pos_row, s_tp_row = int(pr_k.sum()), int((pr_k - mr_k).sum())
        pP = stp / s_n if s_n else 0.0
        pR = 1.0 - s_miss / s_pos if s_pos else float("nan")
        pR_row = s_tp_row / s_pos_row if s_pos_row else float("nan")
        cp = certify_precision_general(tp_k, fp_k, h0, aP, g["em_max"])
        cr_row = certify_row_recall(mr_k, pr_k, e0, aR)
        cr_pair = certify_recall_global(miss_k, pos_k, g["nem_max"], e0, aR)
        matched = set(C[k] for k in np.flatnonzero(g["emitted"]))
        _, _, f1 = M._final_metrics(matched, gt_set)[:3]
        rows.append((tau, g, keep, f1))
        print(f"      {tau:>6.2f} {int(g['emitted'].sum()):>7,} {g['v_calls']:>8,} "
              f"{pP:>12.4f} {pR:>9.4f} {pR_row:>8.4f} {'✓' if cp else '—':>6} "
              f"{'✓' if cr_row else '—':>8} {'✓' if cr_pair else '—':>9} {f1:>8.4f}", flush=True)

    # ── 预指定点（论文报这个；跑之前定死，不看 human GT、不挑网格）──
    fixed = None
    for tau, g, keep, f1 in rows:
        if abs(tau - args.fixed_tau) < 1e-12:
            tp_k = g["tp"][keep]; fp_k = g["fp"][keep]; n_k = g["n"][keep]
            pos_k = g["pos"][keep]; miss_k = g["miss"][keep]
            pr_k = g["pos_row"][keep]; mr_k = g["miss_row"][keep]
            cp = certify_precision_general(tp_k, fp_k, args.fixed_p0, aP, g["em_max"])
            cr_row = certify_row_recall(mr_k, pr_k, args.fixed_eps, aR)
            cr_pair = certify_recall_global(miss_k, pos_k, g["nem_max"], args.fixed_eps, aR)
            stp, s_n = int(tp_k.sum()), int(n_k.sum())
            s_pos, s_miss = int(pos_k.sum()), int(miss_k.sum())
            s_pos_row, s_tp_row = int(pr_k.sum()), int((pr_k - mr_k).sum())
            fixed = dict(tau=float(tau), p0=float(args.fixed_p0), eps=float(args.fixed_eps),
                         precision_cert=bool(cp), row_recall_cert=bool(cr_row),
                         pair_recall_cert=bool(cr_pair),
                         point_precision=(stp / s_n if s_n else 0.0),
                         point_pair_recall=(1.0 - s_miss / s_pos if s_pos else float("nan")),
                         point_row_recall=(s_tp_row / s_pos_row if s_pos_row else float("nan")),
                         emitted=int(g["emitted"].sum()), human_f1=float(f1),
                         n_audit_complete=int(keep.sum()), gate_missing=int(g["gate_missing"]),
                         em_max=int(g["em_max"]), nem_max=int(g["nem_max"]))
            break
    if fixed is None:
        raise ValueError(f"fixed_tau={args.fixed_tau} 不在阶梯 {TAU_LADDER} 上")
    print(f"\n    预指定点（论文报告用）: p0={fixed['p0']:.2f} ε={fixed['eps']:.2f} "
          f"τR={fixed['tau']:.2f} δ={args.delta}", flush=True)
    print(f"      |M|={fixed['emitted']:,}  审计完备行={fixed['n_audit_complete']}  "
          f"门缺失行={fixed['gate_missing']}  E_max={fixed['em_max']}  NEM_max={fixed['nem_max']}", flush=True)
    print(f"      点P={fixed['point_precision']:.4f} 点R(C内)={fixed['point_pair_recall']:.4f} "
          f"点R行={fixed['point_row_recall']:.4f}  humanF1={fixed['human_f1']:.4f}", flush=True)
    print(f"      精度证={fixed['precision_cert']}  行级召回证={fixed['row_recall_cert']}  "
          f"pair召回证={fixed['pair_recall_cert']}", flush=True)

    # ── 前沿矩阵（只作展示，不作为报告点）──
    print(f"\n    前沿矩阵（仅展示；论文报上面的预指定点）  主格 = 精度∧行级召回 ∧ humanF1", flush=True)
    print("      p0\\ε   " + " ".join(f"{e:>12}" for e in args.eps))
    joint = {}
    joint_pair = {}
    for p0 in args.p0:
        cells = []
        for eps in args.eps:
            best = None
            for tau, g, keep, f1 in rows:
                cp = certify_precision_general(g["tp"][keep], g["fp"][keep], p0, aP, g["em_max"])
                cr = certify_row_recall(g["miss_row"][keep], g["pos_row"][keep], eps, aR)
                if cp and cr:
                    best = (tau, f1)
                    break
            if best:
                cells.append((True, best[1]))
                joint[(p0, eps)] = best
            else:
                cells.append((False, None))
        line = " ".join(f"{('✓ %.3f' % f) if ok else '✗':>12}" for ok, f in cells)
        print(f"      {p0:<7.2f} " + line)
    for p0 in args.p0:
        for eps in args.eps:
            for tau, g, keep, f1 in rows:
                cp = certify_precision_general(g["tp"][keep], g["fp"][keep], p0, aP, g["em_max"])
                cr = certify_recall_global(g["miss"][keep], g["pos"][keep], g["nem_max"], eps, aR)
                if cp and cr:
                    joint_pair[(p0, eps)] = (tau, f1)
                    break

    # ── token / LLM 调用估计（teacher 审计 + 部署门 D，同 prompt，0.5）──
    bill_t = token_bill(sp, teacher_pairs, fr["left"], fr["right"])
    bill_d = token_bill(sp, D, fr["left"], fr["right"])
    tot_calls = bill_t["calls"] + bill_d["calls"]
    tot_tok = bill_t["total"] + bill_d["total"]

    print(f"\n    LLM 调用（teacher 审计 {bill_t['calls']:,} + 部署门|D| {bill_d['calls']:,}）"
          f"= {tot_calls:,}", flush=True)
    print(f"    TOKEN（估算, tiktoken cl100k_base 代理）: prompt={bill_t['prompt']+bill_d['prompt']:,}  "
          f"completion={bill_t['completion']+bill_d['completion']:,}  total={tot_tok:,}", flush=True)

    out = dict(ds=sp_key, frame=f"{nL}x{nR}", gt=len(gt_set), C=len(C), auto=int(in_auto.sum()),
               D=len(D), teacher_pairs=len(teacher_pairs), audit_rows=len(label_rows),
               audit_labeled_pairs=n_audit_lab, gate_tag=gate_tag,
               teacher_tag=sp["teacher_tag"],
               gate_missing_pairs=int((gate_labels < 0).sum()),
               delta=args.delta, p0=args.p0, eps=args.eps,
               fixed=fixed,
               fixed_spec=dict(p0=args.fixed_p0, eps=args.fixed_eps, tau=args.fixed_tau),
               joint={f"{p0}/{eps}": (best[0], best[1]) for (p0, eps), best in joint.items()},
               joint_pair={f"{p0}/{eps}": (best[0], best[1]) for (p0, eps), best in joint_pair.items()},
               ladder=[dict(tau=tau, emitted=int(g["emitted"].sum()), v_calls=g["v_calls"],
                            n_audit_complete=int(keep.sum()), gate_missing=int(g["gate_missing"]),
                            em_max=int(g["em_max"]), nem_max=int(g["nem_max"]),
                            preference_note="global constants; row recall L=1",
                            f1=f1) for tau, g, keep, f1 in rows],
               calls=dict(teacher=bill_t["calls"], gate=bill_d["calls"], total=tot_calls),
               tokens=dict(teacher=bill_t["total"], gate=bill_d["total"], total=tot_tok),
               elapsed_min=(time.time() - t0) / 60)
    json.dump(out, open(HERE / f"_cert_{sp_key}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"    结果写入 _cert_{sp_key}.json", flush=True)
    return out


def main(sp_key):
    """7 个 run_fusedjoin_<ds>.py 的统一入口：解析参数 → run(sp_key)。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.05, help="联合失败概率上界（0.4）")
    ap.add_argument("--p0", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=250, help="审计行采样预算（全量，无切分）")
    ap.add_argument("--gate-tag", default=None,
                    help="覆盖部署门缓存 tag（默认用 DS_SPECS[ds]['gate_tag']）")
    ap.add_argument("--fixed-p0", type=float, default=0.90, help="预指定精度目标（跑前定死）")
    ap.add_argument("--fixed-eps", type=float, default=0.10, help="预指定召回损失上界（跑前定死）")
    ap.add_argument("--fixed-tau", type=float, default=0.95, help="预指定 τR（跑前定死）")
    args = ap.parse_args()
    return run(sp_key, args)
