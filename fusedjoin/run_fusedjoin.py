# -*- coding: utf-8 -*-
"""run_fusedjoin.py — 放弃分桶、纯 fused 的 join（按老师反馈重构）+ 两个零 LLM 诊断。

设计（对应「Verification is the Bottleneck: Evidence-Adaptive Gating」）：
  单一 fused 分数 s = w·pct(dense) + (1−w)·pct(sparse) 贯穿全程
  auto       = mutual-NN（免费接受）
  τR 拒绝    = fused 上的行级 CP 单一阈值（替掉三桶独立 τR）
  升级门     = 非 auto 非拒绝的对 → lenient → strict（或 neutral 单段），
              门的 yes 再按 pair 级证据（fused / margin）门控
  ε_V 认证   = 随机审计行对门漏检做行级 CP，替换 GT 回滚 + 手选门（部署可证明）

本文件当前提供两个零 LLM 诊断（读缓存，不调大模型）：
  --diag calib       步骤 1：条件校准 —— fused 的含义是否随结构层漂移
  --diag cardinality 步骤 4：基数口径对照 —— set-based vs greedy-1to1 召回，
                     裁定 lenient 召回倒挂是不是一对一碰撞造成的

用法：
  python run_fusedjoin.py --diag calib
  python run_fusedjoin.py --diag cardinality
  python run_fusedjoin.py --diag calib --datasets products --seed 0
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import betaln, betainc
from scipy.stats import beta as _beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_hybrid_join_v10_10 as v10   # noqa: E402
import run_hybrid_join_v12 as v12      # noqa: E402
import run_hybrid_join_v10_6_old as M  # noqa: E402

K = 10
EPS = 0.05


def _pct(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return np.zeros(0)
    o = np.argsort(v, kind="stable")
    r = np.empty(v.size)
    r[o] = np.arange(1, v.size + 1)
    return r / v.size


def fused_candidates(fused, k=K):
    nL, nR = fused.shape
    row_cand = [set() for _ in range(nL)]
    col_cand = [set() for _ in range(nR)]
    for S, axis, sink in ((fused, 1, row_cand), (fused.T, 0, col_cand)):
        kk = min(k, S.shape[1] - 1)
        if kk <= 0:
            continue
        top = np.argpartition(-S, kth=kk - 1, axis=1)[:, :kk]
        for r in range(top.shape[0]):
            sink[r].update(int(x) for x in top[r])
    cand = set()
    for i in range(nL):
        for j in row_cand[i]:
            cand.add((i, j))
    for j in range(nR):
        for i in col_cand[j]:
            cand.add((i, j))
    return row_cand, col_cand, cand


# ── 候选预算 k（P1-1）：split-conformal ────────────────────────────────────
def rank_rc(S):
    """行/列降序秩（1-based）。"""
    S = np.asarray(S, dtype=float)
    nL, nR = S.shape
    orow = np.argsort(-S, axis=1, kind="stable")
    rr = np.empty((nL, nR), dtype=np.int32)
    np.put_along_axis(rr, orow, np.arange(1, nR + 1, dtype=np.int32)[None, :], axis=1)
    ocol = np.argsort(-S, axis=0, kind="stable")
    rc = np.empty((nL, nR), dtype=np.int32)
    np.put_along_axis(rc, ocol, np.arange(1, nL + 1, dtype=np.int32)[:, None], axis=0)
    return rr, rc


def order_stat_k(r, eps):
    """split-conformal 分位：第 ceil((1−eps)(m+1)) 个次序统计量；越界返回 inf。"""
    r = np.asarray(r)
    m = int(r.size)
    if m == 0:
        return float("inf")
    idx = int(np.ceil((1.0 - eps) * (m + 1)))
    if idx > m:
        return float("inf")
    return int(np.sort(r)[idx - 1])


def calibrate_k(fused, pos_pairs, rows=None, eps=0.10, side="min", bucket_of=None):
    """split-conformal 候选预算（P1-1）。

    fused       nL×nR 分数矩阵
    pos_pairs   校准正例 [(i,j), ...]（部署版应是 dev 行上的 teacher 正例）
    rows        只用左行在 rows 内的正例（None=全部）
    side        'min' → r=min(row_rank,col_rank)（统一 k 的紧口径）
                'row' → 只数 row_rank（Rule S 左侧）
                'col' → 只数 col_rank（Rule S 右侧）
    bucket_of   (i,j)->桶名；给定则逐桶返回 k_by_bucket（空桶回退到全局 k）

    保证（可交换性下）：P(side 秩 ≤ k̂) ≥ 1−eps，无分布假设、有限样本。
    """
    rr, rc = rank_rc(fused)
    keep = set(rows) if rows is not None else None
    pairs = [p for p in pos_pairs if keep is None or p[0] in keep]
    if side == "row":
        vals = np.array([rr[i, j] for (i, j) in pairs], dtype=int)
    elif side == "col":
        vals = np.array([rc[i, j] for (i, j) in pairs], dtype=int)
    else:
        vals = np.array([min(rr[i, j], rc[i, j]) for (i, j) in pairs], dtype=int)
    k = order_stat_k(vals, eps)
    out = dict(k=k, m=int(vals.size), eps=eps, side=side, k_by_bucket=None)
    if bucket_of is not None:
        kb = {}
        for name in ("C1", "C2", "C3"):
            sub = []
            for (i, j) in pairs:
                if bucket_of(i, j) != name:
                    continue
                sub.append(rr[i, j] if side == "row"
                           else rc[i, j] if side == "col"
                           else min(rr[i, j], rc[i, j]))
            kb[name] = order_stat_k(np.array(sub, dtype=int), eps) if sub else k
        out["k_by_bucket"] = kb
    return out


def stratified_candidates(fused, k_row, k_col):
    """Rule S：行 i 取 top-k_row[i]，列 j 取 top-k_col[j]，并集。"""
    S = np.asarray(fused, dtype=float)
    nL, nR = S.shape
    cand = set()
    for i in range(nL):
        kk = int(min(max(int(k_row[i]), 0), nR - 1))
        if kk <= 0:
            continue
        top = np.argpartition(-S[i], kth=kk - 1)[:kk]
        for j in top:
            cand.add((i, int(j)))
    for j in range(nR):
        kk = int(min(max(int(k_col[j]), 0), nL - 1))
        if kk <= 0:
            continue
        top = np.argpartition(-S[:, j], kth=kk - 1)[:kk]
        for i in top:
            cand.add((int(i), j))
    return cand


# ── 预注册路由（P1-2）：默认关闭，FUSED_ROUTER=1 才读 story/_k_router.json ──
ROUTER_FILE = HERE / "fused join exp" / "story" / "_k_router.json"


def router_entry(ds):
    if os.environ.get("FUSED_ROUTER") != "1" or not ds:
        return None
    if not ROUTER_FILE.is_file():
        return None
    try:
        cfg = json.loads(ROUTER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    return (cfg.get("datasets") or {}).get(ds)


def load_gate_readonly(cache, tag, pairs, l_id, r_id, ds_prefix):
    """复刻 _llm_gate 的读键格式，只读缓存，绝不调 LLM。"""
    prefix = f"{ds_prefix}|" if ds_prefix else ""
    dec = {}
    for (i, j) in pairs:
        k = f"{prefix}{tag}|{l_id[i]}|{r_id[j]}"
        d = cache.get(k)
        if d in ("yes", "no"):
            dec[(i, j)] = (d == "yes")
    return dec


def _frame(pre):
    """候选集 + fused 分数 + teacher 标签 + 结构桶。零 LLM。"""
    cfg, gc = pre["cfg"], pre["gc"]
    fused = pre["fused"]
    gt_set = pre["gt_set"]
    left, right = pre["left"], pre["right"]
    lid = left["left_id"].values
    rid = right["right_id"].values

    # 结构桶（cx12 口径 C1/C2/C3）+ 左行 cx12 分位（细粒度）
    lrow, rrow = M.buckets_for(pre, "structure", M.B1, M.B2)

    # 预注册路由：FUSED_ROUTER=1 时按 story/_k_router.json 选候选预算
    ent = router_entry(pre.get("ds"))
    if ent and ent.get("mode") == "stratified" and ent.get("k_row") and ent.get("k_col"):
        nL_, nR_ = np.asarray(fused).shape
        kL = np.array([ent["k_row"][lrow[i]] for i in range(nL_)], dtype=int)
        kR = np.array([ent["k_col"][rrow[j]] for j in range(nR_)], dtype=int)
        cand = stratified_candidates(fused, kL, kR)
        k_used = f"stratified R={ent['k_row']} C={ent['k_col']}"
    elif ent and ent.get("mode") == "uniform" and int(ent.get("k", 0)) > 0:
        _, _, cand = fused_candidates(fused, int(ent["k"]))
        k_used = int(ent["k"])
    else:
        _, _, cand = fused_candidates(fused, K)
        k_used = K

    C = sorted(cand)
    ci = np.array([p[0] for p in C])
    cj = np.array([p[1] for p in C])
    fsc = fused[ci, cj]
    in_auto = np.array([p in pre["auto_set"] for p in C], dtype=bool)

    cache = json.load(open(HERE / cfg["cache"], encoding="utf-8"))
    dec_t = v12.load_decisions(cache, cfg["teacher"], C, lid, rid)
    y_t = np.array([(1.0 if dec_t[p] else 0.0) if p in dec_t else np.nan for p in C])

    b = M._bucket_of(ci, cj, lrow, rrow)
    q12_L = np.asarray(pre["q12_L"], dtype=float)
    q12_R = np.asarray(pre["q12_R"], dtype=float)
    pool = np.concatenate([q12_L, q12_R])
    pctL = _pct(q12_L)
    left_dec = np.floor(pctL[ci] * 10).astype(int)
    left_dec = np.clip(left_dec, 0, 9)

    return dict(cfg=cfg, gc=gc, C=C, ci=ci, cj=cj, fsc=fsc, in_auto=in_auto, y_t=y_t,
                b=b, left_dec=left_dec, cache=cache, left=left, right=right,
                gt_set=gt_set, nL=pre["nL"], nR=pre["nR"], k_used=k_used,
                label_rows=pre["label_rows"], auto_set=pre["auto_set"])


def diag_calibration(fr, ds):
    """步骤 1：fused 的条件校准 —— 结构层内 teacher 正例率随 fused 分位是否同一条曲线。"""
    y = fr["y_t"]; fsc = fr["fsc"]
    lab = ~np.isnan(y)
    print(f"\n  [{ds}] 条件校准（有 teacher 标签的对 n={int(lab.sum()):,}）")
    if lab.sum() < 50:
        print("    标签太少，跳过")
        return
    yy = y[lab].astype(int); ff = fsc[lab]; bb = fr["b"][lab]

    # 1) 全局 fused 十分位内的正例率，按结构桶 C1/C2/C3 分层
    edges = np.percentile(ff, np.linspace(0, 100, 11))
    edges[-1] += 1e-9
    dec = np.clip(np.searchsorted(edges, ff, side="right") - 1, 0, 9)
    names = {0: "C1", 1: "C2", 2: "C3"}
    print(f"    结构桶 × fused 十分位（teacher 正例率，括号=对数）：")
    hdr = "      " + " ".join(f"{q:>6}" for q in range(1, 11))
    print(hdr)
    for b in range(3):
        cells = []
        for q in range(10):
            m = (bb == b) & (dec == q)
            n = int(m.sum())
            if n:
                cells.append(f"{yy[m].mean():.3f}({n})")
            else:
                cells.append("   ·   ")
        print(f"     {names[b]}  " + " ".join(f"{c:>7}" for c in cells))
    # 2) 左行 cx12 十分位内的正例率（细粒度漂移）
    print(f"    左行 cx12 十分位 → 正例率（pct = cx12 越低越易）：")
    ld = fr["left_dec"][lab]
    row = []
    for q in range(10):
        m = ld == q
        row.append(f"{yy[m].mean():.3f}({int(m.sum())})" if m.sum() else "  ·  ")
    print("      " + " ".join(f"{c:>7}" for c in row))

    # 3) logistic：fused 单独 vs fused+结构桶，看桶是否加信号（漂移）
    X = ff.reshape(-1, 1)
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    clf0 = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, yy)
    a0 = roc_auc_score(yy, clf0.predict_proba(Xs)[:, 1])
    d1 = (bb == 1).astype(float); d2 = (bb == 2).astype(float)
    Xb = np.column_stack([Xs, d1, d2])
    clf1 = LogisticRegression(C=1.0, max_iter=2000).fit(Xb, yy)
    a1 = roc_auc_score(yy, clf1.predict_proba(Xb)[:, 1])
    coef = clf1.coef_[0]
    print(f"    AUC(fused)={a0:.3f}  AUC(fused+桶)={a1:.3f}  Δ={a1 - a0:+.3f}")
    print(f"    logistic 系数：fused={coef[0]:+.2f}  I[C2]={coef[1]:+.2f}  I[C3]={coef[2]:+.2f}"
          f"   （桶系数≈0 ⇒ 单阈值够，分桶退化为 Mondrian 分层）")


def _one_to_one_recall(order, gate_yes, gt_set, auto_set, nL, nR):
    """部署口径的一对一贪心：auto 先占行/列，D 按 fused 降序，gate-yes 就占（FP 也会占）。"""
    claimed_l = np.zeros(nL, dtype=bool)
    claimed_r = np.zeros(nR, dtype=bool)
    confirmed = 0
    for (i, j) in auto_set:
        claimed_l[i] = claimed_r[j] = True
        if (i, j) in gt_set:
            confirmed += 1
    for (i, j) in order:
        if claimed_l[i] or claimed_r[j]:
            continue
        if gate_yes.get((i, j), False):
            claimed_l[i] = claimed_r[j] = True
            if (i, j) in gt_set:
                confirmed += 1
    return confirmed / len(gt_set) if gt_set else 0.0


def diag_cardinality(fr, ds):
    """步骤 4：同一批 lenient/neutral 决策，set-based 与 greedy-1to1 两个口径的召回。"""
    fsc = fr["fsc"]; y = fr["y_t"]; in_auto = fr["in_auto"]
    C = fr["C"]; gt_set = fr["gt_set"]; nL, nR = fr["nL"], fr["nR"]
    cfg, gc = fr["cfg"], fr["gc"]
    lid = fr["left"]["left_id"].values; rid = fr["right"]["right_id"].values
    ci = fr["ci"]

    label_mask = np.array([i in fr["label_rows"] for i in ci], dtype=bool)
    calib = label_mask & ~np.isnan(y)
    tR = M.tau_reject(fsc[calib], y[calib] > 0.5, ci[calib], EPS) if calib.sum() else -np.inf
    reject = (fsc <= tR) & (~in_auto)
    D = [C[k] for k in np.flatnonzero(~in_auto & ~reject)]
    if not D:
        print(f"\n  [{ds}] 争议队列为空，跳过")
        return
    print(f"\n  [{ds}] 基数口径对照  |C|={len(C):,} auto={int(in_auto.sum()):,} "
          f"τR={tR:.4f} 拒绝={int(reject.sum()):,} D={len(D):,}")

    dec_len = load_gate_readonly(fr["cache"], gc["lenient"], D, lid, rid, cfg["ds_prefix"])
    dec_neu = load_gate_readonly(fr["cache"], cfg["neutral"], D, lid, rid, cfg["ds_prefix"])
    fsc_of = {p: fsc[k] for k, p in enumerate(C)}
    order = sorted(D, key=lambda p: fsc_of[p], reverse=True)

    print(f"    {'门':>8} {'a(敏感)':>9} {'b(FPR)':>9} {'set召回':>8} {'1to1召回':>8} "
          f"{'覆盖yes':>8}")
    for name, dec in (("lenient", dec_len), ("neutral", dec_neu)):
        inD_gt = [p for p in D if p in gt_set]
        inD_non = [p for p in D if p not in gt_set]
        a = np.mean([dec.get(p, False) for p in inD_gt]) if inD_gt else float("nan")
        b = np.mean([dec.get(p, False) for p in inD_non]) if inD_non else float("nan")
        # set-based（无 occupancy，fusedjoin 现在的口径）
        matched = set(fr["auto_set"]) | {p for p in D if dec.get(p, False)}
        set_rec = len(matched & gt_set) / len(gt_set)
        # 部署一对一（occupancy，FP 占坑）
        o2o_rec = _one_to_one_recall(order, dec, gt_set, fr["auto_set"], nL, nR)
        nyes = sum(1 for p in D if dec.get(p, False))
        print(f"    {name:>8} {a:>9.3f} {b:>9.3f} {set_rec:>8.4f} {o2o_rec:>8.4f} {nyes:>8,}")


def _cp_upper(k, n, alpha):
    """二项比例的 Clopper–Pearson 上界（k 次命中 / n 次试验，1−alpha 置信）。

    召回认证保留 CP：miss 率是边界量（k≈0），betting 区间在边界上退化为 [0,1]（vacuous），
    CP 是 i.i.d. 行的**精确**有限样本上界，是这里的正确工具。
    """
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(_beta.ppf(1 - alpha, k + 1, n - k))


def _betting_rho_two(m, t_opt, alpha_opt):
    """TwoSidedNormalMixture::best_rho（Howard et al. 2018, confseq 源码）。"""
    v_opt = m * (1.0 - m) * t_opt
    L = 2.0 * np.log(1.0 / alpha_opt) + np.log(1.0 + 2.0 * np.log(1.0 / alpha_opt))
    return v_opt / L


def _log_incomplete_beta(a, b, x):
    """log( I_x(a,b) · B(a,b) )，x=1 退化为 log B(a,b)。对 betainc 加 1e-300 地板防 log(0)。"""
    if x >= 1.0:
        return betaln(a, b)
    return np.log(max(betainc(a, b, x), 1e-300)) + betaln(a, b)


def _betting_log_supermg_one(S, n, m, t_opt, alpha_opt):
    """WSR beta-binomial 共轭混合**单边**（正向押注）资本的对数（confseq BetaBinomialMixture,
    is_one_sided=True 的闭式，经 g+h=1 化简）。S=Σx_i，n 试验数，m 待检均值。"""
    m = float(m)
    rho = _betting_rho_two(m, t_opt, 2.0 * alpha_opt)
    r = max(rho - m * (1.0 - m), 1e-3 * m * (1.0 - m))
    A = r / m + (n - S)
    B = r / (1.0 - m) + S
    A0 = r / m
    B0 = r / (1.0 - m)
    x = 1.0 - m
    return (-(S * np.log(m) + (n - S) * np.log(1.0 - m))
            + _log_incomplete_beta(A, B, x) - _log_incomplete_beta(A0, B0, x))


def _betting_lower(S, n, alpha, t_opt=None):
    """Bernoulli 均值的 WSR 单边 betting 下界（有限样本有效，i.i.d. 前提）。

    正向押注资本 K(m) 随 m 单调递减；下界 = 最小的 m 使 K(m) ≤ 1/α。
    """
    S = int(S)
    n = int(n)
    if n == 0 or S == 0:
        return 0.0
    if t_opt is None:
        t_opt = n
    logthr = np.log(1.0 / alpha)
    p = S / n
    lo, hi = 1e-6, min(p, 1.0 - 1e-6)
    f = lambda m: _betting_log_supermg_one(S, n, m, t_opt, alpha) - logthr
    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return hi


def _dispute(fr, eps=0.05):
    """争议队列 D 及门决策（只读缓存）。返回 (D_mask, dec_arr, tR, teacher_vol, tpos, calib)。"""
    C = fr["C"]; ci = fr["ci"]; fsc = fr["fsc"]; y = fr["y_t"]; in_auto = fr["in_auto"]
    cfg, gc = fr["cfg"], fr["gc"]
    lid = fr["left"]["left_id"].values; rid = fr["right"]["right_id"].values
    is_audit = np.array([ci[k] in fr["label_rows"] for k in range(len(C))], dtype=bool)
    calib = is_audit & ~np.isnan(y)
    tR = M.tau_reject(fsc[calib], y[calib] > 0.5, ci[calib], eps) if calib.sum() else -np.inf
    reject = (fsc <= tR) & (~in_auto)
    D_mask = ~in_auto & ~reject
    dec = {
        "lenient": load_gate_readonly(fr["cache"], gc["lenient"], C, lid, rid, cfg["ds_prefix"]),
        "neutral": load_gate_readonly(fr["cache"], cfg["neutral"], C, lid, rid, cfg["ds_prefix"]),
    }
    dec_arr = {k: np.array([v.get(p, False) for p in C], dtype=bool) for k, v in dec.items()}
    return D_mask, dec_arr, tR, int(calib.sum()), calib & (y > 0.5), calib


def _ladder(fr, gate_yes, gate_name):
    """固定门类型的嵌套阶梯：单阈值 τ 升序（fused 上 τR 与 τ_g 冗余，合成一个 free-drop 阈值）。
    后一档 M 是前一档子集（τ 只升不降 ⇒ D 只缩不扩），assert 嵌套，保证 miss 在阶梯上单调不降。
    返回 configs（最保守最贵在前 → 最激进最便宜在后）。"""
    fsc = fr["fsc"]; in_auto = fr["in_auto"]
    T = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    ladder = []
    prev = None
    for tau in T:
        reject = (fsc <= tau) & (~in_auto)
        D = ~in_auto & ~reject
        gated = gate_yes & D
        emitted = in_auto | gated
        v_calls = int(D.sum())
        if prev is not None:
            assert np.all(emitted <= prev), f"阶梯非嵌套 @ {gate_name} τ={tau}"
        ladder.append(dict(gate=gate_name, tau=tau, emitted=emitted, v_calls=v_calls))
        prev = emitted
    return ladder


def _precision_cert(fr, emitted, alpha=0.05, rows=None, seed=12345):
    """M 的 teacher 一致性精度：点估计 + 两个下界。

    - 点估计 prec = ΣTP / Σ(TP+FP)，TP/FP 对 teacher（非 human GT），只在审计行子集 rows 内。
    - P_agree（betting）= WSR beta-binomial 单边 betting 下界，**有限样本有效**（把 emitted 标
      注对当 i.i.d. Bernoulli；同 row 聚类被忽略，故另报 P_boot 对照）。
    - P_boot = 审计行 cluster bootstrap 下分位（聚类稳健，但**非**有限样本保证）。
    返回 (点精度 prec, P_agree_betting, P_boot)。
    """
    C = fr["C"]; ci = fr["ci"]; y = fr["y_t"]
    if rows is None:
        rows = sorted(fr["label_rows"])
    rows = sorted(rows)
    rows_set = set(rows)
    labeled = np.array([ci[k] in rows_set for k in range(len(C))], dtype=bool) & ~np.isnan(y)
    row_idx = {r: i for i, r in enumerate(rows)}
    tp = np.zeros(len(rows), dtype=int)
    fp = np.zeros(len(rows), dtype=int)
    for k in np.flatnonzero(labeled & emitted):
        i = row_idx[int(ci[k])]
        if y[k] > 0.5:
            tp[i] += 1
        else:
            fp[i] += 1
    tot_tp = int(tp.sum()); tot_fp = int(fp.sum())
    denom = tot_tp + tot_fp
    prec = tot_tp / denom if denom else 0.0
    if denom == 0:
        return prec, 0.0, 0.0
    p_agree_betting = _betting_lower(tot_tp, denom, alpha)
    rng = np.random.default_rng(seed)
    B = 2000
    n = len(rows)
    idx = rng.integers(0, n, size=(B, n))
    s_tp = tp[idx].sum(axis=1)
    s_fp = fp[idx].sum(axis=1)
    dd = s_tp + s_fp
    boots = np.where(dd > 0, s_tp / np.maximum(dd, 1), np.nan)
    boots = boots[~np.isnan(boots)]
    p_boot = float(np.quantile(boots, alpha)) if boots.size else 0.0
    return prec, p_agree_betting, p_boot


def _split_rows(fr, seed=0):
    """审计行随机两半：A 用于选择（选门/τ），B 用于认证（召回+精度），消除赢家诅咒。"""
    rows = sorted(fr["label_rows"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    half = len(rows) // 2
    A = [rows[i] for i in perm[:half]]
    B = [rows[i] for i in perm[half:]]
    return A, B


def _cert_split(fr, ds, eps_list, alpha, split_seed):
    """一次审计行切分的完整第 3 步计算（无打印）。返回供打印 / 多切分聚合的 dict。

    认证口径（老师锁定）：
    - 召回：**C 内** teacher 正例的行级漏检 miss，CP 上界（betting 在 miss≈0 边界退化，故用精确 CP）。
    - 精度：对 **teacher**（非 human GT）的"验证器一致性"，WSR betting 下界 P_agree + cluster bootstrap P_boot。
    """
    C = fr["C"]; ci = fr["ci"]; y = fr["y_t"]
    gt_set = fr["gt_set"]; nC = len(C)
    cfg, gc = fr["cfg"], fr["gc"]
    lid = fr["left"]["left_id"].values; rid = fr["right"]["right_id"].values

    # teacher 正例行（teacher 认证 oracle，只在审计行上）
    is_audit = np.array([ci[k] in fr["label_rows"] for k in range(nC)], dtype=bool)
    lab = is_audit & ~np.isnan(y)
    tpos = lab & (y > 0.5)
    row_pos = defaultdict(list)
    for k in np.flatnonzero(tpos):
        row_pos[int(ci[k])].append(k)

    # 审计行切 A/B（选择 / 认证）
    A_rows, B_rows = _split_rows(fr, seed=split_seed)
    A_set, B_set = set(A_rows), set(B_rows)
    pos_rows_A = sorted(r for r in row_pos if r in A_set)
    pos_rows_B = sorted(r for r in row_pos if r in B_set)
    n_pos_A, n_pos_B = len(pos_rows_A), len(pos_rows_B)

    # 门决策（只读缓存，零 LLM）
    dec_len = load_gate_readonly(fr["cache"], gc["lenient"], C, lid, rid, cfg["ds_prefix"])
    dec_neu = load_gate_readonly(fr["cache"], cfg["neutral"], C, lid, rid, cfg["ds_prefix"])
    dec_str = load_gate_readonly(fr["cache"], gc["strict"], C, lid, rid, cfg["ds_prefix"])

    def _arr(d):
        return np.array([d.get(p, False) for p in C], dtype=bool)

    gates = {
        "lenient": _arr(dec_len),
        "neutral": _arr(dec_neu),
        "V1V2": _arr(dec_len) & _arr(dec_str),
    }
    n_gates = len(gates)
    alpha_R = alpha / n_gates   # 召回（B 上单配置，Bonferroni 留余量）
    alpha_P = alpha / n_gates   # 精度

    def _miss(cf, pos_rows):
        return sum(1 for r in pos_rows if any(not cf["emitted"][k] for k in row_pos[r]))

    # 每条门一条嵌套阶梯；A 上 miss_ub 用于选择，B 上 miss_ub 用于认证
    gate_ladders = {}
    for gname, gv in gates.items():
        lad = _ladder(fr, gv, gname)
        for cf in lad:
            cf["miss_A"] = _miss(cf, pos_rows_A)
            cf["miss_ub_A"] = _cp_upper(cf["miss_A"], n_pos_A, alpha_R)
            cf["R_cert_A"] = 1.0 - cf["miss_ub_A"]
            cf["miss_B"] = _miss(cf, pos_rows_B)
            cf["miss_ub_B"] = _cp_upper(cf["miss_B"], n_pos_B, alpha_R)
            cf["R_cert_B"] = 1.0 - cf["miss_ub_B"]
            prec_A, _, _ = _precision_cert(fr, cf["emitted"], alpha=alpha_P, rows=A_rows)
            cf["prec_A"] = prec_A
        gate_ladders[gname] = lad

    cand_rec = len(gt_set & set(C)) / len(gt_set) if gt_set else 0.0

    out = {}
    for eps in eps_list:
        # A 上选：召回达标集内按 prec_A 取最大（同精度取更便宜）
        best_cfg = None
        for gname, lad in gate_ladders.items():
            certified = []
            for cf in lad:
                if cf["miss_ub_A"] <= eps:
                    certified.append(cf)
                else:
                    break  # 嵌套 → miss_ub_A 单调不降
            if certified:
                g_best = max(certified, key=lambda c: (c["prec_A"], -c["v_calls"]))
                if best_cfg is None or (g_best["prec_A"], -g_best["v_calls"]) > \
                        (best_cfg["prec_A"], -best_cfg["v_calls"]):
                    best_cfg = g_best
        if best_cfg is None:
            all_cf = [cf for lad in gate_ladders.values() for cf in lad]
            best_R = max(all_cf, key=lambda c: c["R_cert_A"])
            prec_B, p_agree_B, p_boot_B = _precision_cert(fr, best_R["emitted"],
                                                           alpha=alpha_P, rows=B_rows)
            out[eps] = dict(feasible=False, gate=None, tau=None,
                            R_cert_B=best_R["R_cert_B"], P_agree_B=p_agree_B,
                            P_boot_B=p_boot_B, prec_B=prec_B,
                            v_calls=best_R["v_calls"], miss_B=best_R["miss_B"],
                            R_max_A=best_R["R_cert_A"], recall_ok_B=False)
            continue
        # B 上认证选中配置
        prec_B, p_agree_B, p_boot_B = _precision_cert(fr, best_cfg["emitted"],
                                                      alpha=alpha_P, rows=B_rows)
        recall_ok_B = best_cfg["R_cert_B"] >= 1.0 - eps
        matched = set(C[k] for k in np.flatnonzero(best_cfg["emitted"]))
        p, r, f, mtp, mfp, mfn = M._final_metrics(matched, gt_set)
        out[eps] = dict(feasible=True, gate=best_cfg["gate"], tau=best_cfg["tau"],
                        R_cert_B=best_cfg["R_cert_B"], P_agree_B=p_agree_B,
                        P_boot_B=p_boot_B, prec_B=prec_B,
                        v_calls=best_cfg["v_calls"], p=p, r=r, f=f,
                        miss_B=best_cfg["miss_B"], recall_ok_B=recall_ok_B)
    return dict(gate_ladders=gate_ladders, A_rows=A_rows, B_rows=B_rows,
                n_pos_A=n_pos_A, n_pos_B=n_pos_B, cand_rec=cand_rec,
                alpha_R=alpha_R, gates=list(gates), out=out)


def cert_grid(fr, ds, eps_list=(0.10, 0.15, 0.20), alpha=0.05):
    """第 3 步（切分修正）单次切分：打印阶梯 + 每 ε 选中/认证结果（零 LLM 缓存回放）。"""
    res = _cert_split(fr, ds, eps_list, alpha, split_seed=0)
    gate_ladders = res["gate_ladders"]
    print(f"\n  [{ds}] 第3步（切分修正）：A 选门/τ，B 证召回(CP)+精度(betting)（α_R=α_P={res['alpha_R']:.4f}）")
    print(f"    ε_cand(人GT,C外)={res['cand_rec']:.4f}  审计行={len(fr['label_rows'])} "
          f"(A={len(res['A_rows'])},B={len(res['B_rows'])})  正例行 A={res['n_pos_A']} B={res['n_pos_B']} "
          f"门={res['gates']}")

    print(f"    {'门':>7} {'τ':>6} {'升级调用':>8} {'漏检A':>6} {'R_cert_A':>8} {'prec_A':>7}")
    for gname, lad in gate_ladders.items():
        for cf in lad:
            print(f"    {gname:>7} {cf['tau']:>6.2f} {cf['v_calls']:>8,} {cf['miss_A']:>6} "
                  f"{cf['R_cert_A']:>8.4f} {cf['prec_A']:>7.4f}")

    hdr = (f"    {'ε':>5} | {'选中门':>7} {'τ':>6} | {'R_cert_B':>8} {'P_agree_B':>8} {'升级调用':>8} "
           f"| {'humanP':>6} {'humanR':>6} {'humanF1':>6}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    for eps in eps_list:
        o = res["out"][eps]
        if not o["feasible"]:
            print(f"    {eps:>5.2f} | {'不可行':>7} {'—':>6} | {o['R_cert_B']:>8.4f} {o['P_agree_B']:>7.4f} "
                  f"{o['v_calls']:>8,} |  (A上R_max={o['R_max_A']:.4f}, B上P_agree={o['P_agree_B']:.4f}, "
                  f"P_boot={o['P_boot_B']:.4f})")
            continue
        flag = "" if o["recall_ok_B"] else "  ✗B召回不达标(选门未泛化)"
        print(f"    {eps:>5.2f} | {o['gate']:>7} {o['tau']:>6.2f} | "
              f"{o['R_cert_B']:>8.4f} {o['P_agree_B']:>7.4f} {o['v_calls']:>8,} | "
              f"{o['p']:>6.4f} {o['r']:>6.4f} {o['f']:>6.4f}{flag}  "
              f"(点prec={o['prec_B']:.4f}, P_boot={o['P_boot_B']:.4f})")
    return res["out"]


def cert_grid_multisplit(fr, ds, eps_list, alpha=0.05, n_splits=20):
    """多切分（伪 seed）选门一致性：同一批审计行随机切 A/B 跑 n_splits 次，看每个 ε 的选中 (门,τ)
    稳不稳、B 上认证是否泛化。零 LLM。"""
    print(f"\n  [{ds}] 多切分选门一致性（n={n_splits} 次随机切分，零 LLM）")
    dist = {eps: defaultdict(int) for eps in eps_list}
    feas = {eps: 0 for eps in eps_list}
    recall_ok = {eps: 0 for eps in eps_list}
    p_agree = {eps: [] for eps in eps_list}
    for s in range(n_splits):
        res = _cert_split(fr, ds, eps_list, alpha, split_seed=s)
        for eps in eps_list:
            o = res["out"][eps]
            if o["feasible"]:
                feas[eps] += 1
                dist[eps][f"{o['gate']}|τ={o['tau']:.2f}"] += 1
                if o["recall_ok_B"]:
                    recall_ok[eps] += 1
                p_agree[eps].append(o["P_agree_B"])
    print(f"    {'ε':>5} | {'可行':>6} | {'B召回达标':>8} | 选中(门,τ) 分布（前若干）")
    for eps in eps_list:
        d = ", ".join(f"{k}:{v}" for k, v in sorted(dist[eps].items(), key=lambda x: -x[1]))
        if not d:
            d = "(全不可行)"
        med = np.median(p_agree[eps]) if p_agree[eps] else float("nan")
        print(f"    {eps:>5.2f} | {feas[eps]:>3}/{n_splits} | {recall_ok[eps]:>3}/{n_splits}   | {d}")
        print(f"           P_agree_B 中位数={med:.4f}  (n={len(p_agree[eps])})")
    return dict(feas=feas, recall_ok=recall_ok, dist=dist)


def diag_t3(fr, ds):
    """T3 门控支配条件：gated-lenient 能否在 (a,b) 上同时压过 neutral。"""
    C = fr["C"]; ci = fr["ci"]; fsc = fr["fsc"]; y = fr["y_t"]
    D_mask, dec_arr, tR, teacher_vol, tpos, calib = _dispute(fr, 0.05)
    dk = np.flatnonzero(D_mask)
    if len(dk) == 0:
        print(f"\n  [{ds}] 争议队列为空，跳过 T3")
        return
    yy = y[dk]; ss = fsc[dk]
    len_yes = dec_arr["lenient"][dk]
    neu_yes = dec_arr["neutral"][dk]
    pos = yy > 0.5; neg = ~pos
    pi = float(pos.mean())
    a_len = float(len_yes[pos].mean()) if pos.any() else float("nan")
    b_len = float(len_yes[neg].mean()) if neg.any() else float("nan")
    a_neu = float(neu_yes[pos].mean()) if pos.any() else float("nan")
    b_neu = float(neu_yes[neg].mean()) if neg.any() else float("nan")
    print(f"\n  [{ds}] T3 门控支配条件（D 上 teacher oracle，π={pi:.3f}，|D|={len(dk):,}）")
    print(f"    lenient a={a_len:.3f} b={b_len:.3f}    neutral a={a_neu:.3f} b={b_neu:.3f}")
    print(f"    {'q(τ_g)':>8} {'τ_g':>6} {'a_g':>6} {'b_g':>6} {'prec':>6} {'调占比':>7} {'支配neutral':>11}")
    for q in [None, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]:
        tg = None if q is None else float(np.quantile(ss, q))
        g = len_yes if tg is None else (len_yes & (ss >= tg))
        a_g = float(g[pos].mean()) if pos.any() else float("nan")
        b_g = float(g[neg].mean()) if neg.any() else float("nan")
        denom = a_g * pi + b_g * (1 - pi)
        prec = a_g * pi / denom if denom > 0 else 0.0
        call_frac = 1.0 if q is None else 1.0 - q
        dom = (a_g >= a_neu) and (b_g <= b_neu)
        qs = "∞" if q is None else f"{q:.2f}"
        tgs = "∞" if tg is None else f"{tg:.3f}"
        print(f"    {qs:>8} {tgs:>6} {a_g:>6.3f} {b_g:>6.3f} {prec:>6.4f} {call_frac:>7.3f} "
              f"{'✓' if dom else '—':>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", choices=["calib", "cardinality", "t3", "both"], default=None)
    ap.add_argument("--cert", action="store_true", help="T2/T4 认证网格 + LTT（零 LLM，缓存回放）")
    ap.add_argument("--datasets", nargs="+",
                    default=["products", "company", "quora", "dblp_acm_sample"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.10, 0.15, 0.20],
                    help="认证漏检目标列表（默认 0.10 0.15 0.20）")
    ap.add_argument("--splits", type=int, default=0,
                    help="多切分选门一致性次数（默认 0 关闭；如 20）")
    args = ap.parse_args()

    if args.diag is None and not args.cert:
        args.diag = "both"

    for ds in args.datasets:
        pre = M.prepare(ds, args.seed, 250)
        fr = _frame(pre)
        if args.diag in ("calib", "both"):
            diag_calibration(fr, ds)
        if args.diag in ("cardinality", "both"):
            diag_cardinality(fr, ds)
        if args.diag == "t3":
            diag_t3(fr, ds)
        if args.cert:
            cert_grid(fr, ds, eps_list=list(args.eps))
        if args.splits > 0:
            cert_grid_multisplit(fr, ds, eps_list=list(args.eps), n_splits=args.splits)


if __name__ == "__main__":
    main()
