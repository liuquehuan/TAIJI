# -*- coding: utf-8 -*-
"""run_hybrid_join_v10_6_old.py — 结构分桶候选集 + p_A 主流程（auto 为对照）+ V1+V2 升级门。

本次改动（相对上一版）：
  1. 分桶信号换成「数据本身的结构」，不再用跨表匹配分数：
       --bucket-mode competition   旧口径：dense/sparse 的候选数 + top1−top2 margin（matcher 困惑度）
       --bucket-mode structure     默认结构口径：v10.6 的 12 轴 Structural Complexity Index
                                   （textual 5 / schema 4 / identifier 3，轴内均匀、三轴等权）
       --bucket-mode structure+hub 12 轴 q 再与同侧 hub / dup 两条轴等权平均
       --bucket-mode poverty       旧版 3 信号（token_count / alphanum / rare_token），A/B 用
       --bucket-mode poverty+hub   旧版 7 信号（再叠分辨力 I(r)、歧义度 fanout、hubness、近重复簇）
     12 轴实现见 fdj_cx12.py（自 v10.6 原样搬来），旧口径见 fdj_struct_buckets.py；
     全部按池化分位数切桶（无自由阈值），都不碰 matcher 分数 / GT / LLM。
  2. 阈值与接受层：p_A 升为主流程（每桶交叉拟合 logistic + Clopper–Pearson 下界 ≥ floor 选 τA），
     auto（mutual-NN 免费接受）降为对照。--pA-order struct 用结构先验定 fixed-sequence 的顺序。
  3. **V2 召回回滚**：判据 = **最终 GT 召回**（按用户决定，这一格允许读 human GT）。
     引入 V2 后召回回落超过 --rb-margin（默认 0.05）才退回 neutral；召回持平/上升/小幅
     回落都保留 V2。只看召回、不看 F1。
       V1V2rb    回滚后的结果（rb 与 rbG 同值，rbG 保留为别名）。
       V1V2rbG   同上。
     keep_n/keep_s（teacher 正例上的留存量）只作诊断打印：它继承 teacher 的假阳率，
     当初拿它当代理时在 acm 上白丢 0.09 F1（keep_s=0.185 但 GT 召回不降）。
     成本侧仍然省：回滚掉的对上，未标注那部分 strict 调用不用调。
  4. τR 的证书单元可选 --certunit row（按行，旧口径）/ comp（按候选图连通分量，处理两端依赖）。
  5. 诊断新增：桶纯度（teacher 正例率跨桶方差）与认证覆盖（各桶在 floor 下能认证出多少对）。

桶优先候选集不变：C1(LLM) 行 → --c1-k（默认 1），其余行 → K=10；行∪列对称。
升级带依次过 V1(lenient) → V2(strict)，只保留双通过的对；缺缓存的真调中转站 LLM 补齐。

口径（写死不改）：三桶 C1/C2/C3；阈值只用行采样 + teacher 标签（缓存回放）；human GT 只进最终评测。

用法：
  python run_hybrid_join_v10_6_old.py                      # 四个数据集 × 三种分桶口径
  python run_hybrid_join_v10_6_old.py --bucket-mode structure+hub --datasets products
  python run_hybrid_join_v10_6_old.py --accept auto        # 只跑对照
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta_dist
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fdj_struct_buckets as sb          # noqa: E402
import fdj_cx12 as cx                     # noqa: E402
import run_hybrid_join_v10_10 as v10      # noqa: E402
import run_hybrid_join_v12 as v12         # noqa: E402

FLOOR = 0.90          # p_A 接受层要求：teacher 精度 Clopper–Pearson 下界
PA_ALPHA = 0.05
RB_MARGIN = 0.05      # V2 回滚：最终 GT 召回**回落超过**这个幅度才回滚（判据读 human GT）

# ── token 账本 ───────────────────────────────────────────────────────────────
# 缓存只存 'yes'/'no'，不存 usage；所以按 prompt 原文重建 + 分词估算。
# 编码器用 tiktoken cl100k_base 代理 Qwen 分词（无本地 Qwen tokenizer，±10%）。
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _ntok(s):
    if _ENC is not None:
        return len(_ENC.encode(str(s)))
    return max(1, len(str(s)) // 3)


def _teacher_prompt(lname, ldesc, rname, rdesc):
    """v10.8 join_c1_llm 的 post_filter='teacher' 原文（缓存里的 teacher 标签由它产生）。"""
    return ("Are these the same product? Same brand and model, OR one is a fuller/"
            "terser listing of the same item = Yes. Different brand, model, or "
            "category = No.\n"
            f"Left: {str(lname)} — {str(ldesc)[:300]}\n"
            f"Right: {str(rname)} — {str(rdesc)[:300]}\n"
            "Answer Yes or No.")


def _neutral_prompt(lname, ldesc, rname, rdesc):
    """v10.8 join_c1_llm 的 post_filter='neutral' 原文。

    注意 _gate_prompt 是 if-lenient-else-strict，不能拿它算 neutral。
    """
    return ("A cheap automated matcher flagged these two products as a possible "
            "match, but with low confidence. Judge whether they are the same product.\n"
            f"Left: {str(lname)}\n{str(ldesc)[:300]}\n"
            f"Right: {str(rname)}\n{str(rdesc)[:300]}\n"
            "Answer Yes if they describe the same product — same brand and model, "
            "or one is a fuller or terser listing of the same item. Answer No if "
            "they are different products (different brand, model, or product "
            "category). Answer Yes or No.")


def _token_bill(pairs, kind, left, right, dec):
    """pairs 上 kind 档（teacher/neutral/lenient/strict）的 prompt+completion token。

    completion 按缓存的真实回答分词（'yes'/'no' 各 1 token），未命中按 'no' 计。
    """
    l_name = left["left_name"].values; l_desc = left["left_description"].values
    r_name = right["right_name"].values; r_desc = right["right_description"].values
    p_tok = c_tok = 0
    for (i, j) in pairs:
        if kind == "teacher":
            msg = _teacher_prompt(l_name[i], l_desc[i], r_name[j], r_desc[j])
        elif kind == "neutral":
            msg = _neutral_prompt(l_name[i], l_desc[i], r_name[j], r_desc[j])
        else:
            msg = _gate_prompt(kind, l_name[i], l_desc[i], r_name[j], r_desc[j])
        p_tok += _ntok(msg)
        c_tok += _ntok("yes" if dec.get((i, j)) else "no")
    return dict(prompt=p_tok, completion=c_tok, total=p_tok + c_tok, calls=len(pairs))


def _bill_add(*bills):
    out = dict(prompt=0, completion=0, total=0, calls=0)
    for b in bills:
        for k in out:
            out[k] += b[k]
    return out

DATASET_CFG = {
    "products":        dict(n_target=None, cache=".llm_decision_cache.json", ds_prefix="products",
                            neutral="openai/qwen-plus|neutral|v6", teacher="openai/qwen-plus|teacher|v6"),
    "company":         dict(n_target=200, cache=".llm_decision_cache_company.json", ds_prefix=None,
                            neutral="openai/qwen-plus|neutral|v7", teacher="openai/qwen-plus|teacher|v7"),
    "quora":           dict(n_target=500, cache=".llm_decision_cache_quora_v6.json", ds_prefix=None,
                            neutral="openai/qwen-plus|neutral|v6-quora", teacher="openai/qwen-plus|teacher|v8-quora"),
    "dblp_acm_sample": dict(n_target=1000, cache=".llm_decision_cache_dblp_acm_v6.json", ds_prefix=None,
                            neutral="openai/qwen-plus|neutral|v6", teacher="openai/qwen-plus|teacher|v6"),
    # biodex_500：data/biodex_500 已按 load_data 同口径预采样成 500×500（GT-seeded），
    # n_target 设大到禁掉二次采样；全部决策都要真调 LLM（无历史缓存）。
    "biodex_500":      dict(n_target=5000, cache=".llm_decision_cache_biodex_500.json", ds_prefix=None,
                            neutral="openai/qwen-plus|neutral|v1-biodex", teacher="openai/qwen-plus|teacher|v1-biodex"),
}

# 升级门两段：lenient 复用旧缓存（已很全），strict 用新 tag v8（本次新 prompt，真调 LLM 补齐）。
GATE_CFG = {
    "products":        dict(model="openai/qwen-plus", lenient="openai/qwen-plus|lenient|v6",
                            strict="openai/qwen-plus|strict|v8"),
    "company":         dict(model="openai/qwen-plus", lenient="openai/qwen-plus|lenient|v7",
                            strict="openai/qwen-plus|strict|v8"),
    "quora":           dict(model="openai/qwen-plus", lenient="openai/qwen-plus|lenient|v6-quora",
                            strict="openai/qwen-plus|strict|v8-quora"),
    "dblp_acm_sample": dict(model="openai/qwen-plus", lenient="openai/qwen-plus|lenient|v6",
                            strict="openai/qwen-plus|strict|v8"),
    "biodex_500":      dict(model="openai/qwen-plus", lenient="openai/qwen-plus|lenient|v1-biodex",
                            strict="openai/qwen-plus|strict|v1-biodex"),
}

K = 10
B1 = 0.05
B2 = 0.30
MIN_EVAL_ROWS = 40
CV_FOLDS = 5
GATE_BATCH = 40
GATE_WORKERS = 8       # 升级门的在途请求数上限（见 _llm_gate._ask 的说明）


# ────────────────────────────────────────────────────────────────────────────
# τR：行级精确认证（Clopper–Pearson），自下而上 fixed-sequence
# ────────────────────────────────────────────────────────────────────────────
def tau_reject(s, y, rows, eps, alpha=0.05):
    pos = y > 0.5
    rp = np.unique(rows[pos])
    if len(rp) == 0:
        return -np.inf
    mins = np.sort([s[(rows == r) & pos].min() for r in rp])
    n = len(mins)

    def ub(k):
        if k >= n:
            return 1.0
        return float(_beta_dist.ppf(1 - alpha, k + 1, n - k))

    ks = [j for j in range(n + 1) if ub(j) <= eps]
    k = max(ks) if ks else 0
    return float(mins[0] - 1e-9) if k == 0 else float(mins[k - 1])


BIG_RANK = 1 << 20


def cap_bucket_pairs(idx, score, ci, cap):
    """把某桶的标注对压到 cap 以内：先保**每行 top-1**（按该桶 matcher 分），
    再按分数降序补满到 cap。

    为什么要先保每行 top-1：p_A 的 fixed-sequence 证书只看排序顶端，顶端那些对
    必须被标到；纯随机抽会把顶端抽空，认证直接崩（见 `fixed_sequence_tau`）。
    这一步是确定性的（按分数降序 + 行内去重），没有随机种子。
    """
    if len(idx) <= cap:
        return idx
    order = idx[np.argsort(-np.asarray(score)[idx], kind="mergesort")]
    seen, keep = set(), []
    for t in order:                       # 第一轮：每行保该行分数最高的那一对
        if ci[t] not in seen:
            seen.add(ci[t]); keep.append(t)
    if len(keep) < cap:
        kept = set(keep)
        for t in order:
            if len(keep) >= cap:
                break
            if t not in kept:
                kept.add(t); keep.append(t)
    return np.array(sorted(keep), dtype=int)


def _int_ranks(S, row_cand, col_cand, nL, nR):
    """整数名次：0 = 该视图内 top-1。取**行视图与列视图里较优的那个**。

    候选集是 row_cand ∪ col_cand，所以一个对可以只在列候选里而不在行候选里。
    只看行视图会给这种对填 BIG_RANK（= 名次无穷差），于是一条这样的标注正例就能把
    门槛顶到 BIG_RANK、让整个名次规则**拒不掉任何东西**（acm 上实测就是这个）。
    取 min(行视图, 列视图) 才是"这对在任一视图里有多靠前"的正确定义。
    """
    rk = np.full((nL, nR), BIG_RANK, dtype=np.int32)
    for i in range(nL):
        js = sorted(row_cand[i], key=lambda j: S[i, j], reverse=True)
        for pos, j in enumerate(js):
            rk[i, j] = pos
    for j in range(nR):
        is_ = sorted(col_cand[j], key=lambda i: S[i, j], reverse=True)
        for pos, i in enumerate(is_):
            if pos < rk[i, j]:
                rk[i, j] = pos
    return rk


def certified_rank_q(rank_vals, y, rows, eps, alpha=0.05):
    """认证整数名次门槛 q：「rank > q ⇒ 免费拒绝」在标注正例上丢行率的 CP 上界 ≤ eps。

    与 `tau_reject` 同框架（行级 Clopper–Pearson，自下而上），只是把「分数阈值」换成
    「名次门槛」——名次是整数、行内可比，不吃分数尺度。

    单调性（容易搞反，写清楚）：拒绝条件是 `rank > q`，**q 越大拒得越少**。
      每个含正例的行 r 记 m_r = 该行正例里**最差**的名次（q 缩小时第一个丢的就是它），
      m 升序排列。丢的行数 L(q) = #{r : m_r > q}，随 q 减小而增大；
      L(q) ≤ k ⟺ q ≥ m_(n−k)  ⇒  最大化拒绝 = 取**最小**可行 q = m_(n−k)。
      k 仍取满足 ub(k) ≤ eps 的最大值（与 `tau_reject` 一致，都取最大 k）。

    返回 (q, n, k, best)。n=0（无标注正例）或 k=0 时 q = m_(n) = 标注正例里最差的名次，
    **只拒比它更差的**，是保守侧；没有任何 k 可行时同样退回 k=0（保守，不硬拒）。
    """
    pos = y > 0.5
    rp = np.unique(rows[pos])
    if len(rp) == 0:
        return BIG_RANK, 0, 0, 0.0
    m = np.sort([rank_vals[(rows == r) & pos].max() for r in rp])
    n = len(m)

    def ub(k):
        if k >= n:
            return 1.0
        return float(_beta_dist.ppf(1 - alpha, k + 1, n - k))

    ks = [j for j in range(n + 1) if ub(j) <= eps]
    k = max(ks) if ks else 0
    if k >= n:
        k = 0                     # 退化为保守：只拒比最差正例更差的
    return int(m[n - 1 - k]), n, k, float(ub(k))


def _final_metrics(matched, gt_set):
    mtp = len(matched & gt_set)
    mfp = len(matched) - mtp
    mfn = len(gt_set) - mtp
    p = mtp / len(matched) if matched else 0.0
    r = mtp / len(gt_set) if gt_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f, mtp, mfp, mfn


def _bucket_of(ci, cj, lrow_class, rrow_class):
    _prio = {"C1": 0, "C2": 1, "C3": 2}
    out = np.empty(len(ci), dtype=int)
    for k, (i, j) in enumerate(zip(ci, cj)):
        lc = lrow_class.get(i, "C2"); rc = rrow_class.get(j, "C2")
        out[k] = _prio[min([lc, rc], key=lambda c: _prio.get(c, 99))]
    return out


# ────────────────────────────────────────────────────────────────────────────
# 桶优先候选集 + V1/V2 升级门
# ────────────────────────────────────────────────────────────────────────────
def build_candidates_per_row_full(dense, sparse, kL, kR):
    """候选集 = 每行/每列按各自预算 top-k（dense∪sparse），行∪列对称。

    返回 (row_cand, col_cand, cand)。row_cand/col_cand 供 p_A 的 rank/margin 特征用。
    """
    nL, nR = dense.shape
    row_cand = [set() for _ in range(nL)]
    col_cand = [set() for _ in range(nR)]
    for S, budgets, sink in ((dense, kL, row_cand), (sparse, kL, row_cand)):
        for r in range(nL):
            kk = min(int(budgets[r]), nR - 1)
            if kk > 0:
                sink[r].update(int(x) for x in np.argpartition(-S[r], kk - 1)[:kk])
    for S, budgets, sink in ((dense, kR, col_cand), (sparse, kR, col_cand)):
        for j in range(nR):
            kk = min(int(budgets[j]), nL - 1)
            if kk > 0:
                sink[j].update(int(x) for x in np.argpartition(-S[:, j], kk - 1)[:kk])
    cand = set()
    for i in range(nL):
        for j in row_cand[i]:
            cand.add((i, j))
    for j in range(nR):
        for i in col_cand[j]:
            cand.add((i, j))
    return row_cand, col_cand, cand


def build_candidates_per_row(dense, sparse, kL, kR):
    """兼容旧接口（_probe_call_volume.py 在用）。"""
    return build_candidates_per_row_full(dense, sparse, kL, kR)[2]


# ────────────────────────────────────────────────────────────────────────────
# p_A 接受层：每桶交叉拟合 logistic + Clopper–Pearson 下界 ≥ floor 选 τA
# ────────────────────────────────────────────────────────────────────────────
def crossfit_oof(X, y, folds=CV_FOLDS, seed=0):
    """K 折 out-of-fold 概率：τA 的选择不能用样本内拟合分，否则 CP 下界无效。"""
    n = len(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    oof = np.full(n, np.nan)
    for k in range(folds):
        te = perm[k::folds]
        tr = np.setdiff1d(perm, te)
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        mu = X[tr].mean(axis=0); sd = X[tr].std(axis=0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return oof


def _feasibility_n(alpha_step, floor):
    """认证 floor 需要的「顶端连续全正对数」下界：Beta(α_step, n, 1) = α_step^(1/n) ≥ floor
    ⟺ n ≥ ln α_step / ln floor。返回 ceil。这是 floor=0.90 ⇔ 29 连正的显式形式，
    α 越小（spending 越狠）需要的连正越多。"""
    if not (0 < floor < 1) or not (0 < alpha_step < 1):
        return 0
    return int(math.ceil(math.log(alpha_step) / math.log(floor)))


def fixed_sequence_tau(scores, labels, floor, alpha=PA_ALPHA, spending="uniform"):
    """pair 级 fixed-sequence：自高到低扫 score 阈值，取 CP 下界 ≥ floor 的最小接受集。

    正确性修复：旧版在 m 个切点里各用同一个 α 挑最优（hit[-1]），没为多重检验付 α，
    CP 覆盖是乐观的。这里把 α 均摊到 m 个不同分数层（α_j = α/m），并集界 ⇒ 联合覆盖
    ≥ 1−α，任意数据驱动选点都有效。

    关于「首败即停」：对**精度下界**不适用——小集合的 LCB 天然 < floor（1 个样本证不出
    90% 精度），大集合又可能被负例稀释，所以"首败即停"在两端都会"首点即失败"。等价而
    正确的修正是 α-spending + 并集界（本实现），选点规则保持 hit[-1]（最小接受集）。

    返回 (tau, n, k, best_lcb, n_needed)。n_needed = 该 floor×spending 下的连正需求，
    可行性判据：n_pos < n_needed ⇒ 无论怎么排都认证不到 floor。
    """
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    f = np.isfinite(s); s, y = s[f], y[f]
    if s.size == 0:
        return float("nan"), 0, 0, 0.0, 0
    order = np.argsort(s)
    ss, ys = s[order], y[order]
    uniq, inv = np.unique(ss, return_inverse=True)
    pos_by_u = np.bincount(inv, weights=ys.astype(float), minlength=len(uniq))
    tot_by_u = np.bincount(inv, minlength=len(uniq))
    pos_acc = np.cumsum(pos_by_u[::-1])[::-1]
    tot_acc = np.cumsum(tot_by_u[::-1])[::-1]
    m = len(uniq)
    alpha_j = alpha / m if spending == "uniform" else alpha
    n_needed = _feasibility_n(alpha_j, floor)
    lcb = np.empty(m)
    best = 0.0
    for j in range(m):
        n = int(tot_acc[j]); k = int(pos_acc[j])
        lcb[j] = float(_beta_dist.ppf(alpha_j, k, n - k + 1)) if (n and k) else 0.0
        best = max(best, lcb[j])
    hit = np.where(lcb >= floor)[0]
    if hit.size == 0:
        return float("nan"), 0, 0, float(best), n_needed
    t = int(hit[-1])
    return float(uniq[t]), int(tot_acc[t]), int(pos_acc[t]), float(lcb[t]), n_needed


def row_fixed_sequence_tau(scores, labels, rows, floor, alpha=PA_ALPHA, spending="uniform"):
    """行级 fixed-sequence：统计单元是**行**（每行最多一次），不是 pair。

    同一行多个 pair 强相关，pair 级计数把一行当多次独立观测，CP 下界偏乐观（τR 做了
    行级、p_A 没做）。这里按行聚合：对每个含标注对的行 r，
      a_r = 该行标注对里的最高分（≥τ ⇒ 该行被接受）
      p_r = 该行正例标注对里的最高分（≥τ ⇒ 该行被正确接受）
    在阈值 τ 上 n_rows = #{a_r≥τ}、k_rows = #{p_r≥τ}，行精度 = k_rows/n_rows，CP 照旧。
    α-spending 与 pair 级同口径（α/m）。返回 (tau, n_rows, k_rows, best_lcb, n_needed)。
    """
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    rr = np.asarray(rows, int)
    f = np.isfinite(s)
    s, y, rr = s[f], y[f], rr[f]
    if s.size == 0:
        return float("nan"), 0, 0, 0.0, 0
    A = {}; P = {}
    for i in range(len(s)):
        r = int(rr[i])
        v = float(s[i])
        A[r] = v if r not in A else max(A[r], v)
        if y[i]:
            P[r] = v if r not in P else max(P[r], v)
    keys = sorted(A)
    a_vals = np.array([A[r] for r in keys])
    p_vals = np.array([P.get(r, -np.inf) for r in keys])
    # 候选阈值（高→低）；只扫 a_vals ∪ p_vals 里的值，集合成员在这些点才变
    uniq_tau = np.unique(np.concatenate([a_vals, p_vals[np.isfinite(p_vals)]]))[::-1]
    m = len(uniq_tau)
    alpha_j = alpha / m if spending == "uniform" else alpha
    n_needed = _feasibility_n(alpha_j, floor)
    best = 0.0
    passing = []
    for tau in uniq_tau:
        n = int((a_vals >= tau).sum())
        k = int((p_vals >= tau).sum())
        lcb = float(_beta_dist.ppf(alpha_j, k, n - k + 1)) if (n and k) else 0.0
        best = max(best, lcb)
        if lcb >= floor:
            passing.append((float(tau), n, k, lcb))
    if not passing:
        return float("nan"), 0, 0, float(best), n_needed
    tau, n, k, lcb = passing[0]     # 高→低里第一个通过 = 最小接受集（最高阈值）
    return tau, n, k, lcb, n_needed


def _gate_prompt(post_filter, lname, ldesc, rname, rdesc):
    ln = str(lname); rn = str(rname)
    ld = str(ldesc)[:300]; rd = str(rdesc)[:300]
    if post_filter == "lenient":
        return (f"These two records were matched by a high-recall automated matching system. "
                f"They are LIKELY the same entity.\n"
                f"Left: {ln}\n{ld}\nRight: {rn}\n{rd}\n"
                f"Only reject if they are CLEARLY DIFFERENT entities. "
                f"When in doubt, answer Yes. Answer Yes or No.")
    else:  # strict
        return (f"These two records survived a lenient automated matching filter. "
                f"Now verify carefully: are they genuinely the SAME entity?\n"
                f"Left: {ln}\n{ld}\nRight: {rn}\n{rd}\n"
                f"A match means the same underlying real-world entity. Reject if clearly different. "
                f"If the descriptions describe the same entity just with different wording, answer Yes. "
                f"Answer Yes or No.")


def _llm_gate(pairs, left, right, post_filter, model, tag, cache, cache_path, ds_prefix,
              batch_size=GATE_BATCH, max_rounds=3):
    """对 (i,j) 索引对做 lenient/strict 裁决。先查缓存，缺的真调 LLM 写回。

    返回 (decisions: {pair: bool}, n_cache_hit, n_new_call)。

    失败处理（重要）：litellm.batch_completion 对**逐条**失败不抛异常，而是在返回列表里
    塞 Exception 对象。旧版把这些当 'no' 写进缓存 —— 一次网络抖动就会把真匹配永久钉死成
    非匹配，且事后无法从缓存分辨。现在：失败的对**一律不写缓存**，并在 max_rounds 内重试；
    仍失败则本轮按 'no' 处理（保守）但仍不入缓存，下次运行会自动重来。
    """
    import litellm
    l_id = left["left_id"].values; r_id = right["right_id"].values
    l_name = left["left_name"].values; l_desc = left["left_description"].values
    r_name = right["right_name"].values; r_desc = right["right_description"].values
    prefix = f"{ds_prefix}|" if ds_prefix else ""

    decisions = {}
    uncached = []
    n_hit = 0
    for (i, j) in pairs:
        k = f"{prefix}{tag}|{l_id[i]}|{r_id[j]}"
        d = cache.get(k)
        if d in ("yes", "no"):
            decisions[(i, j)] = (d == "yes")
            n_hit += 1
        else:
            uncached.append((i, j, k))

    n_batch = max(1, (len(uncached) + batch_size - 1) // batch_size)
    n_new = 0
    n_fail = 0
    err_msgs = {}

    def _ask(todo, round_no):
        """一轮调用，返回成功解析的 {k: 'yes'/'no'} 与仍失败的下标。

        用固定并发（GATE_WORKERS）的线程池逐条 litellm.completion，不用
        litellm.batch_completion：后者把整批同时打出去（batch_size=40），中转站在
        持续压力下会限流 → 重试风暴，并且实测会卡死（输出冻结、socket 不返回）。
        限制在途请求数后同样的量能稳定跑完。
        """
        from concurrent.futures import ThreadPoolExecutor

        def _one(item):
            i, j, k = item
            msg = [{"role": "user", "content": _gate_prompt(
                post_filter, l_name[i], l_desc[i], r_name[j], r_desc[j])}]
            try:
                r = litellm.completion(model=model, messages=msg, timeout=60,
                                       max_tokens=20, num_retries=3)
                ans = r.choices[0].message.content.strip().lower()
                return k, ("yes" if "yes" in ans else "no"), None
            except Exception as e:
                return k, None, f"{type(e).__name__}: {e}"[:200]

        got, failed = {}, []
        done = 0
        with ThreadPoolExecutor(max_workers=GATE_WORKERS) as ex:
            for k, v, err in ex.map(_one, todo):
                done += 1
                if v is None:
                    err_msgs[err] = err_msgs.get(err, 0) + 1
                    failed.append(k)
                else:
                    got[k] = v
                if n_batch > 1 and done % (batch_size * 5) == 0:
                    print(f"    [gate:{post_filter}] r{round_no} 进度 {done:,}/{len(todo):,}",
                          flush=True)
        kmap = {k: (i, j) for (i, j, k) in todo}
        return got, [(kmap[k][0], kmap[k][1], k) for k in failed]

    todo = list(uncached)
    for rnd in range(1, max_rounds + 1):
        if not todo:
            break
        got, todo = _ask(todo, rnd)
        for k, v in got.items():
            cache[k] = v
        n_new += len(got)
        if todo and rnd < max_rounds:
            print(f"    [gate:{post_filter}] 第 {rnd} 轮有 {len(todo):,} 条失败，重试 ...",
                  flush=True)
            time.sleep(2.0 * rnd)

    if err_msgs:
        print(f"    [gate:{post_filter}] 错误样本（前 3 类）：", flush=True)
        for m, c in sorted(err_msgs.items(), key=lambda x: -x[1])[:3]:
            print(f"       ×{c}  {m}", flush=True)

    n_fail = len(todo)
    if n_fail:
        print(f"    [gate:{post_filter}] 警告：{n_fail:,} 条最终失败，"
              f"本轮按 'no' 处理且**未写入缓存**（正式口径请重跑到 n_fail=0）", flush=True)
    for (i, j, k) in todo:
        decisions[(i, j)] = False
    for (i, j, k) in uncached:
        if k in cache:
            decisions[(i, j)] = (cache[k] == "yes")

    if n_new:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    return decisions, n_hit, n_new


# ────────────────────────────────────────────────────────────────────────────
# 准备（与分桶口径无关）→ 分桶 → 跑一档
# ────────────────────────────────────────────────────────────────────────────
def _components(ci, cj, nL):
    """候选二部图的连通分量号（τR 的证书单元）。

    pair 同时挂在左行 i 和右列 j 上，按行做 CP 会漏掉列侧依赖；按分量做
    （簇稳健推断）才能同时覆盖两侧。极端 hub 行会把大量行并成一个分量，
    所以分量数太少时 CP 会失效 —— 这一点由 --certunit row 作对照暴露。
    """
    if len(ci) == 0:
        return np.zeros(0, dtype=int)
    n = nL + int(cj.max()) + 1
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(ci.tolist(), cj.tolist()):
        ra, rb = find(i), find(nL + j)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in ci.tolist()], dtype=int)


def prepare(ds, seed, m_req, k_hub=10, dup_thresh=0.5):
    """加载数据 / 分数矩阵 / 结构信号 / 行采样。分桶口径与接受层都在这之后决定。"""
    cfg = DATASET_CFG[ds]
    if cfg["n_target"] is not None:
        v10.N_TARGET = cfg["n_target"]

    model = v10.load_embedding_model()
    left, right, gt = v10.load_data(ds)
    l_idx = {str(v): k for k, v in enumerate(left["left_id"])}
    r_idx = {str(v): k for k, v in enumerate(right["right_id"])}
    e_l = np.array([l_idx[str(a)] for a in gt["left_id"]])
    e_r = np.array([r_idx[str(b)] for b in gt["right_id"]])
    gt_set = set(zip(e_l.tolist(), e_r.tolist()))

    fused, dense, sparse = v10.fused_score(left, right, model, w=0.5)
    nL, nR = dense.shape

    # 结构信号：两表互相当参照（池化 df），同侧算，不碰任何跨表得分
    sigL = sb.compute_signals(left["left_description"].values,
                              right["right_description"].values,
                              k_hub=k_hub, dup_thresh=dup_thresh)
    sigR = sb.compute_signals(right["right_description"].values,
                              left["left_description"].values,
                              k_hub=k_hub, dup_thresh=dup_thresh)
    # v10.6 的 12 轴 Structural Complexity Index（行级、无标签、无 matcher 分）
    lf = cx.compute_row_features([str(x) for x in left["left_description"].values])
    rf = cx.compute_row_features([str(x) for x in right["right_description"].values])
    q12_L = cx.compute_structural_complexity(lf, reference_features=rf)
    q12_R = cx.compute_structural_complexity(rf, reference_features=lf)
    q_comp_L, q_comp_R = v12.competition_degree(dense, sparse, fused, K)

    # 行采样：口径无关 —— 采样框是全部左行（deploy 帧），不是某口径下的候选行，
    # 否则换分桶口径会连带换掉校准集，三个口径就不可比了。
    m = int(min(m_req, max(1, nL - MIN_EVAL_ROWS)))
    label_rows = set(np.random.RandomState(seed).permutation(nL)[:m].tolist())

    return dict(ds=ds, cfg=cfg, gc=GATE_CFG[ds], left=left, right=right, gt_set=gt_set,
                dense=dense, sparse=sparse, fused=fused, nL=nL, nR=nR,
                sigL=sigL, sigR=sigR, q_comp_L=q_comp_L, q_comp_R=q_comp_R,
                q12_L=q12_L, q12_R=q12_R,
                label_rows=label_rows, auto_set=set(v10.mutual_nn_agree(dense, sparse)))


def buckets_for(pre, mode, B1, B2):
    """按口径得每行所属桶（C1/C2/C3）。

    competition    matcher 困惑度（v12 competition_degree；对照基线，换 matcher 就变）
    structure      v10.6 的 12 轴 Structural Complexity Index（默认结构口径）
    structure+hub  12 轴 q 与同侧 hub/dup 两个信号按轴等权平均
    structure_sig  旧版 7 信号口径（保留做 A/B）
    """
    if mode == "competition":
        lc, rc, _ = v12.classify_rows_jointly_by_competition(
            pre["q_comp_L"], pre["q_comp_R"], B1, B2)
    elif mode in ("structure", "structure+hub"):
        qL = np.asarray(pre["q12_L"], dtype=float)
        qR = np.asarray(pre["q12_R"], dtype=float)
        if mode == "structure+hub":
            # 12 轴 q 本身是一条"轴级"量；hub / dup 各再当一条轴，三轴等权平均。
            # 方向沿用 sb.ORIENT：hub/dup 大者更难 ⇒ 直接用池化百分位（不翻转）。
            n = len(qL)
            hub = np.concatenate([np.asarray(pre["sigL"]["hub"], dtype=float),
                                  np.asarray(pre["sigR"]["hub"], dtype=float)])
            dup = np.concatenate([np.asarray(pre["sigL"]["dup"], dtype=float),
                                  np.asarray(pre["sigR"]["dup"], dtype=float)])
            rh = sb._percentile_rank(hub)
            rd = sb._percentile_rank(dup)
            qL = (qL + rh[:n] + rd[:n]) / 3.0
            qR = (qR + rh[n:] + rd[n:]) / 3.0
        lc, rc, _ = sb.classify_rows_jointly(qL, qR, B1, B2)
    elif mode in sb.MODE_PARTS:
        qL, qR = sb.build_q(pre["sigL"], pre["sigR"], sb.MODE_PARTS[mode])
        lc, rc, _ = sb.classify_rows_jointly(qL, qR, B1, B2)
    else:
        raise ValueError(f"未知分桶口径 {mode!r}（可用：competition / structure / "
                         f"structure+hub / {list(sb.MODE_PARTS)}）")
    lrow = {i: c for c, idxs in lc.items() for i in idxs}
    rrow = {j: c for c, idxs in rc.items() for j in idxs}
    return lrow, rrow


def pA_accept_pairs(bucket, X_emb, X_feat, y_t, label_mask, order_scores,
                    order_kind, floor, seed, ci=None):
    """每桶独立：交叉拟合 OOF → fixed-sequence 选 τA → 该桶接受集。

    order_kind='prob'   按 OOF 概率排序（自高到低扫阈值）
    order_kind='struct' 按结构先验排序（分辨力 min(I_L, I_R)，理论 §2：
                        fixed-sequence 的顺序可由先验定，先验错只损失检验力，不损有效性）

    A1 行级修复：p_A 的证书单元从 pair 换成**行**（每行最多一次），与 τR 同口径。
    ci 给定时按行聚合（a_r=行内标注对最高分、p_r=行内标注正例最高分），再走
    row_fixed_sequence_tau；ci=None 退化成"每 pair 当一行"。

    返回 (accept, diag)。diag[b]['best'] 是该桶能达到的最大 CP 下界 —— 够不到 floor
    时 τA=nan、接受集为空，这是**静默失败**，所以 diag 里显式记下来并打印。
    diag[b]['n_needed'] 是 α-spending 下认证 floor 需要的顶端连正数（可行性判据）。
    """
    accept = np.zeros(len(bucket), dtype=bool)
    diag = {}
    for b, X in ((1, X_emb), (2, X_feat)):
        bm = bucket == b
        lab = bm & label_mask & ~np.isnan(y_t)
        n_lab = int(lab.sum()); n_pos = int((y_t[lab] > 0.5).sum())
        d = dict(n_lab=n_lab, n_pos=n_pos, tau=float("nan"), best=0.0,
                 n_cert=0, n_rows=0, k_rows=0, n_needed=0, ok=False)
        if n_lab >= 4 and 2 <= n_pos <= n_lab - 2:
            y_lab = (y_t[lab] > 0.5).astype(float)
            row_lab = ci[lab] if ci is not None else np.arange(n_lab)
            if order_kind == "struct":
                score_lab = order_scores[lab]
                tau, n_rows, k_rows, best, n_needed = \
                    row_fixed_sequence_tau(score_lab, y_lab > 0.5, row_lab, floor)
                d.update(tau=tau, best=best, n_rows=n_rows, k_rows=k_rows,
                         n_needed=n_needed)
                if np.isfinite(tau):
                    sel = bm & (order_scores >= tau)
                    accept |= sel
                    d.update(n_cert=int(sel.sum()), ok=True)
            else:
                oof = crossfit_oof(X[lab], y_lab, seed=seed)
                tau, n_rows, k_rows, best, n_needed = \
                    row_fixed_sequence_tau(oof, y_lab > 0.5, row_lab, floor)
                d.update(tau=tau, best=best, n_rows=n_rows, k_rows=k_rows,
                         n_needed=n_needed)
                if np.isfinite(tau):
                    mu = X[lab].mean(axis=0); sd = X[lab].std(axis=0) + 1e-9
                    clf = LogisticRegression(C=1.0, max_iter=2000)
                    clf.fit((X[lab] - mu) / sd, y_lab)
                    pA_full = np.full(len(bucket), -np.inf)
                    pA_full[bm] = clf.predict_proba((X[bm] - mu) / sd)[:, 1]
                    sel = bm & (pA_full >= tau)
                    accept |= sel
                    d.update(n_cert=int(sel.sum()), ok=True)
        diag[b] = d
    return accept, diag


def run_mode(pre, mode, c1_k=1, recall=0.95, accept_modes=("p_A", "auto"),
             pA_order="prob", certunit="row", floor=FLOOR, seed=0, quiet=False,
             rej_mode="tau", teacher_cap=0, rb_margin=RB_MARGIN):
    """一个分桶口径下的完整一档：候选集 → 免费拒绝 → p_A/auto 接受 → V1+V2 升级门 → 评测。

    rej_mode 选免费拒绝的证书形式：
      tau   —— 每桶独立的 τR **分数**阈值（原口径）
      rank  —— **名次交集**：emb 名次 ≤ q_e 且 feat 名次 ≤ q_f 才升级（替掉 τR）
      union —— 两者都拒（τR 与名次交集取并，拒得最多）
    两者在同一个 ε_R 下认证（rank 用 ε/2 逐腿，见 `certified_rank_q`）。

    teacher_cap>0 把 teacher 校准帧按桶压到该上限（先保每行 top-1，见 `cap_bucket_pairs`）。
    校准帧同时是 τR / 名次门槛 / p_A 的证书样本，所以它变小 = 三处一起变弱，是真实的取舍。

    rb_margin：V2 回滚阈值，判据是**最终 GT 召回**（按用户决定这一格允许读 human GT）。
    引入 V2 后召回回落**超过** rb_margin 才退回 neutral；召回持平/上升/小幅回落都保留 V2。
    rb_margin=0 退化为"一降就滚"。
    """
    ds = pre["ds"]; cfg = pre["cfg"]; gc = pre["gc"]
    left, right = pre["left"], pre["right"]
    gt_set, dense, sparse = pre["gt_set"], pre["dense"], pre["sparse"]
    nL, nR, sigL, sigR = pre["nL"], pre["nR"], pre["sigL"], pre["sigR"]
    ds_prefix = cfg.get("ds_prefix")
    auto_set = pre["auto_set"]

    if not quiet:
        print("\n" + "=" * 108)
        print(f"  {ds} — bucket={mode}  accept={','.join(accept_modes)}(order={pA_order})  "
              f"certunit={certunit}  C1 k={c1_k}  recall={recall}")
        print("=" * 108)

    lrow_class, rrow_class = buckets_for(pre, mode, B1, B2)

    # 桶优先候选：C1(LLM) 行 → c1_k，其余 → K
    kL = np.array([c1_k if lrow_class.get(i, "C2") == "C1" else K for i in range(nL)])
    kR = np.array([c1_k if rrow_class.get(j, "C2") == "C1" else K for j in range(nR)])
    row_cand, col_cand, cand = build_candidates_per_row_full(dense, sparse, kL, kR)
    cand_full = cand if c1_k >= K else v10.build_candidates(dense, sparse, K, K)[2]

    C = sorted(cand)
    C_arr = np.array(C)
    ci, cj = C_arr[:, 0], C_arr[:, 1]
    emb = dense[ci, cj]; feat = sparse[ci, cj]
    in_auto = np.array([p in auto_set for p in C], dtype=bool)
    bucket = _bucket_of(ci, cj, lrow_class, rrow_class)
    n_c1 = int((bucket == 0).sum())
    Cf = np.array(sorted(cand_full))
    n_c1_full = int((_bucket_of(Cf[:, 0], Cf[:, 1], lrow_class, rrow_class) == 0).sum())
    cand_rec = len(gt_set & set(C)) / len(gt_set) if gt_set else 0.0

    # teacher / neutral 标签（缓存回放，零 LLM）
    cache = json.load(open(HERE / cfg["cache"], encoding="utf-8"))
    lid = left["left_id"].values; rid = right["right_id"].values
    dec_t = v12.load_decisions(cache, cfg["teacher"], C, lid, rid)
    y_t = np.array([(1.0 if dec_t[p] else 0.0) if p in dec_t else np.nan for p in C])
    dec_v = v12.load_decisions(cache, cfg["neutral"], C, lid, rid)
    v_yes = np.array([dec_v.get(p, False) for p in C], dtype=bool)

    label_mask = np.array([i in pre["label_rows"] for i in ci], dtype=bool)

    # ── 校准帧：teacher 采样量。默认 = 采样行的全部 C2/C3 候选对（原口径）；
    # teacher_cap>0 时按桶压到上限（先保每行 top-1）。 τR/名次门槛/p_A 都读这个帧。
    calib_mask = label_mask & ((bucket == 1) | (bucket == 2))
    if teacher_cap and teacher_cap > 0:
        keep = []
        for b, sc in ((1, emb), (2, feat)):
            idx = np.flatnonzero(calib_mask & (bucket == b))
            if len(idx) > teacher_cap:
                idx = cap_bucket_pairs(idx, sc, ci, teacher_cap)
            keep.append(idx)
        calib_mask = np.zeros(len(C), dtype=bool)
        if keep:
            calib_mask[np.concatenate(keep)] = True

    # ── 免费拒绝 A：τR（每桶独立的**分数**阈值，证书单元可选）──
    unit = _components(ci, cj, nL) if certunit == "comp" else ci
    eps = 1.0 - recall
    rej_tau = np.zeros(len(C), dtype=bool)
    rej_tau_diag = {}
    for b, s_full in ((1, emb), (2, feat)):
        lab = (bucket == b) & calib_mask & ~np.isnan(y_t)
        if lab.sum():
            tR = tau_reject(s_full[lab], (y_t[lab] > 0.5), unit[lab], eps)
            rej_tau[(bucket == b) & (s_full <= tR)] = True
            rej_tau_diag[b] = (float(tR), int(lab.sum()))
    rej_tau = rej_tau & (~in_auto)    # 只拒非 auto

    # ── 免费拒绝 B：名次交集（emb 名次 ≤ q_e **且** feat 名次 ≤ q_f 才升级）──
    # 名次是整数、行内可比，不吃分数尺度；拒绝集是两条腿的**并集**，比"单个分数阈值"
    # 强得多。两条腿共用 ε_R：ε_e = ε_f = ε/2，由并集界 P(丢) ≤ P(丢_e) + P(丢_f) 得证。
    lab_pos = (calib_mask & ~np.isnan(y_t) & (y_t > 0.5)).astype(float)
    rk_d = _int_ranks(dense, row_cand, col_cand, nL, nR)
    rk_s = _int_ranks(sparse, row_cand, col_cand, nL, nR)
    rank_e = rk_d[ci, cj]; rank_f = rk_s[ci, cj]
    q_e, n_e, k_e, ub_e = certified_rank_q(rank_e, lab_pos, unit, eps / 2.0)
    q_f, n_f, k_f, ub_f = certified_rank_q(rank_f, lab_pos, unit, eps / 2.0)
    ok_e, ok_f = bool(ub_e <= eps / 2.0), bool(ub_f <= eps / 2.0)
    rej_rank = ((rank_e > q_e) | (rank_f > q_f)) & (~in_auto)

    if rej_mode == "rank":
        reject = rej_rank
    elif rej_mode == "union":
        reject = rej_tau | rej_rank
    else:
        reject = rej_tau

    # ── p_A：特征 = 桶内匹配分 + 行/列 rank/margin + in_auto + 结构先验（min-priority）──
    rr_d, cr_d = v12._ranks(dense, row_cand, col_cand)
    rm_d, cm_d = v12._margins(dense, row_cand, col_cand)
    rr_s, cr_s = v12._ranks(sparse, row_cand, col_cand)
    rm_s, cm_s = v12._margins(sparse, row_cand, col_cand)
    psf = sb.pair_struct_feats(ci, cj, sigL, sigR)
    X_emb = np.column_stack([emb, rr_d[ci, cj], cr_d[ci, cj], rm_d[ci, cj], cm_d[ci, cj],
                             in_auto.astype(float), psf["mi_min"], psf["fanout_max"],
                             psf["hub_max"], psf["dup_max"]])
    X_feat = np.column_stack([feat, rr_s[ci, cj], cr_s[ci, cj], rm_s[ci, cj], cm_s[ci, cj],
                              psf["mi_min"], psf["fanout_max"], psf["hub_max"], psf["dup_max"]])
    # 结构先验：pair 分辨力 = 两端取小（min-priority），越大越可能干净
    I_L = np.asarray(sigL["mean_self_info"]) * np.asarray(sigL["token_count"])
    I_R = np.asarray(sigR["mean_self_info"]) * np.asarray(sigR["token_count"])
    order_scores = np.minimum(I_L[ci], I_R[cj])

    accept_pA, pA_diag = pA_accept_pairs(bucket, X_emb, X_feat, y_t, calib_mask,
                                         order_scores, pA_order, floor, seed, ci)

    # ── 诊断：桶纯度 + 认证覆盖 + 分量数 + 校准帧/标注帧口径 ──
    # A2：purity 与认证帧必须同口径。purity 用 label_mask（全标注帧），认证用 calib_mask
    # （cap 后），两者不一致会让人拿"全帧纯度"去解释"cap 帧证书"——所以两个都记下来。
    purity = {}
    purity_calib = {}
    for b in (0, 1, 2):
        m_b = (bucket == b) & label_mask & ~np.isnan(y_t)
        m_cb = (bucket == b) & calib_mask & ~np.isnan(y_t)
        purity[b] = (float((y_t[m_b] > 0.5).mean()), int(m_b.sum()))
        purity_calib[b] = (float((y_t[m_cb] > 0.5).mean()), int(m_cb.sum()))
    rates = [purity[b][0] for b in (0, 1, 2)]
    purity_var = float(np.var(rates))
    comps = _components(ci, cj, nL)
    n_comp = int(np.unique(comps).size)
    label_rows_n = int(np.unique(ci[label_mask]).size)
    calib_rows_n = int(np.unique(ci[calib_mask]).size)

    # ── 两种接受层各自的升级带 ──
    routings = {}
    for name, acc in (("p_A", accept_pA), ("auto", in_auto)):
        if name not in accept_modes:
            continue
        rej = reject & (~acc)
        routings[name] = dict(accept=acc, rej=rej, esc=(~acc) & (~rej))

    if not quiet:
        print(f"  规模：{nL}L × {nR}R  GT={len(gt_set)}  |C|={len(C):,}(全K={len(cand_full):,})  "
              f"auto={int(in_auto.sum()):,}  C1容量={n_c1:,}/{n_c1_full:,} "
              f"(降{(1 - n_c1 / n_c1_full) * 100:.0f}%)  候选recall={cand_rec:.4f}")
        print(f"  桶纯度（teacher 正例率 / 已标注对）："
              f"C1={purity[0][0]:.4f}({purity[0][1]:,})  C2={purity[1][0]:.4f}({purity[1][1]:,})  "
              f"C3={purity[2][0]:.4f}({purity[2][1]:,})   跨桶方差={purity_var:.5f}")
        print(f"  校准帧口径：标注行 {label_rows_n:,} / 校准行 {calib_rows_n:,} / "
              f"候选分量 {n_comp:,}  校准帧纯度 C2={purity_calib[1][0]:.4f}("
              f"{purity_calib[1][1]:,}) C3={purity_calib[2][0]:.4f}({purity_calib[2][1]:,})")
        for b in (1, 2):
            d = pA_diag[b]
            flag = "" if d["ok"] else "  ← τA 认证失败（CP 下界够不到 floor）"
            tau_s = "nan" if not np.isfinite(d["tau"]) else f"{d['tau']:.4f}"
            feas = "可行" if d["n_pos"] >= d["n_needed"] else "不可行"
            print(f"  p_A 桶 C{b + 1}: 标注 {d['n_lab']:,}（正 {d['n_pos']:,}） "
                  f"τA={tau_s} 行级CP={d['k_rows']}/{d['n_rows']} 上限={d['best']:.4f} "
                  f"可认证={d['n_cert']:,} 连正需 {d['n_needed']}（{feas}）{flag}")
        print(f"  校准帧={int(calib_mask.sum()):,}（teacher_cap={teacher_cap or '不限'}）")
        print(f"  p_A 接受合计={int(accept_pA.sum()):,}  "
              f"免费拒绝：τR={int(rej_tau.sum()):,}  名次交集={int(rej_rank.sum()):,}  "
              f"⇒ 本次用 {rej_mode}={int(reject.sum()):,}")
        print(f"    名次门槛：q_e={q_e}(正例行 {n_e}/丢 {k_e}/ub {ub_e:.4f}"
              f"{'' if ok_e else ' ← 无证书，退保守'})  "
              f"q_f={q_f}(正例行 {n_f}/丢 {k_f}/ub {ub_f:.4f}"
              f"{'' if ok_f else ' ← 无证书，退保守'})   ε_e=ε_f={eps / 2:.4f}")
        if rej_tau_diag:
            print("    τR 阈值：" + "  ".join(
                f"C{b + 1}={v[0]:.4f}(标 {v[1]:,})" for b, v in sorted(rej_tau_diag.items())))

    # ── 升级门：两条路由的升级带取并集统一补调，避免重复 LLM ──
    all_esc = set()
    for r in routings.values():
        all_esc |= set(C[k] for k in np.flatnonzero(r["esc"]))
    esc_pairs = sorted(all_esc)
    if not quiet:
        print(f"  V1(lenient)：升级(并集) {len(esc_pairs):,} 对 → 查缓存/补调 LLM ...", flush=True)
    dec_lenient, _, l_new = _llm_gate(esc_pairs, left, right, "lenient", gc["model"],
                                      gc["lenient"], cache, HERE / cfg["cache"], ds_prefix)
    lenient_yes_all = {p: dec_lenient.get(p, False) for p in esc_pairs}

    strict_in = sorted(p for p in esc_pairs if lenient_yes_all[p])
    if not quiet:
        print(f"  V2(strict)：lenient 通过 {len(strict_in):,} 对 → 查缓存/补调 LLM ...", flush=True)
    dec_strict, _, s_new = _llm_gate(strict_in, left, right, "strict", gc["model"],
                                     gc["strict"], cache, HERE / cfg["cache"], ds_prefix)

    # teacher 校准调用量（与路由无关）
    teacher_vol = int(calib_mask.sum())

    # 全域（按 C 的）lenient / strict 判词，供回滚判据使用
    v_len_all = np.array([lenient_yes_all.get(p, False) for p in C], dtype=bool)
    v_str_all = np.array([dec_strict.get(p, False) for p in C], dtype=bool)

    # teacher 校准的 token 账（与路由无关，一次算完）
    teacher_pairs = [C[k] for k in np.flatnonzero(calib_mask)]
    bill_teacher = _token_bill(teacher_pairs, "teacher", left, right, dec_t)

    out = {}
    if not quiet:
        print(f"\n  ── 方法对比（完整 GT；LLM=teacher 校准 + 升级门；"
              f"本次补调 lenient {l_new:,}/strict {s_new:,}）──")
    for rn, r in routings.items():
        acc, esc = r["accept"], r["esc"]
        idx_esc = np.flatnonzero(esc)
        yes_l = v_len_all[idx_esc]
        yes_s = v_str_all[idx_esc]
        n_esc = len(idx_esc); n_strict = int(yes_l.sum())
        base = set(C[k] for k in np.flatnonzero(acc))
        variants = {
            "neutral": base | set(C[k] for k in idx_esc[v_yes[idx_esc]]),
            "V1V2":    base | set(C[k] for k in idx_esc[yes_l & yes_s]),
        }
        # ── V2 召回回滚 ────────────────────────────────────────────────
        # 判据 = **最终 GT 召回**：引入 V2 后召回回落超过 RB_MARGIN 才退回 neutral。
        # 只看召回、不看 F1；召回持平或上升一律保留 V2（V2 的精度收益白拿）。
        #   · quora 0.8054 → 0.8188（升）  ⇒ 保留 V2
        #   · acm   0.9948 → 0.9948（不变）⇒ 保留 V2
        #   · products 0.9484 → 0.9300（−0.018 < 0.05）⇒ 保留 V2
        #   · company  0.9400 → 0.7200（−0.22 > 0.05） ⇒ 回滚
        # 注意：按用户明确决定，**这一格直接读 human GT**（这块允许 oracle 对比），
        # 所以它是 oracle 规则、不是可无标签部署的规则。
        # keep_n/keep_s 保留为诊断量（teacher 正例上的留存量），仅用于解释判据分歧：
        # 它继承 teacher 的假阳率，acm 上 keep_s=0.185 却 GT 召回不降，就是 teacher
        # 在 strict 边界处判错的对 —— 所以当初拿它当代理会在 acm 上白丢 0.09 F1。
        lab_esc = esc & label_mask & ~np.isnan(y_t)
        pos = lab_esc & (y_t > 0.5)
        n_lab_pos = int(pos.sum())
        if n_lab_pos:
            keep_n = float(v_yes[pos].mean())
            keep_s = float((v_len_all[pos] & v_str_all[pos]).mean())
        else:
            keep_n = keep_s = 1.0
        m_n, m_v = variants["neutral"], variants["V1V2"]
        g = len(gt_set) if gt_set else 1
        r_n = len(m_n & gt_set) / g
        r_v = len(m_v & gt_set) / g
        rb = bool(r_v < r_n - rb_margin)
        rb_g = rb                      # 唯一判据就是 GT 召回比较；rbG 保留为同一结果的别名
        variants["V1V2rb"] = m_n if rb else m_v
        variants["V1V2rbG"] = m_n if rb_g else m_v
        # 判据本身读 GT（不再免费），但**成本侧的回滚仍然省**：回滚掉之后，
        # 未标注那部分 strict 判词不用调。所以下面按 rb 决定 strict 的计费口径。
        n_strict_lab = int((yes_l & lab_esc[idx_esc]).sum())
        # 该路由自己的升级带上的 token 账（升级门/neutral 门都只调它自己的那批）
        esc_pairs_r = [C[k] for k in idx_esc]
        strict_pairs_r = [C[k] for k in idx_esc[yes_l]]
        strict_pairs_rb = [C[k] for k in idx_esc[yes_l & lab_esc[idx_esc]]]
        bill_l = _token_bill(esc_pairs_r, "lenient", left, right, dec_lenient)
        bill_s = _token_bill(strict_pairs_r, "strict", left, right, dec_strict)
        bill_s_rb = _token_bill(strict_pairs_rb, "strict", left, right, dec_strict)
        bill_n = _token_bill(esc_pairs_r, "neutral", left, right, dec_v)
        for gname, matched in variants.items():
            p, rr_, f, mtp, mfp, mfn = _final_metrics(matched, gt_set)
            if gname == "V1V2":
                n_extra, bill = n_strict, _bill_add(bill_l, bill_s)
            elif gname == "V1V2rb":
                n_extra = n_strict_lab if rb else n_strict
                bill = _bill_add(bill_l, bill_s_rb if rb else bill_s)
            elif gname == "V1V2rbG":
                n_extra = n_strict_lab if rb_g else n_strict
                bill = _bill_add(bill_l, bill_s_rb if rb_g else bill_s)
            else:
                n_extra, bill = 0, bill_n
            # llm    = 本轮 sweep 实际发生的调用（teacher 校准每口径各付一次）
            # llm_srv= 一轮**部署**的在线调用（teacher 校准只付一次、不随口径重复）
            llm = teacher_vol + n_esc + n_extra
            llm_srv = n_esc + n_extra
            bill = _bill_add(bill_teacher, bill)
            key = f"{rn}+{gname}"
            out[key] = dict(p=p, r=rr_, f=f, matched=len(matched), esc=n_esc, llm=llm,
                            llm_srv=llm_srv, teacher=teacher_vol,
                            accept=int(acc.sum()), reject=int(reject.sum()),
                            tok_total=bill["total"], tok_prompt=bill["prompt"],
                            rb=rb, rb_g=rb_g, keep_n=keep_n, keep_s=keep_s,
                            n_lab_pos=n_lab_pos)
            if not quiet:
                tag = ""
                if gname == "V1V2rb":
                    tag = (f"  ← {'回滚' if rb else '保留V2'}(GT召回) "
                           f"r_n={r_n:.4f} r_v={r_v:.4f} Δ={r_n - r_v:+.4f}"
                           f"(阈值{rb_margin:g})  诊断 keep_n={keep_n:.3f}"
                           f" keep_s={keep_s:.3f} n正={n_lab_pos}")
                elif gname == "V1V2rbG":
                    tag = (f"  ← {'回滚' if rb_g else '保留V2'}(同上) "
                           f"r_n={r_n:.4f} r_v={r_v:.4f} Δ={r_n - r_v:+.4f}")
                print(f"    {key:>14} | acc={int(acc.sum()):>6,} rej={int(reject.sum()):>6,} "
                      f"升级={n_esc:>6,} matched={len(matched):>7,} | "
                      f"P={p:>6.4f} R={rr_:>6.4f} F1={f:>6.4f} | "
                      f"LLM={llm:>7,}(在线 {llm_srv:>7,}) tok={bill['total']:>10,}{tag}")

    return dict(ds=ds, mode=mode, cand_rec=cand_rec, n_c1=n_c1, n_c1_full=n_c1_full,
                purity=purity, purity_var=purity_var, purity_calib=purity_calib,
                pA_diag=pA_diag, n_comp=n_comp, label_rows_n=label_rows_n,
                calib_rows_n=calib_rows_n,
                n_auto=int(in_auto.sum()), accept_pA=int(accept_pA.sum()),
                reject=int(reject.sum()), rej_mode=rej_mode, rb_margin=rb_margin,
                rej_tau=int(rej_tau.sum()), rej_rank=int(rej_rank.sum()),
                q_e=q_e, q_f=q_f, n_e=n_e, n_f=n_f, k_e=k_e, k_f=k_f,
                ok_e=ok_e, ok_f=ok_f, calib=int(calib_mask.sum()),
                teacher=int(teacher_vol), **out)


def _fmt_mode(mode):
    return {"competition": "comp", "structure": "cx12", "structure+hub": "cx12+hub",
            "poverty": "poverty", "poverty+hub": "poverty+hub"}.get(mode, mode)


def main():
    ap = argparse.ArgumentParser(description="结构分桶 + p_A 主流程/auto 对照 + V1+V2 升级门")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CFG.keys()))
    ap.add_argument("--bucket-mode", nargs="+",
                    default=["competition", "structure", "structure+hub"],
                    help="分桶口径，可多选：competition / structure / structure+hub")
    ap.add_argument("--accept", nargs="+", default=["p_A", "auto"],
                    help="接受层：p_A（主）/ auto（对照）")
    ap.add_argument("--pA-order", choices=["prob", "struct"], default="prob",
                    help="p_A 的 fixed-sequence 顺序：OOF 概率 / 结构先验（分辨力）")
    ap.add_argument("--certunit", choices=["row", "comp"], default="row",
                    help="证书单元：行 / 候选图连通分量（τR 与名次门槛共用）")
    ap.add_argument("--rej-mode", choices=["tau", "rank", "union"], default="tau",
                    help="免费拒绝口径：tau=每桶 τR 分数阈值 / rank=名次交集 / union=两者取并")
    ap.add_argument("--teacher-cap", type=int, default=0,
                    help="teacher 校准帧按桶的上限（0=不限，即采样行的全部 C2/C3 候选对）")
    ap.add_argument("--rb-margin", type=float, default=RB_MARGIN,
                    help="V2 回滚阈值：最终 GT 召回回落超过它才回滚（0=一降就滚）")
    ap.add_argument("--floor", type=float, default=FLOOR, help="p_A 的 CP 下界要求")
    ap.add_argument("--recall", type=float, default=0.95)
    ap.add_argument("--m", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c1-k", type=int, default=1)
    ap.add_argument("--k-hub", type=int, default=10, help="hubness 的 kNN 阶数")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        if ds not in DATASET_CFG:
            print(f"[skip] 未知数据集 {ds!r}")
            continue
        pre = prepare(ds, args.seed, args.m, k_hub=args.k_hub)
        for mode in args.bucket_mode:
            rows.append(run_mode(pre, mode, c1_k=args.c1_k, recall=args.recall,
                                 accept_modes=tuple(args.accept), pA_order=args.pA_order,
                                 certunit=args.certunit, floor=args.floor, seed=args.seed,
                                 rej_mode=args.rej_mode, teacher_cap=args.teacher_cap,
                                 rb_margin=args.rb_margin))

    W = 138
    print("\n" + "=" * W)
    print(f"  汇总 1：分桶口径 × 候选集（p_A 主流程；完整 human GT）")
    print("=" * W)
    print(f"  {'dataset':>16} {'bucket':>11} | {'候选recall':>10} {'C1容量降':>9} | "
          f"{'纯度方差':>9} | {'C2可认证':>9} {'C3可认证':>9}")
    for r in rows:
        drop = (1 - r["n_c1"] / r["n_c1_full"]) * 100 if r["n_c1_full"] else 0.0
        print(f"  {r['ds']:>16} {_fmt_mode(r['mode']):>11} | {r['cand_rec']:>10.4f} "
              f"{drop:>8.0f}% | {r['purity_var']:>9.5f} | "
              f"{r['pA_diag'][1]['n_cert']:>9,} {r['pA_diag'][2]['n_cert']:>9,}")

    print("\n" + "=" * W)
    print(f"  汇总 2：P / R / F1（V1+V2 门）—— p_A（主） vs auto（对照）")
    print("=" * W)
    print(f"  {'dataset':>16} {'bucket':>11} | {'p_A P':>7} {'p_A R':>7} {'p_A F1':>7} | "
          f"{'auto P':>7} {'auto R':>7} {'auto F1':>7} | {'ΔF1':>8} | "
          f"{'p_A tok':>11} {'auto tok':>11} {'Δtok':>11}")
    for r in rows:
        a = r.get("auto+V1V2"); b = r.get("p_A+V1V2")
        a_n = r.get("auto+neutral"); b_n = r.get("p_A+neutral")
        if not a or not b:
            continue
        print(f"  {r['ds']:>16} {_fmt_mode(r['mode']):>11} | "
              f"{b['p']:>7.4f} {b['r']:>7.4f} {b['f']:>7.4f} | "
              f"{a['p']:>7.4f} {a['r']:>7.4f} {a['f']:>7.4f} | {b['f'] - a['f']:>+8.4f} | "
              f"{b['tok_total']:>11,} {a['tok_total']:>11,} "
              f"{b['tok_total'] - a['tok_total']:>+11,}")
        if a_n and b_n:
            print(f"  {'(neutral 门同上)':>16} {'':>11} | "
                  f"{b_n['p']:>7.4f} {b_n['r']:>7.4f} {b_n['f']:>7.4f} | "
                  f"{a_n['p']:>7.4f} {a_n['r']:>7.4f} {a_n['f']:>7.4f} | "
                  f"{b_n['f'] - a_n['f']:>+8.4f} | "
                  f"{b_n['tok_total']:>11,} {a_n['tok_total']:>11,} "
                  f"{b_n['tok_total'] - a_n['tok_total']:>+11,}")

    print("\n" + "=" * W)
    print(f"  汇总 2c：V2 召回回滚（判据 = 最终 GT 召回；回落超过 "
          f"{rows[0].get('rb_margin', RB_MARGIN) if rows else RB_MARGIN:g} 才退回 neutral）")
    print("=" * W)
    print(f"  {'dataset':>16} {'bucket':>11} | {'门':>9} | {'R':>7} {'F1':>7} | "
          f"{'LLM':>8} | {'回滚判据(部署)':>18} | {'rbG':>5}")
    for r in rows:
        n_ = r.get("auto+neutral"); v_ = r.get("auto+V1V2")
        rb_ = r.get("auto+V1V2rb"); g_ = r.get("auto+V1V2rbG")
        if not (n_ and v_ and rb_):
            continue
        for lab, d in (("neutral", n_), ("V1V2", v_), ("V1V2rb", rb_),
                       ("V1V2rbG", g_)):
            if not d:
                continue
            extra = ""
            if lab == "V1V2rb":
                extra = (f"{'回滚' if rb_['rb'] else '保留V2':>9} "
                         f"kn={rb_['keep_n']:.3f} ks={rb_['keep_s']:.3f}"
                         f"(n正={rb_['n_lab_pos']})")
            elif lab == "V1V2rbG":
                extra = f"{'回滚' if g_['rb_g'] else '保留V2':>5}"
            print(f"  {r['ds'] if lab == 'neutral' else '':>16} "
                  f"{_fmt_mode(r['mode']) if lab == 'neutral' else '':>11} | "
                  f"{lab:>9} | {d['r']:>7.4f} {d['f']:>7.4f} | {d['llm']:>8,} | "
                  f"{extra}")

    print("\n" + "=" * W)
    print(f"  汇总 2d：免费拒绝口径 A/B —— τR 分数阈值 vs 名次交集（本次跑的是 "
          f"rej_mode={rows[0]['rej_mode'] if rows else '?'}）")
    print("=" * W)
    print(f"  {'dataset':>16} {'bucket':>11} | {'τR拒':>7} {'名次拒':>7} {'Δ拒':>8} | "
          f"{'q_e':>7} {'q_f':>7} | {'本次拒绝':>8} {'auto升级':>9}")
    print(f"  （q_e/q_f 后缀：证=有证书；退=CP 达不到 ε/2 退保守回退）")
    tot_t = tot_r = 0
    for r in rows:
        a = r.get("auto+V1V2")
        if not a:
            continue
        qe = f"{r['q_e']}{'' if r['ok_e'] else '退'}"
        qf = f"{r['q_f']}{'' if r['ok_f'] else '退'}"
        print(f"  {r['ds']:>16} {_fmt_mode(r['mode']):>11} | {r['rej_tau']:>7,} "
              f"{r['rej_rank']:>7,} {r['rej_rank'] - r['rej_tau']:>+8,} | "
              f"{qe:>7} {qf:>7} | {r['reject']:>8,} {a['esc']:>9,}")
        tot_t += r["rej_tau"]; tot_r += r["rej_rank"]
    print(f"  {'全表合计':>16} {'':>11} | {tot_t:>7,} {tot_r:>7,} {tot_r - tot_t:>+8,} |")

    print("\n" + "=" * W)
    print(f"  汇总 2b：token 账（tiktoken cl100k 代理 Qwen；prompt+completion）")
    print("=" * W)
    for gname in ("V1V2", "neutral"):
        tp = sum(r[f"p_A+{gname}"]["tok_total"] for r in rows if f"p_A+{gname}" in r)
        ta = sum(r[f"auto+{gname}"]["tok_total"] for r in rows if f"auto+{gname}" in r)
        cp = sum(r[f"p_A+{gname}"]["llm"] for r in rows if f"p_A+{gname}" in r)
        ca = sum(r[f"auto+{gname}"]["llm"] for r in rows if f"auto+{gname}" in r)
        print(f"  {gname:>8} 全表合计 | p_A {cp:>8,} 次 / {tp:>12,} tok（"
              f"{tp / max(cp, 1):.1f} tok/次） | auto {ca:>8,} 次 / {ta:>12,} tok（"
              f"{ta / max(ca, 1):.1f} tok/次） | Δtok {tp - ta:>+12,}")

    print("\n" + "=" * W)
    print(f"  汇总 2e：把 teacher 校准拆出来（一轮部署只付一次，不随分桶口径重复）")
    print("=" * W)
    print(f"  {'dataset':>16} {'bucket':>11} | {'teacher':>8} {'升级门':>8} {'在线合计':>8} | "
          f"{'含teacher':>10} {'teacher占比':>10}")
    tt = ts = te = 0
    for r in rows:
        a = r.get("auto+V1V2")
        if not a:
            continue
        print(f"  {r['ds']:>16} {_fmt_mode(r['mode']):>11} | {a['teacher']:>8,} "
              f"{a['esc']:>8,} {a['llm_srv']:>8,} | {a['llm']:>10,} "
              f"{a['teacher'] / max(a['llm'], 1) * 100:>9.1f}%")
        tt += a["teacher"]; ts += a["esc"]; te += a["llm_srv"]
    print(f"  {'多口径 sweep 合计':>16} {'':>11} | {tt:>8,} {ts:>8,} {te:>8,} | "
          f"{tt + te:>10,} {tt / max(tt + te, 1) * 100:>9.1f}%")
    print(f"  一轮部署（teacher 只付一次，取单口径最大） | teacher≈{max((r['teacher'] for r in rows), default=0):>7,} "
          f"| 在线 ≈ {te:>7,}  | 部署合计 ≈ "
          f"{max((r['teacher'] for r in rows), default=0) + te:>8,}  "
          f"（vs sweep 逐口径相加 {tt + te:,}，虚高 {tt + te - max((r['teacher'] for r in rows), default=0) - te:,} 次）")

    print("\n" + "=" * W)
    print(f"  汇总 2f：A2 认证诊断 —— p_A 连正可行性 / 分量数 / 校准帧口径")
    print("=" * W)
    print(f"  {'dataset':>16} {'bucket':>11} | {'C2 正/需':>9} {'C3 正/需':>9} | "
          f"{'标注行':>7} {'校准行':>7} {'分量':>7} | {'C2行CP':>7} {'C3行CP':>7}")
    for r in rows:
        d2, d3 = r["pA_diag"][1], r["pA_diag"][2]
        c2 = f"{d2['n_pos']}/{d2['n_needed']}{'' if d2['ok'] else '✗'}"
        c3 = f"{d3['n_pos']}/{d3['n_needed']}{'' if d3['ok'] else '✗'}"
        print(f"  {r['ds']:>16} {_fmt_mode(r['mode']):>11} | {c2:>9} {c3:>9} | "
              f"{r['label_rows_n']:>7,} {r['calib_rows_n']:>7,} {r['n_comp']:>7,} | "
              f"{d2['k_rows']}/{d2['n_rows']:>7} {d3['k_rows']}/{d3['n_rows']:>7}")
    print(f"  （正/需 = 正例标注行数 / α-spending 下认证 floor 需要的连正数；✗=够不到 floor）")

    print("\n" + "=" * W)
    print(f"  汇总 3：按分桶口径聚合（四数据集 F1 均值，V1+V2 门）")
    print("=" * W)
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    print(f"  {'bucket':>11} | {'p_A F1均':>9} {'auto F1均':>10} | {'候选recall均':>12} "
          f"{'纯度方差均':>11} {'C2+C3可认证':>12} | {'p_A tok均':>11} {'auto tok均':>11}")
    for m in modes:
        sub = [r for r in rows if r["mode"] == m]
        pa = [r["p_A+V1V2"]["f"] for r in sub if "p_A+V1V2" in r]
        au = [r["auto+V1V2"]["f"] for r in sub if "auto+V1V2" in r]
        cr = [r["cand_rec"] for r in sub]
        pv = [r["purity_var"] for r in sub]
        nc = [r["pA_diag"][1]["n_cert"] + r["pA_diag"][2]["n_cert"] for r in sub]
        tp = [r["p_A+V1V2"]["tok_total"] for r in sub if "p_A+V1V2" in r]
        ta = [r["auto+V1V2"]["tok_total"] for r in sub if "auto+V1V2" in r]
        print(f"  {_fmt_mode(m):>11} | {np.mean(pa) if pa else float('nan'):>9.4f} "
              f"{np.mean(au) if au else float('nan'):>10.4f} | "
              f"{np.mean(cr):>12.4f} {np.mean(pv):>11.5f} {sum(nc):>12,} | "
              f"{np.mean(tp) if tp else float('nan'):>11,.0f} "
              f"{np.mean(ta) if ta else float('nan'):>11,.0f}")


if __name__ == "__main__":
    main()
