# -*- coding: utf-8 -*-
"""run_hybrid_join_v12.py — 候选集 hybrid + 竞争度分桶 + 加性重标定双阈值（τR/τM）。

相对 v10.6 的四点改动，且借鉴 v11 的理论（两者优点各取）：

  1. **C3 从「LLM 设计 feature」改成「bm25 / tf-idf / jaccard」简单词法特征**（零 LLM）。
     v10.6 的 co_generate_c3_features / extract_c3_features / compile_compare_fn 全部移除，
     C3 判定层只用 token 级简单特征：tf-idf cosine（候选索引）、bm25 cosine、jaccard、
     name_jaccard、contain 与数字 token 重叠。

  2. **分桶信号从 q_struct(结构复杂度) 换成 q_competition(竞争度)**。
     5 数据集已证伪 q_struct：Spearman(q_struct, 恢复深度) 没有一个显著为正（quora 甚至
     显著为负）。底层事实是候选集 k=10 已饱和（recall@10≈1，co-miss 只有个位数），没有
     残余漏检留给 q 去预测。q_competition 有信号，但信号在「auto-accept 覆盖 / 精度风险」，
     不在 recall：mutual-NN 自动接受的覆盖率随竞争度单调塌陷（products 0.77→0.29），而
     精度恒 ~0.97——mutual-NN 安全但覆盖不足，高竞争行（近似候选多、margin 小）必须交 C1。
     所以 C1/C2/C3 的「标签」不再按「语义难度」，而是按「候选歧义度」路由 LLM 预算。

  3. **C2/C3 的 τR/τM 借鉴 v11 的「加性重标定后验 nb_flex + 双阈值 + 精度硬下限」**。
     不再用 v10.6 的 CP / order-statistic / tail-census 证书堆。η 是廉价证据的 logistic OOF，
     τR(拒) 与 τM(接) 是 η 的分位阈值；中段 (τR, τM) 查验证器，按
     nb_flex = logistic[1, zη, V] ≥ θ 判定（η 与验证器 V 的权重都从数据学，加性、无交互）。

  4. **删掉 v10.6 所有 δ**（delta margin / 选择性弃权 |s-tau|<δ / _selective_params /
     risk_controlled_selective_routing 里的 δ 网格）。升级路径改为 η 分位双阈值 (τR, τM)，
     「中段」由两个分位点夹出，不需要任何手调 δ。

本文件是**诊断/原型版（GT 标签）**：a/b/η/nb_flex 用 GT 估，验证器决策从
.llm_decision_cache.json 回放（neutral/teacher/lenient 三档）。端到端要把标签换成校准行
+ 审计模型（审计器 ≠ C1 族），见 main() 里 TODO。

零新 LLM 调用；--live 才实时调用 LLM 取 token/latency。

复用 v10.10 的 load_embedding_model / load_data / fused_score / build_candidates /
mutual_nn_agree / make_llm_decider（cheap 打分、候选集、C1 验证器）。
"""
import os
import re
import sys
import json
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_hybrid_join_v10_10 as v10

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import beta as _beta, norm as _norm

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_decision_cache.json")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# 验证器决策回放标签（v10.8 缓存 key 前缀）
TAGS = {
    "neutral": "openai/qwen-plus|neutral|v6",
    "teacher": "openai/qwen-plus|teacher|v6",
    "lenient": "openai/qwen-plus|lenient|v6",
}

# C1/C2/C3 行预算（竞争度分位）：C1 = 顶部 B1 最高竞争行（LLM），C2 = 次 B2，C3 = 余下。
C1_ROW_BUDGET = 0.05   # top 5% 最高竞争度 -> C1（LLM 直验）
C2_ROW_BUDGET = 0.30   # 次 30% -> C2（双阈值 + 中段查询）

# 验证器评估的最低样本量：决策太少 / 正例太少时，a/b、nb_flex、τR/τM 前沿都不可信，直接跳过。
MIN_VERIFY_PAIRS = 50
MIN_VERIFY_POS = 5


# ────────────────────────────────────────────────────────────────────────────
# 词法 C3（bm25 / tf-idf / jaccard 简单特征，零 LLM）
# ────────────────────────────────────────────────────────────────────────────
def _toks(s):
    return TOKEN_RE.findall(str(s).lower())


def _jaccard(ta, tb):
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _bm25_cos(l_toks, r_toks, k1=1.5, b=0.75):
    """bm25 加权的词袋余弦（对 D 内每个 pair，惰性求值）；借自 v11。

    全矩阵的 BM25 无闭式点积（有长度归一化项），只在候选集 D 上逐对算，成本 ~|D|·token。
    """
    docs = l_toks + r_toks
    n = len(docs)
    df = Counter()
    for t in docs:
        df.update(set(t))
    avgdl = sum(len(t) for t in docs) / n
    idf = {w: np.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in df}

    def vec(toks):
        dl = len(toks) or 1
        tf = Counter(toks)
        return {w: idf.get(w, 0.0) * c * (k1 + 1) / (c + k1 * (1 - b + b * dl / avgdl))
                for w, c in tf.items()}

    lv = [vec(t) for t in l_toks]
    rv = [vec(t) for t in r_toks]
    ln = np.array([np.sqrt(sum(v * v for v in d.values())) for d in lv])
    rn = np.array([np.sqrt(sum(v * v for v in d.values())) for d in rv])

    def cos(i, j):
        a, bb = lv[i], rv[j]
        if len(a) > len(bb):
            a, bb = bb, a
        dot = sum(val * bb.get(w, 0.0) for w, val in a.items())
        denom = ln[i] * rn[j]
        return dot / denom if denom > 0 else 0.0

    return cos


def _ranks(S, row_cand, col_cand):
    """行/列内降序排名归一化到 [0,1]（rr[i,j]=0 表示 j 是 i 的 top-1）。"""
    nL, nR = S.shape
    rr = np.zeros((nL, nR)); cr = np.zeros((nL, nR))
    for i in range(nL):
        js = sorted(row_cand[i], key=lambda j: S[i, j], reverse=True)
        n = len(js)
        if n < 2:
            continue
        for pos, j in enumerate(js):
            rr[i, j] = (n - 1 - pos) / (n - 1)
    for j in range(nR):
        is_ = sorted(col_cand[j], key=lambda i: S[i, j], reverse=True)
        n = len(is_)
        if n < 2:
            continue
        for pos, i in enumerate(is_):
            cr[i, j] = (n - 1 - pos) / (n - 1)
    return rr, cr


def _margins(S, row_cand, col_cand):
    """行/列 top1-top2 证据 margin（rm[i,j] = S[i,j] - S[i,次优]，j 非 top1 时用 top1 做次优）。"""
    nL, nR = S.shape
    rm = np.zeros((nL, nR)); cm = np.zeros((nL, nR))
    for i in range(nL):
        js = list(row_cand[i])
        if not js:
            continue
        js.sort(key=lambda j: S[i, j], reverse=True)
        top, second = js[0], (js[1] if len(js) > 1 else js[0])
        for j in js:
            rival = second if j == top else top
            rm[i, j] = S[i, j] - S[i, rival]
    for j in range(nR):
        is_ = list(col_cand[j])
        if not is_:
            continue
        is_.sort(key=lambda i: S[i, j], reverse=True)
        top, second = is_[0], (is_[1] if len(is_) > 1 else is_[0])
        for i in is_:
            rival = second if i == top else top
            cm[i, j] = S[i, j] - S[i, rival]
    return rm, cm


# η 模型特征（简单特征 = C2 dense + C3 词法）。"sparse" 即 tf-idf cosine（C3 候选索引），
# bm25 / jaccard / name_jaccard / contain / num 是额外的简单 C3 特征（借自 v11）。
FEAT = ["dense", "dense_rrank", "dense_crank", "dense_rmargin", "dense_cmargin",
        "sparse", "bm25", "jaccard", "name_jaccard", "contain_lr", "contain_rl",
        "num_inter", "num_l_in_r", "num_r_in_l",
        "name_num_inter", "name_num_conflict",
        "sparse_rrank", "sparse_crank", "fused"]


def extract_features(left, right, dense, sparse, fused, row_cand, col_cand, D):
    """对 D 内每个 pair 提取简单特征表 X（C2 dense + C3 词法，无 LLM）。"""
    dense_rr, dense_cr = _ranks(dense, row_cand, col_cand)
    sparse_rr, sparse_cr = _ranks(sparse, row_cand, col_cand)
    dense_rm, dense_cm = _margins(dense, row_cand, col_cand)

    l_nd = [f"{n} {d}".strip() for n, d in zip(left["left_name"], left["left_description"])]
    r_nd = [f"{n} {d}".strip() for n, d in zip(right["right_name"], right["right_description"])]
    l_toks = [_toks(s) for s in l_nd]
    r_toks = [_toks(s) for s in r_nd]
    l_all = [set(t) for t in l_toks]
    r_all = [set(t) for t in r_toks]
    l_name = [set(_toks(str(n))) for n in left["left_name"]]
    r_name = [set(_toks(str(n))) for n in right["right_name"]]
    l_num = [set(t for t in s if any(c.isdigit() for c in t)) for s in l_toks]
    r_num = [set(t for t in s if any(c.isdigit() for c in t)) for s in r_toks]
    l_name_num = [set(t for t in s if any(c.isdigit() for c in t)) for s in l_name]
    r_name_num = [set(t for t in s if any(c.isdigit() for c in t)) for s in r_name]
    bm25_cos = _bm25_cos(l_toks, r_toks)

    X = np.empty((len(D), len(FEAT)))
    for r, (i, j) in enumerate(D):
        la, ra = l_all[i], r_all[j]
        ln_, rn_ = l_name[i], r_name[j]
        lnu, rnu = l_num[i], r_num[j]
        lnn, rnn = l_name_num[i], r_name_num[j]
        un = la | ra
        nun = lnu | rnu
        name_inter = len(lnn & rnn)
        name_conflict = 1.0 if (lnn and rnn and name_inter == 0) else 0.0
        X[r] = [
            dense[i, j], dense_rr[i, j], dense_cr[i, j], dense_rm[i, j], dense_cm[i, j],
            sparse[i, j], bm25_cos(i, j),
            len(la & ra) / len(un) if un else 0.0,
            len(ln_ & rn_) / len(ln_ | rn_) if (ln_ | rn_) else 0.0,
            len(ln_ & ra) / len(ln_) if ln_ else 0.0,
            len(rn_ & la) / len(rn_) if rn_ else 0.0,
            len(lnu & rnu),
            len(lnu & ra) / len(lnu) if lnu else 0.0,
            len(rnu & la) / len(rnu) if rnu else 0.0,
            float(name_inter), name_conflict,
            sparse_rr[i, j], sparse_cr[i, j], fused[i, j],
        ]
    return X


# ────────────────────────────────────────────────────────────────────────────
# 竞争度分桶（替代 q_struct；信号在 auto-accept 覆盖/精度，不在 recall）
# ────────────────────────────────────────────────────────────────────────────
def _pct(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return np.zeros(0)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size)
    ranks[order] = np.arange(1, v.size + 1)
    return ranks / v.size


def competition_degree(dense, sparse, fused, k):
    """行级竞争度 q_competition ∈ [0,1]，label-free，全部来自已算量：

        q = mean( pct_rank(该行候选数), 1 - pct_rank(top1-top2 margin) )

    候选数越多、top1 与 top2 越接近 → 该行有更多近似候选、margin 越小 → 竞争越激烈 →
    mutual-NN 越不敢自动接受（auto 覆盖率塌陷），越需要 C1。对左右表各算一遍，返回 (qL, qR)。
    """
    row_cand, col_cand, cand = v10.build_candidates(dense, sparse, k, k)
    nL, nR = dense.shape

    l_sizes = np.array([len(row_cand[i]) for i in range(nL)], dtype=float)
    l_marg = np.zeros(nL)
    for i in range(nL):
        order = np.argsort(-fused[i], kind="stable")
        s1 = fused[i, order[0]]
        s2 = fused[i, order[1]] if nR > 1 else s1
        l_marg[i] = s1 - s2

    r_sizes = np.array([len(col_cand[j]) for j in range(nR)], dtype=float)
    r_marg = np.zeros(nR)
    for j in range(nR):
        order = np.argsort(-fused[:, j], kind="stable")
        s1 = fused[order[0], j]
        s2 = fused[order[1], j] if nL > 1 else s1
        r_marg[j] = s1 - s2

    # pooled 分位秩（两表共享同一尺度，v10.6 joint routing 的做法）
    ps = _pct(np.concatenate([l_sizes, r_sizes]))
    pm = _pct(np.concatenate([l_marg, r_marg]))
    qL = 0.5 * (ps[:nL] + (1.0 - pm[:nL]))
    qR = 0.5 * (ps[nL:] + (1.0 - pm[nL:]))
    return qL, qR


def classify_rows_jointly_by_competition(qL, qR, B1=C1_ROW_BUDGET, B2=C2_ROW_BUDGET):
    """竞争度分位行路由（v10.6 quantile budget property 的竞争度版）。

    tau1 = Q_{1-B1}(qL∪qR)，tau2 = Q_{1-B1-B2}(qL∪qR)；高竞争 = C1（LLM 直验）。
    返回 (left_classes, right_classes, thresholds)，每张表对同一组阈值打标签。
    """
    lq = np.asarray(qL, dtype=float)
    rq = np.asarray(qR, dtype=float)
    pool = np.concatenate([lq, rq])
    tau1 = float(np.quantile(pool, 1.0 - B1))
    tau2 = float(np.quantile(pool, 1.0 - B1 - B2))

    def _assign(qs):
        c1 = [i for i in range(len(qs)) if qs[i] >= tau1]
        c2 = [i for i in range(len(qs)) if tau2 <= qs[i] < tau1]
        c3 = [i for i in range(len(qs)) if qs[i] < tau2]
        return {"C1": c1, "C2": c2, "C3": c3}

    return _assign(lq), _assign(rq), {"tau1": tau1, "tau2": tau2}


# ────────────────────────────────────────────────────────────────────────────
# 指标与 η / nb_flex（借自 v11）
# ────────────────────────────────────────────────────────────────────────────
def _auc(score, y):
    order = np.argsort(score, kind="stable")
    y = np.asarray(y, dtype=float)[order]
    rank = np.arange(1, len(y) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rank[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(p, y, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    ece = 0.0
    for b in range(nbins):
        m = (p >= edges[b]) & (p < edges[b + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def binom_lcb(k, n, delta, method="clopper_pearson"):
    """二项比例 p 的**单侧 (1-δ) 下置信界**（k 成功 / n 试验）——有限样本 PAC 保证的数学工具。

    意义：若真实成功率为 p，则 P(下界 ≤ p) ≥ 1-δ。用下界代替点估计 p̂=k/n 来断言
    「精度/recall ≥ 某目标」，就把「在样本上恰好达标」升级成「以 1-δ 置信度达标」。

    method:
      clopper_pearson —— 精确、保守（真实覆盖 ≥ 1-δ，绝不欠覆盖），「保证」的黄金标准。
          下界 = Beta(δ; k, n-k+1) 分位；k=0 → 0。
      wilson          —— Wilson score 下界，更紧；大样本下近似覆盖，小样本可能轻微欠覆盖。
    """
    k = int(k); n = int(n)
    if n <= 0 or k <= 0:
        return 0.0
    if method == "wilson":
        phat = k / n
        z = _norm.ppf(1.0 - delta)
        denom = 1.0 + z * z / n
        center = phat + z * z / (2.0 * n)
        half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
        return float(max(0.0, (center - half) / denom))
    # clopper_pearson（默认）
    return float(_beta.ppf(delta, k, n - k + 1))


def fit_nb_flex_deploy(le, V, y):
    """部署版 nb_flex：全量拟合（非 OOF）→ (mu, sd, b0, b1, b2)，供在线算 rho。

    rho = logistic[ b0 + b1*z(le) + b2*V ]，z 是 le 的 z-score。
    注意：**不**给 LogisticRegression 传显式 ones 列——sklearn 自带 intercept；否则
    coef_[0] 会多出一列冗余常数，b1/b2 张冠李戴（v12_full 首版 bug）。
    """
    mu = float(le.mean()); sd = float(le.std()) + 1e-12
    z = (le - mu) / sd
    if len(np.unique(y)) < 2:
        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return mu, sd, float(np.log(p / (1 - p))), 0.0, 0.0
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(np.column_stack([z, V]), y)
    b0 = float(clf.intercept_[0]); b1 = float(clf.coef_[0][0]); b2 = float(clf.coef_[0][1])
    return mu, sd, b0, b1, b2


def rho_deploy(le, V, mu, sd, b0, b1, b2):
    z = (le - mu) / sd
    return 1.0 / (1.0 + np.exp(-(b0 + b1 * z + b2 * V)))


def _crossfit_logistic(X, y, rows, folds, cols, C=1.0):
    Xc = X[:, cols]
    n = len(y)
    oof = np.full(n, np.nan)
    for k in range(folds):
        tr = rows != k; te = rows == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()   # 训练折单类：退回常数先验，避免 logistic 崩溃
            continue
        mu = Xc[tr].mean(0); sd = Xc[tr].std(0) + 1e-12
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit((Xc[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((Xc[te] - mu) / sd)[:, 1]
    return oof


def _isotonic_oof(logit, y):
    n = len(logit)
    rng = np.random.RandomState(0)
    order = rng.permutation(n)
    cal = np.empty(n)
    for half in (0, 1):
        fit_idx = order[half::2]; cal_idx = order[1 - half::2]
        iso = IsotonicRegression(out_of_bounds="clip").fit(logit[fit_idx], y[fit_idx])
        cal[cal_idx] = iso.predict(logit[cal_idx])
    return cal


def load_decisions(cache, tag, D, left_ids, right_ids, dataset=None):
    """回放验证器决策。缓存键带数据集名前缀 `{dataset}|{tag}|{lid}|{rid}`（修复
    旧 bug：键缺数据集名导致跨数据集串号）。dataset 缺省时从 cache 的 `_dataset`
    标记取；再退到旧格式 `{tag}|{lid}|{rid}` 兼容未迁移的缓存。"""
    ds = dataset if dataset is not None else cache.get("_dataset", None)
    dec = {}
    for (i, j) in D:
        lid, rid = left_ids[i], right_ids[j]
        d = None
        if ds is not None:
            d = cache.get(f"{ds}|{tag}|{lid}|{rid}")
        if d is None:                      # 未迁移 / 旧缓存回退
            d = cache.get(f"{tag}|{lid}|{rid}")
        if d in ("yes", "no"):
            dec[(i, j)] = (d == "yes")
    return dec


# ────────────────────────────────────────────────────────────────────────────
# 加性重标定后验（nb 基线 vs nb_flex 正式融合层）+ 双阈值 τR/τM（借自 v11）
# ────────────────────────────────────────────────────────────────────────────
def _laplace_ab(V, y):
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    a = (V[y == 1].sum() + 0.5) / (n_pos + 1.0)
    b = (V[y == 0].sum() + 0.5) / (n_neg + 1.0)
    return a, b


def posterior_crossfit(logit_eta, V, y, rows, folds):
    """加性重标定后验（nb_flex 为正式融合层）+ nb 常数 (a,b) 基线，返回 {variant: OOF ρ}。

    nb      —— 生成式常数 (a,b)：η 系数固定=1、V 权重=log(a/b)（基线，v11 已证伪）。
    nb_flex —— 正式融合层：logistic[1, zη, V]，η 与 V 的权重都从数据学（加性，无交互）。
    """
    n = len(y)
    out = {k: np.full(n, np.nan) for k in ("nb", "nb_flex")}
    le = logit_eta
    for k in range(folds):
        tr = rows != k; te = rows == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        a, b = _laplace_ab(V[tr], y[tr])
        w_yes = np.log(a / b); w_no = np.log((1 - a) / (1 - b))
        out["nb"][te] = 1.0 / (1.0 + np.exp(-(le[te] + V[te] * w_yes + (1 - V[te]) * w_no)))
        if len(np.unique(y[tr])) < 2:
            out["nb_flex"][te] = out["nb"][te]   # 退化：训练折单类，nb_flex 退回常数 nb
            continue
        mu_le = le[tr].mean(); sd_le = le[tr].std() + 1e-12
        zle = lambda arr: (arr - mu_le) / sd_le
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(np.column_stack([np.ones(tr.sum()), zle(le[tr]), V[tr]]), y[tr])
        out["nb_flex"][te] = clf.predict_proba(
            np.column_stack([np.ones(te.sum()), zle(le[te]), V[te]]))[:, 1]
    return out


def frontier(eta, rho_obs, y, rec_targets=(0.85, 0.90, 0.95, 0.98),
             prec_floors=(0.5, 0.7, 0.8, 0.9), theta=0.5, grid=41, delta=None):
    """双阈值 (τR, τM) 扫网格，**在精度约束下**找 recall 目标的最小查询代价（借自 v11）。

    中段判定用判别式后验 ρ≥θ（不是裸 V==1）。返回 best[r][pf] = {queries, recall,
    precision, tau_R, tau_M, n_acc}。精度约束防止退化到「全部接受」。

    delta 非 None 时，「recall≥r / precision≥pf」的判定用**二项比例单侧 (1-δ) 下置信界**
    （binom_lcb）而非点估计——即选出的阈值是「以 1-δ 置信度达标」，而非「样本上恰好达标」。
    返回的 dict 额外带 recall_lcb / precision_lcb（点估计仍存于 recall/precision 供对照）。
    注意：这是**逐格点**的保证；扫网格再挑最优仍有选择偏差，最终保证要在 held-out 行集上
    复核（见 holdout_pac_eval）。
    """
    n = len(y); n_pos = int(y.sum())
    qs = np.quantile(eta, np.linspace(0, 1, grid + 1))
    best = {r: {pf: None for pf in prec_floors} for r in rec_targets}
    for ri in range(1, grid):
        tau_R = qs[ri]
        for ai in range(ri, grid + 1):
            tau_M = qs[ai]
            direct_acc = eta >= tau_M
            direct_rej = eta <= tau_R
            band = (~direct_acc) & (~direct_rej)
            acc = direct_acc | (band & (rho_obs >= theta))
            n_acc = int(acc.sum())
            tp = float(y[acc].sum())
            rec = tp / n_pos if n_pos else 0.0
            prec = tp / n_acc if n_acc else float("nan")
            if delta is not None:
                rec_lo = binom_lcb(tp, n_pos, delta)
                prec_lo = binom_lcb(tp, n_acc, delta) if n_acc else 0.0
            else:
                rec_lo, prec_lo = rec, prec
            queries = int(band.sum())
            for r in rec_targets:
                if rec_lo < r:
                    continue
                for pf in prec_floors:
                    if prec_lo < pf:
                        continue
                    if best[r][pf] is None or queries < best[r][pf]["queries"]:
                        best[r][pf] = {"queries": queries, "recall": float(rec),
                                       "precision": float(prec),
                                       "recall_lcb": float(rec_lo),
                                       "precision_lcb": float(prec_lo),
                                       "tau_R": float(tau_R), "tau_M": float(tau_M),
                                       "n_acc": n_acc}
    return best


def _pick_op(fr, rec_targets, prec_floors, want_r=0.90, want_pf=0.80):
    for r in sorted([t for t in rec_targets if t <= want_r], reverse=True):
        for pf in sorted([p for p in prec_floors if p <= want_pf], reverse=True):
            if fr[r][pf] is not None:
                return fr[r][pf]
    cand = [fr[r][pf] for r in rec_targets for pf in prec_floors if fr[r][pf] is not None]
    return min(cand, key=lambda b: b["queries"]) if cand else None


def end_to_end(eta, rho_obs, D_pairs, gt_set, auto, tau_R, tau_M, theta=0.5):
    """全管线端到端：auto(mutual-NN) ∪ 争议队策略 → 最终匹配集 → P/R/F1 + 查询数。

    set-overlap 评估；查询数 = 中段 (τR, τM) 的对数（这些要过验证器）。
    """
    direct_acc = eta >= tau_M
    direct_rej = eta <= tau_R
    band = (~direct_acc) & (~direct_rej)
    acc_mask = direct_acc | (band & (rho_obs >= theta))
    acc_pairs = {D_pairs[k] for k in range(len(eta)) if acc_mask[k]}
    matched = set(auto) | acc_pairs
    tp = len(matched & gt_set)
    fp = len(matched) - tp
    fn = len(gt_set) - tp
    p = tp / len(matched) if matched else 0.0
    r = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
            "n_queries": int(band.sum()), "n_matched": len(matched)}


def holdout_pac_eval(eta, rho, y, pairs, gt_set, auto_set, valid_frac=0.25,
                     rec_targets=(0.85, 0.90, 0.95, 0.98), prec_floors=(0.5, 0.7, 0.8, 0.9),
                     theta=0.5, delta=0.05, seed=0):
    """held-out PAC 复核：在一个行子集上选 τR/τM（LCB 前沿），在另一个**没见过的行子集**上给下界。

    拆掉「扫网格再挑最优」的选择偏差——阈值只在 select 行上定，valid 行的点估计与下界才是
    诚实的有限样本保证（阈值已冻结、不依赖 valid 行）。η/ρ 传进来的是 OOF 分数，故 valid 行
    的分数本身也是 out-of-sample（训练时没看过 valid 行）。

    返回 {op, valid: {tp,n_acc,n_pos,precision,recall,precision_lcb,recall_lcb,
    n_select_rows,n_valid_rows}, e2e: 含 auto 的 set-overlap 端到端}。valid 里的
    precision_lcb/recall_lcb 即「以 1-δ 置信度，D 接受规则的精度/recall ≥ 此值」。
    """
    n = len(y)
    rows_raw = np.array([p[0] for p in pairs])          # 左行 id：行级 hold-out（行内 pair 不拆）
    uniq = np.unique(rows_raw)
    rng = np.random.RandomState(seed); rng.shuffle(uniq)
    nv = max(1, int(round(len(uniq) * valid_frac)))
    valid_rows = set(uniq[:nv].tolist())
    sel_mask = np.array([r not in valid_rows for r in rows_raw], dtype=bool)
    val_mask = ~sel_mask
    if val_mask.sum() == 0 or sel_mask.sum() == 0:
        return None
    fr = frontier(eta[sel_mask], rho[sel_mask], y[sel_mask], rec_targets, prec_floors,
                  theta=theta, delta=delta)
    op = _pick_op(fr, rec_targets, prec_floors)
    if op is None:
        return None
    direct_acc = eta[val_mask] >= op["tau_M"]
    direct_rej = eta[val_mask] <= op["tau_R"]
    band = ~direct_acc & ~direct_rej
    acc = direct_acc | (band & (rho[val_mask] >= theta))
    yv = y[val_mask]
    tp = float(yv[acc].sum()); n_acc = int(acc.sum()); n_pos = int(yv.sum())
    prec = tp / n_acc if n_acc else float("nan")
    rec = tp / n_pos if n_pos else 0.0
    prec_lcb = binom_lcb(tp, n_acc, delta) if n_acc else 0.0
    rec_lcb = binom_lcb(tp, n_pos, delta)
    valid_pairs = [pairs[k] for k in range(n) if val_mask[k]]
    e2e = end_to_end(eta[val_mask], rho[val_mask], valid_pairs, gt_set, auto_set,
                     op["tau_R"], op["tau_M"], theta=theta)
    return {"op": op,
            "valid": {"tp": int(tp), "n_acc": n_acc, "n_pos": n_pos,
                      "precision": float(prec), "recall": float(rec),
                      "precision_lcb": float(prec_lcb), "recall_lcb": float(rec_lcb),
                      "n_select_rows": int(len(uniq) - nv), "n_valid_rows": int(nv)},
            "e2e": e2e}


def assess_verifier(le, V, ys, rows_s, eta_s, folds, rec_targets, prec_floors,
                    gt_set, auto_set, pairs, theta=0.5, delta=None, valid_frac=0.25):
    """对一个验证器决策 V 跑完整评估：a/b、nb_flex 后验、前沿、端到端。返回 dict（借自 v11）。"""
    a, b = _laplace_ab(V, ys)
    rho = posterior_crossfit(le, V, ys, rows_s, folds)
    fr = frontier(eta_s, rho["nb_flex"], ys, rec_targets, prec_floors, theta=theta, delta=delta)
    op = _pick_op(fr, rec_targets, prec_floors)
    e2e = end_to_end(eta_s, rho["nb_flex"], pairs, gt_set, auto_set,
                     op["tau_R"], op["tau_M"], theta=theta) if op else None
    pac = holdout_pac_eval(eta_s, rho["nb_flex"], ys, pairs, gt_set, auto_set,
                           valid_frac=valid_frac, rec_targets=rec_targets,
                           prec_floors=prec_floors, theta=theta, delta=delta) \
        if delta is not None else None
    return {"a": float(a), "b": float(b), "rho": rho, "fr": fr, "op": op, "e2e": e2e, "pac": pac}


# ────────────────────────────────────────────────────────────────────────────
# 竞争度诊断（复现结论：recall@10 饱和、co-miss 稀疏、auto 覆盖塌陷/精度恒稳）
# ────────────────────────────────────────────────────────────────────────────
def _rank_matrix(M):
    """rank_matrix[i,j] = j 在行 i 内的降序排名（0 起）。"""
    order = np.argsort(-M, axis=1, kind="stable")
    r = np.empty_like(M, dtype=np.int64)
    np.put_along_axis(r, order, np.arange(M.shape[1])[None, :], axis=1)
    return r


def report_competition_diagnostic(qL, dense, sparse, fused, e_l, e_r, gt_set, auto, k, n_bins=4):
    """按左行竞争度四分位分桶，报告「恢复深度 / recall@k / co-miss / mutual-NN auto 覆盖与精度」。

    这直接复现 v10.6 跑完后的三条结论：
      (1) recall@10 所有桶 ≈1 —— 候选集已饱和，竞争度救不了 recall；
      (2) co-miss 稀疏且不集中 —— q 没有「漏检」可预测；
      (3) auto_recall 随竞争度单调塌陷、auto_precision 恒 ~0.97 —— 真正的信号在
          auto-accept 覆盖/精度风险，高竞争行必须交 C1。
    """
    n_gt = len(e_l)
    rD = _rank_matrix(dense); cD = _rank_matrix(dense.T).T
    rS = _rank_matrix(sparse); cS = _rank_matrix(sparse.T).T
    depths = np.zeros(n_gt, dtype=np.int64)
    for t in range(n_gt):
        i, j = int(e_l[t]), int(e_r[t])
        depths[t] = min(rD[i, j], cD[i, j], rS[i, j], cS[i, j]) + 1

    edges = np.quantile(qL, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    qbin = np.searchsorted(edges[1:-1], qL, side="right")

    rows = []
    for b in range(n_bins):
        gt_in_b = [t for t in range(n_gt) if qbin[e_l[t]] == b]
        nb = len(gt_in_b)
        rec1 = rec3 = rec10 = comiss = 0.0
        for t in gt_in_b:
            rec1 += (depths[t] <= 1); rec3 += (depths[t] <= 3); rec10 += (depths[t] <= k)
            comiss += (depths[t] > k)
        auto_in_b = [(i, j) for (i, j) in auto if qbin[i] == b]
        auto_tp = sum(1 for (i, j) in auto_in_b if (i, j) in gt_set)
        auto_rec = (sum(1 for t in gt_in_b if (e_l[t], e_r[t]) in auto) / nb) if nb else float("nan")
        rows.append({
            "bin": b, "n_gt": nb,
            "recall@1": (rec1 / nb) if nb else float("nan"),
            "recall@3": (rec3 / nb) if nb else float("nan"),
            "recall@10": (rec10 / nb) if nb else float("nan"),
            "co_miss@10": int(comiss),
            "auto_recall": auto_rec,
            "auto_precision": (auto_tp / len(auto_in_b)) if auto_in_b else float("nan"),
        })
    return {"bins": rows, "co_miss_total": int(sum(r["co_miss@10"] for r in rows))}


def bucketed_end_to_end(eta_cal, rho_flex, V, pairs, bucket_id, gt_set, auto_set,
                        ops, theta=0.5):
    """按竞争度分桶的端到端：auto ∪ C1直验 ∪ C2/C3(τR/τM + nb_flex)。

    bucket_id: 0=C1(高竞争, 直验 V==yes), 1=C2, 2=C3。ops 是 C2/C3 各自的 frontier 操作点。
    返回 P/R/F1 + 总查询数（C1 直验对 + C2/C3 中段对）。
    """
    matched = set(auto_set)
    n_queries = 0
    n = len(pairs)
    for b in range(3):
        m = bucket_id == b
        if m.sum() == 0:
            continue
        idx = np.where(m)[0]
        if b == 0:
            acc = V[m]                       # C1：直接 LLM 验证
            n_queries += int(m.sum())
        else:
            op = ops.get(b)
            if op is None:
                continue
            tau_R, tau_M = op["tau_R"], op["tau_M"]
            eb = eta_cal[m]; rb = rho_flex[m]
            direct_acc = eb >= tau_M
            direct_rej = eb <= tau_R
            band = ~direct_acc & ~direct_rej
            acc = direct_acc | (band & (rb >= theta))
            n_queries += int(band.sum())
        matched |= {pairs[idx[k]] for k in range(len(idx)) if acc[k]}
    tp = len(matched & gt_set)
    fp = len(matched) - tp
    fn = len(gt_set) - tp
    p = tp / len(matched) if matched else 0.0
    r = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
            "n_queries": n_queries, "n_matched": len(matched)}


# ────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="products")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--b1", type=float, default=C1_ROW_BUDGET, help="C1 行预算（顶部竞争度分位）")
    ap.add_argument("--b2", type=float, default=C2_ROW_BUDGET, help="C2 行预算（次高竞争度分位）")
    ap.add_argument("--rec-targets", default="0.85,0.90,0.95,0.98")
    ap.add_argument("--prec-floors", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="中段 nb_flex 后验判定阈值 ρ≥θ。前沿『不可达』时调低它（如 0.3）恢复"
                         "被丢弃的『验证器 yes 但廉价证据弱』的对")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="有限样本 PAC 保证的置信度：以 1-δ 置信度断言『D 接受规则的精度/recall ≥ 目标』。"
                         "默认 0.05（95%% 置信）。None 传给 frontier 即退回点估计")
    ap.add_argument("--valid-frac", type=float, default=0.25,
                    help="held-out 复核的验证行比例（行级 split，阈值只在此之外的行上选）")
    ap.add_argument("--live", action="store_true",
                    help="额外跑部署基线：实时调用 LLM(neutral) 取真实 token/latency")
    args = ap.parse_args()
    rec_targets = tuple(float(x) for x in args.rec_targets.split(","))
    prec_floors = tuple(float(x) for x in args.prec_floors.split(","))

    model = v10.load_embedding_model()
    left, right, gt = v10.load_data(args.dataset)
    nL, nR = len(left), len(right)
    l_idx = {str(v): k for k, v in enumerate(left["left_id"])}
    r_idx = {str(v): k for k, v in enumerate(right["right_id"])}
    e_l = np.array([l_idx[str(a)] for a in gt["left_id"]])
    e_r = np.array([r_idx[str(b)] for b in gt["right_id"]])
    gt_set = set(zip(e_l.tolist(), e_r.tolist()))

    # C2 = dense，C3 = tf-idf 词法（候选索引）；fused = 融合分位（借 v10.10）
    fused, dense, sparse = v10.fused_score(left, right, model, w=0.5)
    row_cand, col_cand, cand = v10.build_candidates(dense, sparse, args.k, args.k)
    auto = v10.mutual_nn_agree(dense, sparse)
    D = sorted(cand - auto)

    print(f"\n[{args.dataset}] nL={nL} nR={nR} n_gt={len(e_l)}  |C|={len(cand):,}  "
          f"auto={len(auto)}  D={len(D):,}")
    auto_tp = sum(1 for p in auto if p in gt_set)
    print(f"  auto(mutual-NN) 精度 = {auto_tp/len(auto):.4f} ({auto_tp}/{len(auto)})")

    # ── 竞争度分桶（替代 q_struct 的新 C1/C2/C3 标签） ──
    qL, qR = competition_degree(dense, sparse, fused, args.k)
    left_classes, right_classes, tau = classify_rows_jointly_by_competition(
        qL, qR, args.b1, args.b2)
    print(f"\n=== 竞争度分桶（q_competition；替代被证伪的 q_struct） ===")
    print(f"  joint tau1={tau['tau1']:.4f} (C1>=) tau2={tau['tau2']:.4f} (C2>=)  [shared L/R]")
    for cat in ["C1", "C2", "C3"]:
        print(f"  {cat}: left={len(left_classes[cat])}, right={len(right_classes[cat])}")

    # pair 分桶：min-priority（哪边竞争度高就归哪边）
    _prio = {"C1": 0, "C2": 1, "C3": 2}
    lrow_class = {}
    for cat, idxs in left_classes.items():
        for i in idxs:
            lrow_class[i] = cat
    rrow_class = {}
    for cat, idxs in right_classes.items():
        for j in idxs:
            rrow_class[j] = cat

    def _bucket_of(i, j):
        lc = lrow_class.get(i, "C2")
        rc = rrow_class.get(j, "C2")
        return min([lc, rc], key=lambda c: _prio.get(c, 99))

    bucket_id = np.array([_prio[_bucket_of(i, j)] for (i, j) in D], dtype=int)
    n_b1, n_b2, n_b3 = (int((bucket_id == 0).sum()), int((bucket_id == 1).sum()),
                        int((bucket_id == 2).sum()))
    print(f"  D 内 pair 分桶：C1(直验)={n_b1:,}  C2={n_b2:,}  C3={n_b3:,}")

    # ── 竞争度诊断（复现实验结论） ──
    diag = report_competition_diagnostic(qL, dense, sparse, fused, e_l, e_r, gt_set, auto, args.k)
    print(f"\n=== 竞争度四分位诊断（auto 覆盖/精度 + recall@k + co-miss） ===")
    print(f"  {'bin':>4}{'n_gt':>6}{'recall@1':>10}{'recall@3':>10}{'recall@10':>10}"
          f"{'co-miss':>9}{'auto_rec':>9}{'auto_prec':>9}")
    for r in diag["bins"]:
        print(f"  {r['bin']:>4}{r['n_gt']:>6}{r['recall@1']:>10.3f}{r['recall@3']:>10.3f}"
              f"{r['recall@10']:>10.3f}{r['co_miss@10']:>9}{r['auto_recall']:>9.3f}"
              f"{r['auto_precision']:>9.3f}")
    print(f"  co-miss 合计 = {diag['co_miss_total']}（k={args.k} 并集候选已饱和，无残余漏检留给 q 预测 recall）")

    # ── η 模型（dense + 简单词法 C3 + fused） ──
    X = extract_features(left, right, dense, sparse, fused, row_cand, col_cand, D)
    y = np.array([1.0 if (i, j) in gt_set else 0.0 for (i, j) in D], dtype=float)
    n_pos_D = int(y.sum())

    rows_uniq = sorted({i for (i, j) in D})
    rng = np.random.RandomState(0)
    rng.shuffle(rows_uniq)
    folds_arr = np.array_split(rows_uniq, args.folds)
    row2fold = {rr: k for k, fold in enumerate(folds_arr) for rr in fold}
    pair_rows = np.array([row2fold[i] for (i, j) in D])

    eta_oof = _crossfit_logistic(X, y, pair_rows, args.folds, list(range(len(FEAT))))
    logit_eta = np.log(np.clip(eta_oof, 1e-12, 1 - 1e-12) / np.clip(1 - eta_oof, 1e-12, 1 - 1e-12))
    print(f"\n  η 模型: OOF AUC={_auc(eta_oof, y):.4f}  ECE(逻辑)={_ece(eta_oof, y):.4f}")

    cache = json.load(open(CACHE_PATH, encoding="utf-8"))
    left_ids = left["left_id"].values
    right_ids = right["right_id"].values

    out = {"dataset": args.dataset, "nL": nL, "nR": nR, "n_gt": int(len(e_l)),
           "C": int(len(cand)), "auto": int(len(auto)), "D": int(len(D)),
           "auto_precision": float(auto_tp / len(auto)),
           "eta_auc": float(_auc(eta_oof, y)), "eta_ece": float(_ece(eta_oof, y)),
           "competition_buckets": {
               "b1": args.b1, "b2": args.b2, "tau": tau,
               "row_classes": {"left": {k: len(v) for k, v in left_classes.items()},
                               "right": {k: len(v) for k, v in right_classes.items()}},
               "pair_buckets_in_D": {"C1": n_b1, "C2": n_b2, "C3": n_b3},
           },
           "competition_diagnostic": diag,
           "posterior": {}, "frontier": {}, "end_to_end": {},
           "holdout_pac": {},
           "bucketed_end_to_end": {}}

    # TODO(端到端换装): 以下 a/b/nb_flex 用 GT 标签（诊断版）。端到端要换成校准行 + 审计模型
    #   标签，且审计器与 C1 不同族（否则审计与 V 错误相关、b 被低估）。η 也要换成
    #   「m 个标注行训练 + held-out 行评测」，不是全 D 的 GT 交叉拟合。
    eta_cal = _isotonic_oof(eta_oof, y)
    auto_set = set(auto)

    for pname, tag in TAGS.items():
        dec = load_decisions(cache, tag, D, left_ids, right_ids)
        if not dec:
            print(f"\n  [{pname}] 无决策，跳过")
            continue
        if len(dec) < 0.9 * len(D):
            print(f"\n  [{pname}] 决策覆盖不全 {len(dec)}/{len(D)}（跨折可能出 NaN），结果仅供参考")
        pairs = sorted(dec.keys())
        idx = {p: k for k, p in enumerate(D)}
        sel = np.array([idx[p] for p in pairs])
        V = np.array([dec[p] for p in pairs], dtype=float)
        ys = y[sel]
        if len(ys) < MIN_VERIFY_PAIRS or ys.sum() < MIN_VERIFY_POS:
            print(f"\n  [{pname}] 决策 n={len(ys)} 正例={int(ys.sum())} 太少，跳过 τR/τM 评估"
                  f"（qwen-plus 缓存主要覆盖 products）")
            out["posterior"][pname] = None
            continue
        le = logit_eta[sel]
        rows_s = pair_rows[sel]
        eta_s = eta_cal[sel]
        b_id = bucket_id[sel]

        ass = assess_verifier(le, V, ys, rows_s, eta_s, args.folds,
                              rec_targets, prec_floors, gt_set, auto_set, pairs,
                              theta=args.theta, delta=args.delta, valid_frac=args.valid_frac)
        print(f"\n{'='*88}\n[{pname}] 决策 n={len(ys)} (yes={int(V.sum())})  "
              f"正例={int(ys.sum())} 负例={int((1-ys).sum())}  a={ass['a']:.3f} b={ass['b']:.3f}")

        print(f"  ── 后验（nb 基线 vs nb_flex 正式融合层，held-out logloss/ECE）──")
        for vname in ("nb", "nb_flex"):
            print(f"    {vname:<8} logloss={_logloss(ass['rho'][vname], ys):.5f}  "
                  f"ECE={_ece(ass['rho'][vname], ys):.4f}")
        out["posterior"][pname] = {v: {"logloss": _logloss(ass["rho"][v], ys),
                                       "ece": _ece(ass["rho"][v], ys)}
                                   for v in ("nb", "nb_flex")}

        fr = ass["fr"]
        lcb_note = "（LCB 达标口径）" if args.delta is not None else "（点估计口径）"
        print(f"  ── 前沿（nb_flex 双阈值 τR/τM + 精度约束，min 查询数）{lcb_note} ──")
        print("    recall\\prec " + "".join(f"p≥{pf:<5}" for pf in prec_floors))
        for r in rec_targets:
            row = f"    r≥{r:.2f}     "
            for pf in prec_floors:
                b = fr[r][pf]
                row += (f"{b['queries']:<8}" if b is not None else "不可达  ")
            print(row)
        out["frontier"][pname] = {str(r): {str(pf): fr[r][pf] for pf in prec_floors}
                                  for r in rec_targets}

        print(f"  ── 端到端 P/R/F1（auto 全计入，τR/τM 全局）──")
        if ass["e2e"] is None:
            print(f"    无可行操作点")
            out["end_to_end"][pname] = None
        else:
            m = ass["e2e"]
            m["normalized_cost"] = m["n_queries"] * v10.COST_VERIFY + (nL * nR) * v10.COST_CHEAP
            out["end_to_end"][pname] = {**m, "op": {k: ass["op"][k] for k in ("tau_R", "tau_M", "recall", "precision")}}
            print(f"    τR={ass['op']['tau_R']:.3f} τM={ass['op']['tau_M']:.3f} → "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}  "
                  f"查询={m['n_queries']:,} 匹配={m['n_matched']:,} "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

        if ass["pac"] is not None:
            pac = ass["pac"]
            v = pac["valid"]
            op_sel = pac["op"]
            print(f"  ── held-out PAC 复核（δ={args.delta}，阈值只在 select 行选，valid 行出下界）──")
            print(f"    select 行 {v['n_select_rows']:,} / valid 行 {v['n_valid_rows']:,} → "
                  f"τR={op_sel['tau_R']:.3f} τM={op_sel['tau_M']:.3f}")
            print(f"    valid D 对：n_acc={v['n_acc']:,}  n_pos={v['n_pos']:,}  TP={v['tp']}")
            print(f"    点估计     precision={v['precision']:.4f}  recall={v['recall']:.4f}")
            print(f"    {1-args.delta:.0%} 置信下界 precision≥{v['precision_lcb']:.4f}  "
                  f"recall≥{v['recall_lcb']:.4f}")
            print(f"    （含 auto 的 set-overlap 端到端 P={pac['e2e']['precision']:.4f} "
                  f"R={pac['e2e']['recall']:.4f} F1={pac['e2e']['f1']:.4f}，仅供参考，非保证口径）")
            out["holdout_pac"][pname] = {"op": op_sel, "valid": v,
                                         "e2e": pac["e2e"],
                                         "delta": args.delta, "valid_frac": args.valid_frac}

        # ── 竞争度分桶的 τR/τM + 分桶端到端（新 C1/C2/C3 路由） ──
        if pname == "neutral":
            ops = {}
            for b in (1, 2):   # C2 / C3
                m = b_id == b
                if m.sum() < 40:
                    ops[b] = None
                    continue
                fr_b = frontier(eta_s[m], ass["rho"]["nb_flex"][m], ys[m], rec_targets, prec_floors,
                                theta=args.theta)
                ops[b] = _pick_op(fr_b, rec_targets, prec_floors)
            b_e2e = bucketed_end_to_end(eta_s, ass["rho"]["nb_flex"], V, pairs, b_id,
                                        gt_set, auto_set, ops, theta=args.theta)
            out["bucketed_end_to_end"][pname] = {"ops": {str(b): (ops[b] if ops.get(b) else None)
                                                          for b in (1, 2)}, **b_e2e}
            no_op = [("C2" if b == 1 else "C3") for b in (1, 2) if ops.get(b) is None]
            print(f"  ── 竞争度分桶端到端（C1直验 + C2/C3 τR/τM + nb_flex） ──")
            if no_op:
                print(f"     ⚠ {','.join(no_op)} 无可行 τR/τM（前沿不可达）→ 该桶未接受任何 pair，"
                      f"结果退化为 auto + C1 直验")
            print(f"    P={b_e2e['precision']:.4f} R={b_e2e['recall']:.4f} F1={b_e2e['f1']:.4f}  "
                  f"查询={b_e2e['n_queries']:,} 匹配={b_e2e['n_matched']:,} "
                  f"(TP={b_e2e['tp']} FP={b_e2e['fp']} FN={b_e2e['fn']})")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products_output",
                        "hybrid_v12.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
