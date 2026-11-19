# -*- coding: utf-8 -*-
"""fdj_struct_buckets.py — 从「数据本身」导出三桶分桶信号（不碰任何跨表匹配分数）。

与 `run_hybrid_join_v12.competition_degree` 的区别：那个用的是 dense/sparse/fused 的
跨表得分（候选数、top1−top2 margin），反映 matcher 的困惑程度；换 embedding 桶就变。
本模块只用令牌/文本结构，同侧可算，换 matcher 桶不变、换数据桶才变。

信号（全部同侧或纯文本，无跨表得分）：

  贫困度 poverty
      token_count / alphanum_ratio / rare_token_ratio（沿用 v10.5 的三个信号）
  分辨力 resolution
      I(r)/|r| = 行内 token 平均自信息，p(t)=df(t)/N 在两表池化语料上估。
      自信息低 = 这行说的话太常见 = 便宜通道没资格裁决它。
  歧义度 fanout
      A(r) = 对侧有多少行与 r 共享「r 在对方表里最稀有的 2 个 token」。
      纯令牌阻塞图的度数，是 1×1 / 1×n / n×m 基数形态的直接读数。
  hubness / 近重复簇
      N_k(r) = 同侧 token-Jaccard kNN 里 r 被点名的次数；dup(r) = 同侧
      Jaccard ≥ dup_thresh 图的连通分量大小。二者都是语义通道洗不干净的高尾来源。

理论（先验，不是保证 —— 保证仍来自 CP）：
    分辨力 vs 歧义度：便宜通道要裁决 r，需 I(r) ≥ log N_other + log(1/δ)。
    pair 的歧义是两端取难 ⇒ min-priority，对 pair 取两端里更难的一端。
    token 实际相关，所以上式只是排序先验；fixed-sequence 的顺序可由先验定，
    先验错只是检验力下降，有效性不变。

分桶：把上述信号按「大者更难」的方向统一后，取各分量百分位的**均匀平均**得 q(r)
（沿用 v10.5 的无魔法权重纪律），再在两表池化分布上按分位数切 C1/C2/C3。
"""
import re
from collections import Counter
from typing import Dict, List

import numpy as np
from scipy import sparse as _sp

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def token_sets(descs):
    return [set(tokenize(d)) for d in descs]


def _percentile_rank(x):
    """经验 CDF（平均秩 / N），落在 (0,1]；并列取平均秩，无自由参数。"""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks / n


# ── 单表信号 ────────────────────────────────────────────────────────────────

def _lexical(sets, df_pool, N_pool, n_rare_frac=0.1):
    """token_count / alphanum_ratio / rare_token_ratio / mean_self_info。"""
    tc = np.zeros(len(sets), dtype=float)
    ar = np.zeros(len(sets), dtype=float)
    rr = np.zeros(len(sets), dtype=float)
    mi = np.zeros(len(sets), dtype=float)
    for i, s in enumerate(sets):
        toks = sorted(s)
        n = len(toks)
        if n == 0:
            continue
        tc[i] = n
        ar[i] = sum(1 for t in toks if re.search(r"[a-z]", t) and re.search(r"\d", t)) / n
        rr[i] = sum(1 for t in toks if df_pool.get(t, 0) < n_rare_frac * N_pool) / n
        mi[i] = sum(-np.log(max(df_pool.get(t, 1), 1) / N_pool) for t in toks) / n
    return dict(token_count=tc, alphanum_ratio=ar, rare_token_ratio=rr, mean_self_info=mi)


def block_fanout(mine, others):
    """A(r) = 对侧与 r 共享「r 在对方表里最稀有的 2 个 token」的行数。

    取对方表内 df 最小的两个 token 求倒排交集；posting 长度由最稀 token 决定，
    所以整体代价被最小倒排表卡住，不会爆。
    """
    df_o = Counter()
    posting = {}
    for j, s in enumerate(others):
        for t in s:
            df_o[t] += 1
            posting.setdefault(t, []).append(j)
    out = np.zeros(len(mine), dtype=float)
    for i, s in enumerate(mine):
        toks = sorted((t for t in s if t in df_o), key=lambda t: df_o[t])
        if len(toks) < 2:
            continue
        p0 = posting[toks[0]]
        s1 = set(posting[toks[1]])
        out[i] = sum(1 for j in p0 if j in s1)
    return out


def near_dup_hubness(sets, k=10, dup_thresh=0.5):
    """同侧 token-Jaccard 空间：hubness N_k(r) 与近重复簇大小 dup(r)。

    返回 (hub, dup)。孤立行 dup=1、hub≈0；变体扎堆的簇 dup 大。
    """
    n = len(sets)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    vocab = {}
    rows, cols = [], []
    for i, s in enumerate(sets):
        for t in s:
            j = vocab.get(t)
            if j is None:
                j = len(vocab)
                vocab[t] = j
            rows.append(i)
            cols.append(j)
    B = _sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(vocab)))
    B.data[:] = 1.0
    inter = np.asarray((B @ B.T).todense())
    size = np.array([len(s) for s in sets], dtype=float)
    union = size[:, None] + size[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(J, -1.0)

    kk = min(k, n - 1)
    if kk > 0:
        nn = np.argpartition(-J, kk - 1, axis=1)[:, :kk]
        hub = np.bincount(nn.ravel(), minlength=n).astype(float)
    else:
        hub = np.zeros(n)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(J >= dup_thresh)
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = Counter(find(i) for i in range(n))
    dup = np.array([comp[find(i)] for i in range(n)], dtype=float)
    return hub, dup


def compute_signals(descs, other_descs, k_hub=10, dup_thresh=0.5):
    """一套完整的同侧结构信号（原始值，未做百分位）。"""
    mine = token_sets(descs)
    others = token_sets(other_descs)
    df_pool = Counter()
    for s in mine + others:
        for t in s:
            df_pool[t] += 1
    N_pool = len(mine) + len(others)
    sig = _lexical(mine, df_pool, N_pool)
    sig["fanout"] = block_fanout(mine, others)
    hub, dup = near_dup_hubness(mine, k=k_hub, dup_thresh=dup_thresh)
    sig["hub"] = hub
    sig["dup"] = dup
    sig["n_other"] = N_pool
    return sig


# ── 信号 → q(r) ─────────────────────────────────────────────────────────────

#: +1 = 大者更难（该行的这些信号越大，越该送 LLM）
ORIENT = {
    "token_count": -1, "alphanum_ratio": -1, "rare_token_ratio": -1,
    "mean_self_info": -1,
    "fanout": +1, "hub": +1, "dup": +1,
}

#: 旧版信号口径（保留 A/B 用；默认结构口径已换成 fdj_cx12 的 12 轴）
MODE_PARTS = {
    "poverty": ["token_count", "alphanum_ratio", "rare_token_ratio"],
    "poverty+hub": ["token_count", "alphanum_ratio", "rare_token_ratio",
                    "mean_self_info", "fanout", "hub", "dup"],
}


def build_q(sigL, sigR, parts):
    """把两表信号熔成 qL/qR ∈ [0,1]（越大越难）。百分位在两表池化分布上取。"""
    qL_parts, qR_parts = [], []
    for p in parts:
        va = np.asarray(sigL[p], dtype=float)
        vb = np.asarray(sigR[p], dtype=float)
        pool = np.concatenate([va, vb])
        R = _percentile_rank(pool)
        rl, rr = R[:len(va)], R[len(va):]
        if ORIENT[p] > 0:          # 大者更难
            qL_parts.append(rl); qR_parts.append(rr)
        else:                      # 小者更难
            qL_parts.append(1.0 - rl); qR_parts.append(1.0 - rr)
    qL = np.mean(qL_parts, axis=0)
    qR = np.mean(qR_parts, axis=0)
    return qL, qR


def classify_rows_jointly(qL, qR, B1, B2):
    """池化分位数切桶（与 v10.5 同口径，无自由阈值）：q ≥ τ1 → C1，τ2 ≤ q < τ1 → C2，余 → C3。"""
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


def resolution_gate(sig, n_other, delta=0.10):
    """理论判据 I(r) ≥ log N_other + log(1/δ)：该行的分辨力够不够唯一锁定。

    这不是分桶规则，是诊断量 —— 用来解释某一桶为什么认证不出来。
    返回 bool 数组。
    """
    need = np.log(max(n_other, 2)) + np.log(1.0 / delta)
    return np.asarray(sig["mean_self_info"], dtype=float) * np.asarray(
        sig["token_count"], dtype=float) >= need


def pair_struct_feats(ci, cj, sigL, sigR):
    """pair 级结构特征，min-priority：歧义取两端更难的一端。

    返回 dict：mi_min（两端较小的分辨力）、fanout_max、hub_max、dup_max。
    """
    mi = np.minimum(np.asarray(sigL["mean_self_info"])[ci],
                    np.asarray(sigR["mean_self_info"])[cj])
    fo = np.maximum(np.asarray(sigL["fanout"])[ci], np.asarray(sigR["fanout"])[cj])
    hb = np.maximum(np.asarray(sigL["hub"])[ci], np.asarray(sigR["hub"])[cj])
    dp = np.maximum(np.asarray(sigL["dup"])[ci], np.asarray(sigR["dup"])[cj])
    return dict(mi_min=mi, fanout_max=fo, hub_max=hb, dup_max=dp)
