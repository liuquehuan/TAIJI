# -*- coding: utf-8 -*-
"""fdj_cx12.py — v10.6 的 12 轴 Structural Complexity Index，抽成独立模块。

从 `run_hybrid_join_v10_6.py` 原样搬来（`_entropy` / `compute_row_features` /
`_STRUCTURAL_AXES` / `compute_structural_complexity`），只把 `tokenize` 与
`_percentile_rank` 改为从 `fdj_struct_buckets` 复用，避免两份实现漂移。

三条轴，12 个特征，全部行级、推理时可观测、不碰 GT / 不碰 matcher 分数 / 不碰 LLM：

  textual    (5) token_count, unique_token_ratio, lexical_entropy,
                 char_entropy, rare_token_ratio
  schema     (4) field_count, missing_rate, name_ratio, free_text_ratio
  identifier (3) numeric_ratio, id_like_ratio, punctuation_ratio

聚合纪律：特征先在**轴内**均匀平均，三条轴再**等权**平均（每轴 1/3，与特征数无关，
所以 identifier 轴不会被 textual 轴的 5 个特征淹掉）。特征先归约成两表池化分布的
百分位秩 R_k，再按轴内符号定向（+1 大者更难 → R_k；-1 小者更难 → 1-R_k）。
没有学习到的权重，唯一的建模选择是每个符号，且逐个写在 `_STRUCTURAL_AXES` 里。
"""
import re
from collections import Counter

import numpy as np

from fdj_struct_buckets import tokenize, _percentile_rank

#: (轴名, [(特征名, 符号)])；符号 +1 = 大者更难 → 贡献 R_k，-1 = 小者更难 → 1 - R_k
_STRUCTURAL_AXES = [
    # ── Textual axis: lexical processing burden / diversity ──
    ("textual", [
        ("token_count",        -1),   # sparse text = harder (structural poverty)
        ("unique_token_ratio", -1),   # repetitive vocabulary = harder
        ("lexical_entropy",    -1),   # low token entropy = harder
        ("char_entropy",       -1),   # low character variety = harder
        ("rare_token_ratio",   -1),   # no distinctive rare tokens = harder
    ]),
    # ── Schema axis: field structure / completeness ──
    ("schema", [
        ("field_count",        -1),   # few fields = poor schema = harder
        ("missing_rate",       +1),   # more missing fields = harder
        ("name_ratio",         -1),   # little structured name = harder
        ("free_text_ratio",    +1),   # more free text = harder
    ]),
    # ── Identifier axis: distinctive markers ──
    ("identifier", [
        ("numeric_ratio",      +1),   # more numeric spec noise = harder
        ("id_like_ratio",      -1),   # no model/SKU numbers = harder
        ("punctuation_ratio",  +1),   # more markup/punctuation = harder
    ]),
]

#: 扁平化的 12 个特征名（顺序与 _STRUCTURAL_AXES 一致）
AXIS_KEYS = [k for _, feats in _STRUCTURAL_AXES for k, _ in feats]


def _entropy(counter):
    """Shannon entropy (bits) of a Counter / iterable of counts."""
    counts = list(counter.values()) if hasattr(counter, "values") else list(counter)
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def compute_row_features(descriptions):
    """Per-row STRUCTURAL statistics for the Structural Complexity Index.

    与 v10.6 逐字一致（含 `desc_preview`），输入是**描述列**原文；v10.6 约定
    行文本形如 "name | free text"，`|` 分隔段即事实上的 schema 字段。
    """
    doc_freq = Counter()
    for desc in descriptions:
        for t in set(tokenize(desc)):
            doc_freq[t] += 1
    n_docs = max(len(descriptions), 1)

    features = []
    for i, desc in enumerate(descriptions):
        text = str(desc)
        tokens = tokenize(text)
        n = len(tokens)

        # ── textual ──
        if n == 0:
            unique_ratio = rare_ratio = numeric_ratio = id_like_ratio = 0.0
            lex_entropy = 0.0
        else:
            unique_ratio = len(set(tokens)) / n
            lex_entropy = _entropy(Counter(tokens))
            rare_ratio = sum(1 for t in tokens if doc_freq.get(t, 0) < n_docs * 0.1) / n
            numeric_ratio = sum(1 for t in tokens if t.isdigit()) / n
            id_like_ratio = sum(1 for t in tokens
                                if re.search(r'[a-z]', t) and re.search(r'\d', t)) / n
        char_entropy = _entropy(Counter(text)) if text else 0.0

        # ── schema（按 "|" 切，事实上的字段分隔符）──
        parts = re.split(r'\s*\|\s*', text)
        nonempty = [p for p in parts if p.strip()]
        field_count = len(nonempty)
        missing_rate = (len(parts) - len(nonempty)) / len(parts) if parts else 0.0
        name_len = len(nonempty[0].strip()) if nonempty else 0
        name_ratio = name_len / len(text) if len(text) else 0.0
        free_text_ratio = 1.0 - name_ratio

        # ── identifier ──
        punc = sum(1 for c in text if not c.isalnum() and not c.isspace())
        punctuation_ratio = punc / len(text) if len(text) else 0.0

        features.append({
            "index": i,
            # textual
            "token_count": n,
            "unique_token_ratio": unique_ratio,
            "lexical_entropy": round(lex_entropy, 4),
            "char_entropy": round(char_entropy, 4),
            "rare_token_ratio": rare_ratio,
            # schema
            "field_count": field_count,
            "missing_rate": missing_rate,
            "name_ratio": name_ratio,
            "free_text_ratio": free_text_ratio,
            # identifier
            "numeric_ratio": numeric_ratio,
            "id_like_ratio": id_like_ratio,
            "punctuation_ratio": punctuation_ratio,
            "desc_preview": text[:120],
        })

    return features


def compute_structural_complexity(features, reference_features=None):
    """Structural Complexity Index q(x) ∈ [0,1] — 多轴、与标签无关。

    JOINT（pair-aware）模式：传 `reference_features`（另一表的行）即按**池化**
    (features ∪ reference_features) 分布算百分位秩，两表共享同一把 q 尺。
    返回与 `features` 对齐的 q。
    """
    keys = AXIS_KEYS
    vals = {k: np.array([f[k] for f in features], dtype=float) for k in keys}
    if reference_features:
        ref = {k: np.array([f[k] for f in reference_features], dtype=float)
               for k in keys}
        ranks = {k: _percentile_rank(np.concatenate([vals[k], ref[k]]))[:len(vals[k])]
                 for k in keys}
    else:
        ranks = {k: _percentile_rank(vals[k]) for k in keys}

    axis_scores = []
    for _axis_name, feats in _STRUCTURAL_AXES:
        contrib = []
        for k, sign in feats:
            R = ranks[k]
            contrib.append(R if sign > 0 else 1.0 - R)
        axis_scores.append(np.mean(np.vstack(contrib), axis=0))
    q = np.mean(np.vstack(axis_scores), axis=0)
    return q


def axis_breakdown(features, reference_features=None):
    """诊断用：返回 {轴名: q_axis} 与 {特征名: R_k}，便于看哪条轴在推高 q。"""
    keys = AXIS_KEYS
    vals = {k: np.array([f[k] for f in features], dtype=float) for k in keys}
    if reference_features:
        ref = {k: np.array([f[k] for f in reference_features], dtype=float)
               for k in keys}
        ranks = {k: _percentile_rank(np.concatenate([vals[k], ref[k]]))[:len(vals[k])]
                 for k in keys}
    else:
        ranks = {k: _percentile_rank(vals[k]) for k in keys}
    axes = {}
    for axis_name, feats in _STRUCTURAL_AXES:
        contrib = [(ranks[k] if sign > 0 else 1.0 - ranks[k]) for k, sign in feats]
        axes[axis_name] = np.mean(np.vstack(contrib), axis=0)
    return axes, ranks
