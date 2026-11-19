# -*- coding: utf-8 -*-
"""run_hybrid_join_v10_10.py — Hybrid Join Router（候选集 + 行级 recall 证书版）。

把 v10.9 的「pair 空间分位数阈值」路由换成「候选集 + 双向 mutual-NN + 争议队列」结构，
并把校准的采样单位从「对」改成「行」（阳性率从 ~0.09% 提到 ~95%，让 recall 认证可行）。

路由（真正的 hybrid —— 并行访问路径的并集，不是串行过滤）：
    候选集 C = 每行 top-k2(dense) ∪ top-k3(sparse) ∪ 每列 top-k2(dense) ∪ top-k3(sparse)
      * 自动接受  —— 两条路径都认为 (i,j) 互为最近邻（mutual-NN），不花 LLM
      * 争议队列  —— C 里其余对，按优先级（noisy-OR 或 fused rank）进 C1 验证
      * C1 验证   —— V1(lenient 宽松屏) → V2(neutral 中性验证)，一对一 + 传播
      * 拒绝      —— C 之外免费丢弃

证书（相对审计模型，深度 K' 之外无法认证）：
    T1  并集优于串行：互补系数 κ = P(M2∧M3)/(ε2·ε3)，κ<1 互补 / ≈1 独立 / →1/max(ε) 冗余
    T2  行级 recall 证书：均匀抽 m 行，ε(k) ≤ Clopper-Pearson 上界（零漏检时 m≥59 证 5%，
        m≥149 证 2%）
    T3  端到端 recall 分解：E[loss] ≤ ε(k) + (1−a) + 2b(k−1)
    T4  最优 k：k* = argmin_k ε(k) + 2b(k−1)

审计模型：--teacher qwen（qwen-plus，= C1 验证器，self-audit）。a/b 因此偏乐观，GT 只做
最后的 validity audit（不进入任何决策）。Jev 路径保留但默认不再使用（geo 受限）。

用法:
    # oracle 模式（LLM-free 验证结构）
    python run_hybrid_join_v10_10.py --dataset products --budget 1500
    # 完整 LLM 模式（密钥内联传入，勿写文件）
    OPENAI_API_KEY=sk-... OPENAI_API_BASE=https://pro.aiskt.com/v1 \
        python run_hybrid_join_v10_10.py --use-llm --dataset products --budget 1500 --calibrate
    # 5 组实验（oracle / 缓存回放 / GT 审计）
    python run_hybrid_join_v10_10.py --experiments
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_PAT = r"[a-z0-9]+"
N_TARGET, TARGET_GT, SEED = 200, 100, 2026

MODEL_ID = "all-MiniLM-L6-v2"
_MODEL_DIR_NAME = "models--sentence-transformers--all-MiniLM-L6-v2"

# 中转站 qwen-plus（openai/ 前缀）。密钥不在这里：join_c1_llm 从环境变量 OPENAI_API_KEY /
# OPENAI_API_BASE 读取。v10.8 的 ENSEMBLE_MODELS 也正是这一档。
LLM_MODEL = ["openai/qwen-plus"]
LLM_TAG = "openai/qwen-plus|neutral|v6"     # v10.8 缓存里的决策 key 前缀（回放用）
# 归一化成本模型（对齐 v10.8 COST_PER_PAIR = {"C1":1.0,"C2":0.01,"C3":0.02} 的语义）：
# 一次强语义（LLM）验证 = 1 单位；一次廉价 dense+sparse 打分摊到全 pair ≈ 0.001/对。
COST_VERIFY, COST_CHEAP = 1.0, 0.001
BATCH = 50

# Jev（OpenRouter decisions model）—— 校准 teacher。密钥只经环境变量传入，绝不写文件。
# 与 v10.8 的 qwen-plus（C1 验证器）分开：Jev 只给校准样本打 label 定阈值，不参与 C1 判断。
JEV_MODEL = "typesafe/jev-1.13"
JEV_API_BASE = "https://openrouter.ai/api/v1"
JEV_QUESTIONS = {
    "is_match": {
        "type": "noul",
        "instructions": "Are Left and Right the same product?",
        "criteria": {
            "true": "Same brand and model, OR one is a fuller/terser listing of the same item.",
            "false": "Different brand, model, or category.",
        },
    }
}

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _cached_model_path():
    roots = [Path.home() / ".cache" / "huggingface" / "hub"]
    for win_root in ("/mnt/c/Users", "C:/Users", "/c/Users"):
        p = Path(win_root)
        if p.is_dir():
            for u in p.iterdir():
                roots.append(u / ".cache" / "huggingface" / "hub")
    for root in roots:
        snap = root / _MODEL_DIR_NAME / "snapshots"
        if snap.is_dir():
            for s in sorted(snap.iterdir(), reverse=True):
                if (s / "model.safetensors").is_file():
                    return s
    return None


def load_embedding_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    local = _cached_model_path()
    if local is not None:
        return SentenceTransformer(str(local))
    return SentenceTransformer(MODEL_ID)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 数据加载（products 不采样；其余 GT-seeded 采样到 200x200）
# ──────────────────────────────────────────────────────────────────────────────
def load_data(ds, data_root="data"):
    left = pd.read_csv(f"{data_root}/{ds}/left.csv")
    right = pd.read_csv(f"{data_root}/{ds}/right.csv")
    gt = pd.read_csv(f"{data_root}/{ds}/ground_truth.csv")
    rng = np.random.RandomState(SEED)
    if ds != "products" and (len(left) > N_TARGET or len(right) > N_TARGET):
        gt_pairs = list(zip(gt["left_id"], gt["right_id"]))
        seed_edges = gt.iloc[rng.choice(len(gt_pairs), size=min(TARGET_GT, len(gt_pairs)),
                                        replace=False)]
        core_l = set(pd.Series(seed_edges["left_id"]).drop_duplicates())
        core_r = set(pd.Series(seed_edges["right_id"]).drop_duplicates())
        fill_l = left[~left["left_id"].isin(core_l)]
        fill_r = right[~right["right_id"].isin(core_r)]
        left = pd.concat([left[left["left_id"].isin(core_l)],
                          fill_l.iloc[rng.choice(len(fill_l), size=max(0, N_TARGET - len(core_l)),
                                                 replace=False)]], ignore_index=True)
        right = pd.concat([right[right["right_id"].isin(core_r)],
                           fill_r.iloc[rng.choice(len(fill_r), size=max(0, N_TARGET - len(core_r)),
                                                  replace=False)]], ignore_index=True)
        lids, rids = set(left["left_id"]), set(right["right_id"])
        gt = gt[gt["left_id"].isin(lids) & gt["right_id"].isin(rids)]
    return left.reset_index(drop=True), right.reset_index(drop=True), gt.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 廉价分：C2 dense + C3 sparse 融合（rank-sum，可加权）
# ──────────────────────────────────────────────────────────────────────────────
def _pct(M):
    x = M.ravel(); order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    return (ranks / len(x)).reshape(M.shape)


def fused_score(left, right, model, w=0.5):
    l_nd = [f"{n} {d}".strip() for n, d in zip(left["left_name"], left["left_description"])]
    r_nd = [f"{n} {d}".strip() for n, d in zip(right["right_name"], right["right_description"])]
    L = model.encode(l_nd, batch_size=128, normalize_embeddings=True)
    R = model.encode(r_nd, batch_size=128, normalize_embeddings=True)
    dense = L @ R.T
    vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PAT, lowercase=True, sublinear_tf=True)
    vec.fit(l_nd + r_nd)
    sparse = np.asarray((vec.transform(l_nd) @ vec.transform(r_nd).T).toarray())
    return w * _pct(dense) + (1 - w) * _pct(sparse), dense, sparse


# ──────────────────────────────────────────────────────────────────────────────
# 3. 一对一 recall 天花板 = GT 最大二分匹配
# ──────────────────────────────────────────────────────────────────────────────
def one_to_one_ceiling(e_l, e_r, nL, nR):
    adj = [[] for _ in range(nL)]
    for a, b in zip(e_l, e_r):
        adj[a].append(b)
    match_r = [-1] * nR

    def kuhn(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or kuhn(match_r[v], seen):
                match_r[v] = u
                return True
        return False
    return sum(kuhn(u, [False] * nR) for u in range(nL))


# ──────────────────────────────────────────────────────────────────────────────
# 4. 三级混合路由核心（accept / C1-LLM / reject + 一对一 + 传播）
# ──────────────────────────────────────────────────────────────────────────────
def _llm_screen(left, right):
    """校验中转站凭据并复用 v10.8 join_c1_llm；返回 (v8, left_ids, right_ids)。"""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import run_hybrid_join_v10_8 as v8
    if not v8.API_KEY or not v8.API_BASE:
        raise SystemExit(
            "[v10.10 --use-llm] 缺少中转站凭据，litellm 会回落到默认端点并卡住。\n"
            "请以内联环境变量运行（密钥只经环境变量传入，绝不写进文件）：\n\n"
            "  OPENAI_API_KEY=sk-... OPENAI_API_BASE=https://pro.aiskt.com/v1 \\\n"
            "      python run_hybrid_join_v10_10.py --use-llm --dataset products --budget 1500"
        )
    v8.LLM_TOKEN_USAGE.update({"prompt": 0, "completion": 0, "total": 0})
    return v8, left["left_id"].values, right["right_id"].values


def make_llm_decider(left, right):
    """C1 = 单段 neutral 验证。decide(ij_pairs) -> [bool yes...]，ij_pairs 是 (i,j) 行索引元组。"""
    v8, left_ids, right_ids = _llm_screen(left, right)
    state = {"t_llm": 0.0, "v8": v8}

    def decide(ij_pairs):
        id_pairs = [(left_ids[i], right_ids[j]) for (i, j) in ij_pairs]
        t0 = time.time()
        yes = v8.join_c1_llm(id_pairs, left, right, model=LLM_MODEL,
                             batch_size=len(id_pairs), post_filter="neutral")
        state["t_llm"] += time.time() - t0
        yes_set = set(yes)
        return [(left_ids[i], right_ids[j]) in yes_set for (i, j) in ij_pairs]
    return decide, state


def make_llm_2stage_decider(left, right):
    """C1 = V1(lenient 宽松屏) → V2(neutral 中性验证) 两段。decide(ij_pairs) -> [bool yes]。

    V1 只筛掉「明显不同」（recall-preserving），V2 对幸存者精确验证（precision-preserving）。
    """
    v8, left_ids, right_ids = _llm_screen(left, right)
    state = {"t_llm": 0.0, "v8": v8}

    def decide(ij_pairs):
        if not ij_pairs:
            return []
        id_pairs = [(left_ids[i], right_ids[j]) for (i, j) in ij_pairs]
        t0 = time.time()
        v1 = set(v8.join_c1_llm(id_pairs, left, right, model=LLM_MODEL,
                                batch_size=len(id_pairs), post_filter="lenient"))
        survivors = [p for p in id_pairs if p in v1]
        v2 = set()
        if survivors:
            v2 = set(v8.join_c1_llm(survivors, left, right, model=LLM_MODEL,
                                    batch_size=len(survivors), post_filter="neutral"))
        state["t_llm"] += time.time() - t0
        return [p in v2 for p in id_pairs]
    return decide, state


# ──────────────────────────────────────────────────────────────────────────────
# 4b. 校准 teacher（Jev，OpenRouter decisions model）+ 分层采样 / IPW 阈值推导
# ──────────────────────────────────────────────────────────────────────────────
def make_jev_decider(left, right, nR):
    """返回 (decide_prob, state)。decide_prob(poss) -> [P(同款) ...]（Jev 校准概率）。

    Jev 是 OpenRouter 的 decisions model，走 /api/alpha/decisions（非 chat/completions），
    输出 calibrated P(同款)。密钥只经环境变量 JEV_API_KEY / JEV_API_BASE / JEV_MODEL 传入，绝不写文件。
    """
    import requests
    sess = requests.Session()
    key = os.environ.get("JEV_API_KEY", "")
    base = os.environ.get("JEV_API_BASE", JEV_API_BASE).rstrip("/")
    model = os.environ.get("JEV_MODEL", JEV_MODEL)
    if not key:
        raise SystemExit(
            "[v10.10 --calibrate] 缺少 JEV_API_KEY。请以内联环境变量运行（密钥只经环境变量传入，绝不写文件）：\n\n"
            "  JEV_API_KEY=sk-or-v1-... python run_hybrid_join_v10_10.py --calibrate --dataset products"
        )
    dec_url = base.replace("/api/v1", "") + "/api/alpha/decisions"
    ln = left["left_name"].astype(str).values
    ld = left["left_description"].astype(str).values
    rn = right["right_name"].astype(str).values
    rd = right["right_description"].astype(str).values
    state = {"t": 0.0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0, "n": 0,
             "errors": 0, "call": 0, "r429": 0, "wall0": time.time()}

    def decide_prob(poss):
        probs = []
        for p in poss:
            i, j = p // nR, p % nR
            st = {"left": f"{ln[i]} — {ld[i][:300]}", "right": f"{rn[j]} — {rd[j][:300]}"}
            last_err = "unknown"
            for attempt in range(4):
                try:
                    t0 = time.time()
                    r = sess.post(dec_url,
                                  headers={"Authorization": f"Bearer {key}",
                                           "Content-Type": "application/json"},
                                  json={"model": model, "state": st, "questions": JEV_QUESTIONS},
                                  timeout=60)
                    state["t"] += time.time() - t0
                    if r.status_code == 429:
                        state["r429"] += 1
                        print(f"    [Jev] 429 限流 ×{state['r429']}（backoff {2 * (attempt + 1)}s）")
                        time.sleep(2 * (attempt + 1))
                        continue
                    if r.status_code != 200:
                        # 致命配置错误（401/400/403 等）：立刻中止，别把「0」静默灌进校准
                        raise SystemExit(f"[Jev] HTTP {r.status_code}：{r.text[:300]}")
                    d = r.json()
                    pr = d.get("answers", {}).get("is_match", {}).get("noul")
                    if pr is None:
                        raise SystemExit(f"[Jev] 响应缺 noul：{json.dumps(d)[:300]}")
                    u = d.get("usage") or {}
                    state["in_tokens"] += u.get("input_tokens", 0) or 0
                    state["out_tokens"] += u.get("output_tokens", 0) or 0
                    state["cost"] += u.get("cost", 0) or 0
                    state["n"] += 1
                    probs.append(float(pr))
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    time.sleep(1)
            else:
                # 4 次全失败（持续 429 或网络错误）：中止，避免静默产出垃圾校准
                raise SystemExit(f"[Jev] 连续 4 次失败，中止：{last_err}")
            state["call"] += 1
            if state["call"] % 100 == 0:
                wall = time.time() - state["wall0"]
                print(f"    [Jev] 已标注 {state['call']} 对  墙钟 {wall:.0f}s "
                      f"(≈{wall / state['call']:.2f}s/对)  429×{state['r429']}")
        return probs
    return decide_prob, state


def make_qwen_teacher_decider(left, right, nR):
    """qwen-plus 校准 teacher（中转站，chat/completions，post_filter='teacher'）。

    返回 (decide_prob, state)。decide_prob(poss) -> [1.0/0.0 硬标签]。
    用于 --teacher qwen：Jev 在某些区域被 OpenRouter 的 geo 限制（403）时，退回中转站 qwen-plus。
    提示词用 v10.8 的 teacher 口径（与 Jev 的校准口径一致）。
    """
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import run_hybrid_join_v10_8 as v8
    if not v8.API_KEY or not v8.API_BASE:
        raise SystemExit(
            "[v10.10 --teacher qwen] 缺少中转站凭据。请以内联环境变量运行：\n\n"
            "  OPENAI_API_KEY=sk-... OPENAI_API_BASE=https://pro.aiskt.com/v1 \\\n"
            "      python run_hybrid_join_v10_10.py --calibrate --teacher qwen --dataset products"
        )
    v8.LLM_TOKEN_USAGE.update({"prompt": 0, "completion": 0, "total": 0})
    left_ids = left["left_id"].values
    right_ids = right["right_id"].values
    state = {"t": 0.0, "v8": v8, "n": 0, "call": 0, "wall0": time.time()}

    def decide_prob(poss):
        probs = [0.0] * len(poss)
        for k in range(0, len(poss), 50):
            chunk = poss[k:k + 50]
            id_pairs = [(left_ids[p // nR], right_ids[p % nR]) for p in chunk]
            t0 = time.time()
            yes = v8.join_c1_llm(id_pairs, left, right, model=LLM_MODEL,
                                 batch_size=len(id_pairs), post_filter="teacher")
            state["t"] += time.time() - t0
            yes_set = set(yes)
            for idx, p in enumerate(chunk):
                probs[k + idx] = 1.0 if (left_ids[p // nR], right_ids[p % nR]) in yes_set else 0.0
            state["n"] += len(chunk)
            state["call"] += len(chunk)
            if state["call"] % 100 == 0:
                wall = time.time() - state["wall0"]
                print(f"    [qwen teacher] 已标注 {state['call']} 对  墙钟 {wall:.0f}s "
                      f"(≈{wall / state['call']:.2f}s/对)")
        state["tokens"] = dict(v8.LLM_TOKEN_USAGE)
        return probs
    return decide_prob, state


def teacher_calibrate(fused_flat, nL, nR, n_all, jev_decide, n_cal=1200, pi0=0.90, alpha=0.05):
    """分层采样 + Jev 打 label + IPW(Hájek) → 推导 teacher-relative 阈值 τ_M / τ_R。

    分层：按 fused 分位数把全 pair space 切成 B 层（顶部加密，因匹配是集中在高分的稀有事件）。
    抽样：每层等量 n_cal//B（balanced），包含概率 π_k = n_b/N_b，IPW 权重 w_k = 1/π_k。
    估计：Hájek 加权比例 + 线性化方差下界（保守）。τ_M 是「teacher 精度下界 ≥ π0 可认证的
    最宽松 accept 阈值」，τ_R 是「teacher 正例分数分布的 α 下尾分位」（teacher-relative reject）。

    注意：τ_M/τ_R 都是「相对 teacher(Jev)」的证书，不是 human-GT 绝对保证。
    """
    rng = np.random.RandomState(2026)
    edges_pct = [0, 80, 90, 95, 97.5, 99, 99.5, 99.9, 99.95, 99.99, 100]
    edges = np.percentile(fused_flat, edges_pct)
    B = len(edges_pct) - 1
    n_per = max(1, n_cal // B)

    sampled = []  # [pos, score, w, b]
    strata_meta = []  # [lo_pct, hi_pct, N_b, n_b]
    for b in range(B):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            idx = np.where(fused_flat <= hi)[0]
        elif b == B - 1:
            idx = np.where(fused_flat > lo)[0]
        else:
            idx = np.where((fused_flat > lo) & (fused_flat <= hi))[0]
        N_b = len(idx)
        n_b = min(n_per, N_b)
        strata_meta.append([edges_pct[b], edges_pct[b + 1], N_b, n_b])
        if n_b == 0:
            continue
        pick = idx[rng.choice(N_b, size=n_b, replace=False)]
        w = N_b / n_b
        for pos in pick:
            sampled.append((int(pos), float(fused_flat[pos]), w, b))

    poss = [s[0] for s in sampled]
    probs = jev_decide(poss)
    scores = np.array([s[1] for s in sampled])
    weights = np.array([s[2] for s in sampled])
    sb = np.array([s[3] for s in sampled])
    labels = np.array([1.0 if p >= 0.5 else 0.0 for p in probs])

    strata = []
    for b in range(B):
        lo, hi, Nb, nb = strata_meta[b]
        nm = int(labels[sb == b].sum()) if nb > 0 else 0
        strata.append((lo, hi, Nb, nb, nm))

    n_pos = int(labels.sum())
    match_rate_ipw = float((weights * labels).sum() / weights.sum())
    prior = 1.0 / max(nL, nR)

    # ── τ_M：精度下限（Hájek PPV + 线性化方差下界） ──
    order_s = np.argsort(-scores, kind="stable")
    sc = scores[order_s]
    lb_ = labels[order_s]
    w_ = weights[order_s]
    cum_w = np.cumsum(w_)
    cum_wy = np.cumsum(w_ * lb_)
    ppv = cum_wy / cum_w
    var = np.cumsum(w_ ** 2 * (lb_ - ppv) ** 2) / (cum_w ** 2)
    se = np.sqrt(np.maximum(var, 0))
    lb_ppv = ppv - 1.96 * se
    cert_idx = np.where(lb_ppv >= pi0)[0]
    if len(cert_idx) == 0:
        tau_M = float("inf")
        ppv_at_tauM, lb_at_tauM = 0.0, 0.0
    else:
        k = int(cert_idx[-1])
        tau_M = float(sc[k])
        ppv_at_tauM, lb_at_tauM = float(ppv[k]), float(lb_ppv[k])

    # ── τ_R：teacher 正例分数分布的 α 下尾分位（IPW 加权） ──
    if n_pos == 0:
        tau_R = float("inf")  # 无正例 → 无法定 reject 边界（退化）
    else:
        pm = labels == 1
        ps = scores[pm]
        pw = weights[pm]
        op = np.argsort(ps, kind="stable")
        cs = ps[op]
        cw = np.cumsum(pw[op])
        cw /= cw[-1]
        tau_R = float(cs[np.searchsorted(cw, alpha, side="left")])

    return {
        "tau_M": tau_M, "tau_R": tau_R, "pi0": pi0, "alpha": alpha,
        "n_cal": len(sampled), "n_pos": n_pos,
        "ppv_at_tauM": ppv_at_tauM, "lb_ppv_at_tauM": lb_at_tauM,
        "match_rate_ipw": match_rate_ipw, "prior": prior,
        "strata": strata, "B": B,
        "tauM_pct": float(np.mean(fused_flat >= tau_M) * 100) if np.isfinite(tau_M) else 0.0,
        "tauR_pct": float(np.mean(fused_flat < tau_R) * 100) if np.isfinite(tau_R) else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4c. hybrid 结构：候选集 + mutual-NN + 争议队列 + 行级证书（T1-T4）
# ──────────────────────────────────────────────────────────────────────────────
def cp_upper_fail(k_fail, n, alpha):
    """Clopper-Pearson 单侧上界：n 次中 k_fail 次失败，失败率 p 的 (1-alpha) 上界。"""
    import scipy.stats as _st
    if n <= 0 or k_fail >= n:
        return 1.0
    return float(_st.beta.ppf(1.0 - alpha, k_fail + 1, n - k_fail))


def build_candidates(dense, sparse, k2, k3):
    """候选集 C = 行 top-k2(dense) ∪ 行 top-k3(sparse) ∪ 列 top-k2(dense) ∪ 列 top-k3(sparse)。
    返回 (row_cand, col_cand, cand)，cand 是 set of (i,j)。"""
    nL, nR = dense.shape
    row_cand = [set() for _ in range(nL)]
    col_cand = [set() for _ in range(nR)]
    for S, k, axis, sink in ((dense, k2, 1, row_cand), (sparse, k3, 1, row_cand),
                             (dense.T, k2, 0, col_cand), (sparse.T, k3, 0, col_cand)):
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


def mutual_nn_agree(dense, sparse):
    """auto-accept：两条路径都认为 (i,j) 互为 top-1 最近邻。返回 set of (i,j)。"""
    nL, nR = dense.shape
    drow = np.argmax(dense, axis=1)
    dcol = np.argmax(dense, axis=0)
    srow = np.argmax(sparse, axis=1)
    scol = np.argmax(sparse, axis=0)
    auto = set()
    for i in range(nL):
        j = int(drow[i])
        if int(dcol[j]) == i and int(srow[i]) == j and int(scol[j]) == i:
            auto.add((i, j))
    return auto


def oracle_row_coverage(dense, sparse, e_l, e_r, k2, k3):
    """oracle 行级覆盖：每个 GT 左行的真搭档是否在候选集。返回 (miss bool 数组, miss_rate)。"""
    row_cand, col_cand, _ = build_candidates(dense, sparse, k2, k3)
    miss = np.zeros(len(e_l), dtype=bool)
    for idx in range(len(e_l)):
        i, j = int(e_l[idx]), int(e_r[idx])
        miss[idx] = (j not in row_cand[i]) and (i not in col_cand[j])
    return miss, float(miss.mean())


def candidate_priority_order(cand, auto_accept, fused):
    """按 fused 分数从高到低返回候选列表（排除 auto_accept）。"""
    rest = [p for p in cand if p not in auto_accept]
    return sorted(rest, key=lambda p: fused[p[0], p[1]], reverse=True)


def hybrid_router_candidates(priority_order, auto_accept, budget, decide, nL, nR, batch=BATCH):
    """候选集路由：auto 直接确认，其余按优先级进 C1（一对一 + 传播）。
    decide(ij_pairs) -> [bool yes...]（oracle 或 LLM 两段）。"""
    claimed_l = np.zeros(nL, dtype=bool)
    claimed_r = np.zeros(nR, dtype=bool)
    matched = list(auto_accept)
    for (i, j) in auto_accept:
        claimed_l[i] = True
        claimed_r[j] = True
    pending = []
    llm_verified = 0
    pruned = 0

    def flush():
        if not pending:
            return
        verdicts = decide(pending)
        for (pos, yes) in zip(pending, verdicts):
            i, j = pos
            if yes and not claimed_l[i] and not claimed_r[j]:
                matched.append((i, j))
                claimed_l[i] = True
                claimed_r[j] = True
        pending.clear()

    for (i, j) in priority_order:
        if llm_verified >= budget:
            break
        if claimed_l[i] or claimed_r[j]:
            pruned += 1
            continue
        pending.append((i, j))
        llm_verified += 1
        if len(pending) >= batch:
            flush()
    flush()
    return {"matched": matched, "auto_accept": list(auto_accept),
            "llm_verified": llm_verified, "pruned": pruned}


def estimate_ab_from_cache(cache, gt_pos, cand, left_ids, right_ids):
    """用缓存 LLM 决策估计验证器 a/b（不花新 LLM）。a=P(yes|match), b=P(yes|non-match)，在候选集 cand 上。"""
    tag = LLM_TAG
    tp = fn = fp = tn = 0
    for (i, j) in cand:
        y = (i, j) in gt_pos
        d = cache.get(f"{tag}|{left_ids[i]}|{right_ids[j]}") == "yes"
        tp += (y and d); fn += (y and not d); fp += (not y and d); tn += (not y and not d)
    a = tp / (tp + fn) if (tp + fn) else 1.0
    b = fp / (fp + tn) if (fp + tn) else 0.0
    return a, b, (tp + fn), (fp + tn)


def t3_bound(eps_k, a, b, k):
    """T3: E[recall loss] ≤ ε(k) + (1−a) + 2b(k−1)。"""
    return eps_k + (1.0 - a) + 2.0 * b * (k - 1)


def optimal_k(eps, b, ks):
    """T4: k* = argmin_k ε(k) + 2b(k−1)。eps 是 dict k->miss_rate。"""
    return min(ks, key=lambda k: eps[k] + 2.0 * b * (k - 1))


def row_calibrate(dense, sparse, left, right, nL, nR, teacher_decide, m=200, k=10, delta=0.05):
    """行采样校准（self-audit teacher=qwen-plus）。返回证书 dict。

    ε(k)：均匀抽 m 行，X_i = teacher 在候选集里找不到该行搭档 → CP 上界（T2）。
    self-audit 下 ε 是「相对 teacher」的：teacher 自己漏判会被计入 ε，故偏保守（上界偏高）。
    """
    rng = np.random.RandomState(2026)
    rows = rng.choice(nL, size=min(m, nL), replace=False)
    row_cand, col_cand, _ = build_candidates(dense, sparse, k, k)

    label_pairs = set()
    pair_of_row = {}
    for i in rows:
        i = int(i)
        ps = set((i, j) for j in row_cand[i])
        for j in list(row_cand[i]):
            ps.update((i2, j) for i2 in col_cand[j])
        label_pairs.update(ps)
        pair_of_row[i] = ps

    lp = list(label_pairs)
    flat = [i * nR + j for (i, j) in lp]
    probs = teacher_decide(flat) if flat else []
    match_set = set(lp[t] for t in range(len(lp)) if probs[t] >= 0.5)

    X = []
    for i in rows:
        i = int(i)
        X.append(0 if any(p in match_set for p in pair_of_row[i]) else 1)
    k_fail = int(sum(X))
    n_rows = len(X)
    eps_hat = k_fail / n_rows if n_rows else 1.0
    eps_ub = cp_upper_fail(k_fail, n_rows, delta)
    return {"eps_hat": eps_hat, "eps_ub": eps_ub, "n_rows": n_rows, "k_fail": k_fail,
            "n_labeled": len(lp)}


def hybrid_router(order, fused_flat, nL, nR, accept_thresh, reject_thresh, budget, decide,
                  batch=BATCH):
    """三级路由 + 一对一 + 传播。decide(poss) -> [bool yes...]（oracle 或 LLM）。

    返回 dict：matched（全部确认 (i,j)）、auto_accept（自动接受的 (i,j)）、
    llm_verified（LLM 验证次数）、pruned（传播剪枝次数）。
    """
    claimed_l = np.zeros(nL, dtype=bool)
    claimed_r = np.zeros(nR, dtype=bool)
    matched, auto_accept = [], []
    pending = []
    llm_verified, pruned = 0, 0

    def flush():
        if not pending:
            return
        verdicts = decide(pending)
        for pos, yes in zip(pending, verdicts):
            i, j = pos // nR, pos % nR
            # 严格一对一：同一批内若已被更靠前的 pair 占了行/列，则不再重复确认
            if yes and not claimed_l[i] and not claimed_r[j]:
                matched.append((i, j))
                claimed_l[i] = True
                claimed_r[j] = True
        pending.clear()

    for pos in order:
        s = fused_flat[pos]
        if s < reject_thresh:
            break                       # reject 层：尾部整段免费丢弃
        i, j = pos // nR, pos % nR
        if claimed_l[i] or claimed_r[j]:
            pruned += 1
            continue
        if s >= accept_thresh:
            auto_accept.append((i, j))   # accept 层：C2/C3 高置信直接判 match
            matched.append((i, j))
            claimed_l[i] = True
            claimed_r[j] = True
        else:
            if llm_verified >= budget:
                break                   # C1 预算耗尽，其余中间对作废
            pending.append(pos)
            llm_verified += 1
            if len(pending) >= batch:
                flush()
    flush()
    return {"matched": matched, "auto_accept": auto_accept,
            "llm_verified": llm_verified, "pruned": pruned}


def eval_positions(matched, is_gt, nR, n_gt):
    ms = set(matched)
    tp = sum(1 for (i, j) in ms if is_gt[i * nR + j])
    fp = len(ms) - tp
    fn = n_gt - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────────
# 5. 实验辅助（E1 基线 / E2 消融 / E3-E4 回放）
# ──────────────────────────────────────────────────────────────────────────────
def greedy_1to1(order, is_gt, nL, nR, budget=None):
    claimed_l = np.zeros(nL, dtype=bool)
    claimed_r = np.zeros(nR, dtype=bool)
    confirmed_at, pruned_at = [], []
    confirmed = verified = pruned = 0
    for pos in order:
        if budget is not None and verified >= budget:
            break
        i, j = pos // nR, pos % nR
        if claimed_l[i] or claimed_r[j]:
            pruned += 1
            continue
        verified += 1
        if is_gt[pos]:
            claimed_l[i] = True
            claimed_r[j] = True
            confirmed += 1
        confirmed_at.append(confirmed)
        pruned_at.append(pruned)
    return (np.array(confirmed_at), np.array(pruned_at), confirmed, verified, pruned)


def global_rec_at(order, is_gt, k):
    return int(np.cumsum(is_gt[order])[k - 1])


def per_row_topk_recall(S, is_gt, nL, nR, k):
    hit = 0
    for i in range(nL):
        top = np.argsort(-S[i], kind="stable")[:k]
        hit += int(np.any(is_gt[i * nR + top]))
    return hit


def hungarian_recall(S, is_gt, nL, nR, e_l, e_r):
    if not _HAS_SCIPY:
        return None, None
    ri, ci = linear_sum_assignment(-S)
    matched = set(zip(ri, ci))
    gt_set = set(zip(e_l, e_r))
    return sum(1 for p in matched if p in gt_set), len(ri)


def mutual_nn_recall(S, is_gt, nL, nR, e_l, e_r):
    best_l = np.argmax(S, axis=1)
    best_r = np.argmax(S, axis=0)
    mutual = set()
    for i in range(nL):
        j = best_l[i]
        if best_r[j] == i:
            mutual.add((i, j))
    gt_set = set(zip(e_l, e_r))
    return sum(1 for p in mutual if p in gt_set), len(mutual)


def replay_diagnostic(order, left, right, gt, nL, nR, budget, cache, batch=BATCH):
    """回放 v10.9 的贪心+传播（含 batch 语义），用缓存决策代替 LLM，归因 GT 边命运。"""
    left_ids = left["left_id"].values
    right_ids = right["right_id"].values
    l_idx_of = {str(v): k for k, v in enumerate(left_ids)}
    r_idx_of = {str(v): k for k, v in enumerate(right_ids)}
    gt_pos = set()
    for a, b in zip(gt["left_id"], gt["right_id"]):
        gt_pos.add(l_idx_of[str(a)] * nR + r_idx_of[str(b)])

    claim_l = np.full(nL, -1, dtype=np.int64)
    claim_r = np.full(nR, -1, dtype=np.int64)
    fate, yes_seq, tp_seq = {}, [], []
    pending = []
    verified = 0
    tp = 0

    def flush():
        nonlocal tp
        for (pos, i, j) in pending:
            yes = (cache.get(f"{LLM_TAG}|{left_ids[i]}|{right_ids[j]}") == "yes")
            yes_seq.append(yes)
            if yes and claim_l[i] == -1 and claim_r[j] == -1:
                claim_l[i] = pos
                claim_r[j] = pos
                if pos in gt_pos:
                    fate[pos] = "confirmed"
                    tp += 1
            elif yes and pos in gt_pos:
                # 判 Yes 但同批更靠前的 pair 已占行/列 -> 被占位
                c = claim_l[i] if claim_l[i] != -1 else claim_r[j]
                fate[pos] = "blocked_fp" if c not in gt_pos else "blocked_tp"
            elif pos in gt_pos:
                fate[pos] = "llm_no"
            tp_seq.append(tp)
        pending.clear()

    for pos in order:
        if verified >= budget:
            break
        i, j = pos // nR, pos % nR
        claimed = claim_l[i] != -1 or claim_r[j] != -1
        if claimed:
            if pos in gt_pos:
                c = claim_l[i] if claim_l[i] != -1 else claim_r[j]
                fate[pos] = "blocked_fp" if c not in gt_pos else "blocked_tp"
            continue
        pending.append((pos, i, j))
        verified += 1
        if len(pending) >= batch:
            flush()
    flush()
    for pos in gt_pos:
        fate.setdefault(pos, "not_reached")
    return fate, yes_seq, tp_seq


def stopping_rule(yes_seq, window=100, min_yes=2):
    run = 0
    for t in range(len(yes_seq)):
        run += yes_seq[t]
        if t >= window:
            run -= yes_seq[t - window]
        if t + 1 >= window and run < min_yes:
            return t + 1
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 6. main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="products")
    ap.add_argument("--budget", type=int, default=1500, help="C1(LLM) 验证预算（争议队列对数量上限）")
    ap.add_argument("--weight", type=float, default=0.5, help="融合权重 w*dense+(1-w)*sparse（只用于争议队列优先级）")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--k2", type=int, default=10, help="dense 候选深度（每行/列 top-k2）")
    ap.add_argument("--k3", type=int, default=10, help="sparse 候选深度（每行/列 top-k3）")
    ap.add_argument("--use-llm", action="store_true", help="C1 = qwen-plus 两段 V1(lenient)→V2(neutral)")
    ap.add_argument("--experiments", action="store_true", help="额外跑 5 组实验")
    ap.add_argument("--calibrate", action="store_true",
                    help="行采样校准（self-audit teacher）推导 ε(k) 的 CP 上界")
    ap.add_argument("--teacher", choices=["jev", "qwen"], default="qwen",
                    help="校准 teacher：qwen(qwen-plus 中转站，默认) / jev(geo 受限)")
    ap.add_argument("--m", type=int, default=200, help="校准行采样数")
    ap.add_argument("--delta", type=float, default=0.05, help="CP 置信 1-delta")
    args = ap.parse_args()
    t0 = time.time()

    left, right, gt = load_data(args.dataset)
    nL, nR, n_gt = len(left), len(right), len(gt)
    n_all = nL * nR
    l_idx_of = {str(v): k for k, v in enumerate(left["left_id"])}
    r_idx_of = {str(v): k for k, v in enumerate(right["right_id"])}
    e_l = np.array([l_idx_of[str(a)] for a in gt["left_id"]])
    e_r = np.array([r_idx_of[str(b)] for b in gt["right_id"]])
    is_gt = np.zeros(n_all, dtype=bool)
    for a, b in zip(e_l, e_r):
        is_gt[a * nR + b] = True

    model = load_embedding_model()
    t_cheap0 = time.time()
    fused, dense, sparse = fused_score(left, right, model, args.weight)
    t_cheap = time.time() - t_cheap0
    fused_flat = fused.ravel()
    order = np.argsort(-fused_flat, kind="stable")

    ceil = one_to_one_ceiling(e_l.tolist(), e_r.tolist(), nL, nR)

    # ── hybrid 结构：候选集 + mutual-NN + 争议队列 ──
    row_cand, col_cand, cand = build_candidates(dense, sparse, args.k2, args.k3)
    auto = mutual_nn_agree(dense, sparse)
    auto_in_cand = sorted(auto)                    # mutual-NN ⊆ 候选集（行/列 top-1 必在 C 内）
    priority = candidate_priority_order(cand, auto, fused)
    _miss_arr, eps_true = oracle_row_coverage(dense, sparse, e_l, e_r, args.k2, args.k3)

    # ── 校准（self-audit teacher） ──
    cal = None
    teacher_state = None
    if args.calibrate:
        if args.teacher == "qwen":
            teacher_decide, teacher_state = make_qwen_teacher_decider(left, right, nR)
        else:
            teacher_decide, teacher_state = make_jev_decider(left, right, nR)
        cal = row_calibrate(dense, sparse, left, right, nL, nR, teacher_decide,
                            m=args.m, k=max(args.k2, args.k3), delta=args.delta)

    # ── C1 验证器 ──
    if args.use_llm:
        decide, llm_state = make_llm_2stage_decider(left, right)
        verifier = "LLM(qwen-plus) V1→V2"
    else:
        decide = lambda ij: [bool(is_gt[i * nR + j]) for (i, j) in ij]
        llm_state = None
        verifier = "oracle(GT)"

    print(f"\n{'='*72}")
    print(f"  Hybrid Join Router (v10.10) — {args.dataset}  {nL}L x {nR}R = {n_all:,} pairs, {n_gt} GT")
    print(f"  C1={verifier} | C2=dense top-{args.k2} | C3=sparse top-{args.k3} | budget {args.budget}")
    print(f"  候选集 |C|={len(cand):,}  自动接受(mutual-NN)={len(auto_in_cand):,}")
    print(f"  one-to-one ceiling = {ceil}/{n_gt} = {ceil/n_gt:.4f}")
    print(f"{'='*72}")

    res = hybrid_router_candidates(priority, auto_in_cand, args.budget, decide, nL, nR, args.batch)
    matched, auto_accept = res["matched"], res["auto_accept"]
    llm_verified, pruned = res["llm_verified"], res["pruned"]
    auto_pos = set(auto_accept)
    llm_matched = [p for p in matched if p not in auto_pos]

    auto_tp = sum(1 for (i, j) in auto_accept if is_gt[i * nR + j])
    auto_prec = auto_tp / len(auto_accept) if auto_accept else 0.0
    metrics = eval_positions(matched, is_gt, nR, n_gt)

    print(f"\n  ── 路由分解 ──")
    print(f"    自动接受 (mutual-NN)  = {len(auto_accept):>8,}  precision={auto_prec:.4f}")
    print(f"    争议队列 C1 验证       = {llm_verified:>8,}  确认 {len(llm_matched):,}")
    print(f"    传播剪枝 (免费)        = {pruned:>8,}")
    print(f"    候选外拒绝 (免费)      = {n_all - len(cand):>8,}")
    print(f"    最终匹配 = {len(matched):,}（auto {len(auto_accept):,} + C1 {len(llm_matched):,}）")

    print(f"\n  ── 端到端评估 ──")
    print(f"    Precision = {metrics['precision']:.4f}")
    print(f"    Recall    = {metrics['recall']:.4f}   (oracle ceiling {ceil/n_gt:.4f})")
    print(f"    F1        = {metrics['f1']:.4f}")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")

    print(f"\n  ── 证书（T2，oracle 行覆盖） ──")
    print(f"    ε(k={args.k2}/{args.k3}) = {eps_true:.4f}（真值，oracle）")
    if cal is not None:
        print(f"    ε̂(teacher 自评) = {cal['eps_hat']:.4f}  CP 上界(1-δ={1-args.delta}) = {cal['eps_ub']:.4f}"
              f"  （{cal['n_rows']} 行，{cal['k_fail']} 漏，标注 {cal['n_labeled']} 对）")
        print(f"    诚实边界：ε̂/CP 是「相对 teacher({args.teacher})」的证书，不是 human-GT 绝对保证。")

    cost = llm_verified * COST_VERIFY + n_all * COST_CHEAP
    print(f"\n  ── cost / token / latency ──")
    print(f"    normalized cost = {llm_verified}*{COST_VERIFY} (C1) + {n_all:,}*{COST_CHEAP} "
          f"(cheap) = {cost:,.1f}")
    if args.use_llm:
        tokens = dict(llm_state["v8"].LLM_TOKEN_USAGE)
        print(f"    tokens: prompt={tokens['prompt']} completion={tokens['completion']} "
              f"total={tokens['total']}")
        print(f"    latency: cheap={t_cheap:.1f}s  llm={llm_state['t_llm']:.1f}s  "
              f"total={time.time()-t0:.1f}s")
    else:
        print(f"    latency: cheap={t_cheap:.1f}s  total={time.time()-t0:.1f}s")

    out_dir = os.path.join(PROJECT_ROOT, "products_output")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_out = {"version": "v10.10", "mode": "llm" if args.use_llm else "oracle",
               "dataset": args.dataset, "nL": nL, "nR": nR, "n_gt": n_gt,
               "k2": args.k2, "k3": args.k3, "weight": args.weight,
               "budget": args.budget, "ceiling": ceil, "ceiling_recall": ceil / n_gt,
               "candidate_set": len(cand), "auto_accept": len(auto_accept),
               "eps_true": eps_true,
               "tiers": {"auto_accept": len(auto_accept), "auto_precision": auto_prec,
                         "llm_verified": llm_verified, "llm_matched": len(llm_matched),
                         "pruned": pruned},
               "evaluation": metrics, "cost": {"normalized": cost, "llm_verified": llm_verified},
               "runtime_s": time.time() - t0}
    if cal is not None:
        res_out["calibration"] = {"teacher": args.teacher, "eps_hat": cal["eps_hat"],
                                  "eps_ub": cal["eps_ub"], "n_rows": cal["n_rows"],
                                  "k_fail": cal["k_fail"], "n_labeled": cal["n_labeled"]}
    with open(os.path.join(out_dir, f"v10_10_{args.dataset}_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(res_out, f, indent=2)

    # ── 5 组实验（LLM-free / 缓存回放） ──
    if args.experiments:
        run_experiments(args, left, right, gt, fused, dense, sparse, is_gt, order,
                        fused_flat, nL, nR, n_gt, n_all, ceil, e_l, e_r)
    print(f"\n  wrote products_output/v10_10_{args.dataset}_{ts}.json  [{time.time()-t0:.1f}s]")


def run_experiments(args, left, right, gt, fused, dense, sparse, is_gt, order, fused_flat,
                    nL, nR, n_gt, n_all, ceil, e_l, e_r):
    ks = [1, 2, 3, 5, 10, 20, 50]
    gt_pos = set(zip(e_l.tolist(), e_r.tolist()))
    left_ids = left["left_id"].values
    right_ids = right["right_id"].values

    # E1: recall@k + κ（oracle，并集候选互补性）
    print(f"\n=== E1 recall@k 与互补系数 κ（oracle，并集候选） ===")
    print(f"  {'k':>4}{'dense':>9}{'sparse':>9}{'union':>9}{'κ':>8}{'冗余度':>8}")
    for k in ks:
        m2, _ = oracle_row_coverage(dense, sparse, e_l, e_r, k, 0)
        m3, _ = oracle_row_coverage(dense, sparse, e_l, e_r, 0, k)
        mu, _ = oracle_row_coverage(dense, sparse, e_l, e_r, k, k)
        e2, e3, eu = float(m2.mean()), float(m3.mean()), float(mu.mean())
        if eu == 0:
            kappa_s, red_s = "饱和", "饱和"
        elif e2 * e3 == 0:
            kappa_s, red_s = "nan", "nan"
        else:
            kappa = eu / (e2 * e3)
            ub = 1.0 / max(e2, e3) if max(e2, e3) > 0 else float("nan")
            red = (kappa - 1.0) / (ub - 1.0) if np.isfinite(ub) and ub > 1 else float("nan")
            kappa_s, red_s = f"{kappa:.3f}", f"{red:.1%}"
        print(f"  {k:>4}{1-e2:>9.4f}{1-e3:>9.4f}{1-eu:>9.4f}{kappa_s:>8}{red_s:>8}")

    # E2: 100 种子覆盖（CP 上界 vs 真 ε(k)）
    m = min(args.m, nL)
    row_cand, col_cand, cand = build_candidates(dense, sparse, args.k2, args.k3)
    from collections import defaultdict
    partners = defaultdict(list)
    for a, b in zip(e_l, e_r):
        partners[int(a)].append(int(b))
    gt_rows = sorted(partners)
    row_miss = np.array([
        int(any((j not in row_cand[i]) and (i not in col_cand[j]) for j in partners[i]))
        for i in gt_rows])
    eps_pop = float(row_miss.mean())
    rng = np.random.RandomState(0)
    viol = 0
    n_samp = min(m, len(gt_rows))
    for _ in range(100):
        idx = rng.choice(len(gt_rows), size=n_samp, replace=False)
        fails = int(row_miss[idx].sum())
        ub = cp_upper_fail(fails, n_samp, args.delta)
        viol += (eps_pop > ub)
    print(f"\n=== E2 100 种子覆盖：CP 上界 vs 真 ε(k)（δ={args.delta}） ===")
    print(f"  m={m}  真 ε(k)={eps_pop:.4f}  |  100 次上界被违反 {viol} 次（期望 ≈ {100*args.delta:.0f}）")

    # E3/E4: a/b 缓存回放 → T3/T4
    cache_path = os.path.join(PROJECT_ROOT, ".llm_decision_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        a, b, nm, nn = estimate_ab_from_cache(cache, gt_pos, cand, left_ids, right_ids)
        print(f"\n=== E3 a/b 估计（缓存回放，候选集 k={args.k2}/{args.k3} 上） ===")
        print(f"  验证器 a=P(yes|match)={a:.4f}（{nm} 匹配）  b=P(yes|non-match)={b:.4f}（{nn} 非匹配）")

        print(f"\n=== E4 T3 上界 / T4 最优 k（oracle ε(k) + 缓存 b，budget={args.budget}） ===")
        print(f"  {'k':>4}{'ε(k)':>9}{'T3上界':>9}{'实际loss':>9}")
        eps_by_k = {}
        auto = mutual_nn_agree(dense, sparse)
        for k in ks:
            _m, eps = oracle_row_coverage(dense, sparse, e_l, e_r, k, k)
            eps_by_k[k] = float(eps)
            bound = t3_bound(eps_by_k[k], a, b, k)
            _rc, _cc, c_k = build_candidates(dense, sparse, k, k)
            pr = candidate_priority_order(c_k, auto, fused)
            r = hybrid_router_candidates(pr, auto, args.budget,
                                         lambda ij: [bool(is_gt[i * nR + j]) for (i, j) in ij],
                                         nL, nR)
            rec = eval_positions(r["matched"], is_gt, nR, n_gt)["recall"]
            print(f"  {k:>4}{eps:>9.4f}{bound:>9.4f}{1-rec:>9.4f}")
        kstar = optimal_k(eps_by_k, b, ks)
        print(f"  T4: k* = {kstar}（argmin ε(k)+2b(k−1)，b={b:.4f}）")

    # E5: 同预算对比（v10.9 分位 vs v10.10 候选集）
    print(f"\n=== E5 同预算对比（oracle，budget={args.budget}） ===")
    ath = np.percentile(fused_flat, 100 - 0.02)
    rth = np.percentile(fused_flat, 80.0)
    r_old = hybrid_router(order, fused_flat, nL, nR, ath, rth, args.budget,
                          lambda poss: [bool(is_gt[p]) for p in poss])
    m_old = eval_positions(r_old["matched"], is_gt, nR, n_gt)
    auto = mutual_nn_agree(dense, sparse)
    cand = build_candidates(dense, sparse, args.k2, args.k3)[2]
    r_new = hybrid_router_candidates(candidate_priority_order(cand, auto, fused), auto,
                                     args.budget,
                                     lambda ij: [bool(is_gt[i * nR + j]) for (i, j) in ij],
                                     nL, nR)
    m_new = eval_positions(r_new["matched"], is_gt, nR, n_gt)
    print(f"  {'方法':<30}{'C1验证':>9}{'recall':>9}{'prec':>8}{'F1':>8}")
    print(f"  {'v10.9 分位(2%/80%)':<30}{r_old['llm_verified']:>9,}{m_old['recall']:>9.4f}"
          f"{m_old['precision']:>8.4f}{m_old['f1']:>8.4f}")
    print(f"  {'v10.10 候选集(k=10)+mutual':<30}{r_new['llm_verified']:>9,}{m_new['recall']:>9.4f}"
          f"{m_new['precision']:>8.4f}{m_new['f1']:>8.4f}")


if __name__ == "__main__":
    main()
