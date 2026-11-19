"""Deployment adapter over the unchanged fusedjoin research source.

Uses run_fusedjoin.fused_candidates, v10.fused_score/mutual_nn_agree,
M.tau_reject and cert_common.build_prompt('company'). It does not call the
research loader (which also loads ground truth), or the old lenient/strict gate.
The latter has a different prompt and maps failed requests to False. This adapter
retains bounded retries and decision caching, but preserves failures as unresolved.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import math
import re
import sys
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import httpx

from .tables import read_table
from .measurement import UsageMeter

LOG = logging.getLogger(__name__)


class FusedJoinAdapter:
    def __init__(self, *, source_dir: Path, model: str, api_key: str, base_url: str,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_cache: str | None = None, output_dir: Path = Path("outputs/company"),
                 cache_dir: Path | None = None, k: int = 10, weight: float = 0.5,
                 audit_rows: int = 20, epsilon: float = 0.05, seed: int = 0,
                 workers: int = 4, retries: int = 2, request_timeout: float = 90) -> None:
        if k < 1 or audit_rows < 0 or not 0 <= weight <= 1 or not 0 < epsilon < 1:
            raise ValueError("Invalid candidate, weight, audit or epsilon configuration")
        if workers < 1 or retries < 0 or request_timeout <= 0:
            raise ValueError("Invalid request configuration")
        if not model or not api_key or not base_url:
            raise ValueError("Verification model, API key and base URL are required")
        self.source_dir = source_dir.resolve()
        self.model, self._key = model, api_key
        base = base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.embedding_model, self.model_cache = embedding_model, model_cache
        self.output_dir = output_dir.resolve()
        self.cache_dir = cache_dir.resolve() if cache_dir else None
        self.k, self.weight, self.audit_rows = k, weight, audit_rows
        self.epsilon, self.seed = epsilon, seed
        self.workers, self.retries, self.request_timeout = workers, retries, request_timeout
        self._embedding: Any = None
        self._modules: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._modules is None:
            if not (self.source_dir / "run_fusedjoin.py").is_file():
                raise ValueError("source_dir must contain the original fusedjoin sources")
            sys.path.insert(0, str(self.source_dir))
            with redirect_stdout(sys.stderr):
                self._modules = (importlib.import_module("run_fusedjoin"),
                                 importlib.import_module("cert_common"))
        return self._modules

    def _scores(self, left: Any, right: Any, v10: Any) -> Any:
        with redirect_stdout(sys.stderr):
            if self._embedding is None:
                from sentence_transformers import SentenceTransformer
                # Bypass the research loader, which unconditionally forces offline mode.
                self._embedding = SentenceTransformer(self.embedding_model, cache_folder=self.model_cache)
            return v10.fused_score(left, right, self._embedding, w=self.weight)

    def _save_cache(self, cache: dict) -> None:
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / "company-decisions-v1.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(path)

    async def _decide(self, pairs: list, left: list, right: list, phase: str,
                      cc: Any, http: httpx.AsyncClient, cache: dict, stats: dict) -> dict:
        decisions = {}
        sem = asyncio.Semaphore(self.workers)
        sp = cc.DS_SPECS["company"]

        async def one(pair: tuple[int, int]) -> None:
            i, j = pair
            prompt = cc.build_prompt("company", left[i]["name"], left[i]["description"],
                                     right[j]["name"], right[j]["description"], sp["cap_l"], sp["cap_r"])
            # Phase/model/endpoint/prompt/content/IDs are part of the namespace. No key is persisted.
            payload = ["company-v7-adapter-v1", phase, self.model, self.endpoint,
                       left[i]["id"], right[j]["id"], prompt]
            key = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
            if cache.get(key) in ("yes", "no"):
                decisions[pair] = cache[key] == "yes"
                stats["cache_hits"] += 1
                return
            async with sem:
                if stats.get("fatal_model_error"):
                    stats[phase + "_failed_pairs"] += 1
                    return
                for attempt in range(self.retries + 1):
                    stats["api_requests"] += 1
                    stats[phase + "_api_requests"] += 1
                    try:
                        response = await http.post(self.endpoint,
                            headers={"Authorization": "Bearer " + self._key},
                            extensions={"taiji_measurement": {"phase": phase, "left_id": left[i]["id"],
                                        "right_id": right[j]["id"], "attempt": attempt + 1}},
                            json={"model": self.model, "temperature": 0, "max_tokens": 20,
                                  "messages": [{"role": "user", "content": prompt}]})
                        response.raise_for_status()
                        answer = response.json()["choices"][0]["message"]["content"].strip()
                        parsed = re.fullmatch(r"\s*(yes|no)[.!]?\s*", answer, flags=re.I)
                        if not parsed:
                            raise ValueError("Expected an unambiguous Yes/No response")
                        value = parsed.group(1).lower()
                        decisions[pair] = value == "yes"
                        cache[key] = value
                        self._save_cache(cache)
                        return
                    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                        stats["request_errors"] += 1
                        # Log error class only: a provider error body may contain credentials.
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404):
                            if not stats.get("fatal_model_error"):
                                LOG.warning("Model configuration/authentication failed (HTTP %s); stopping further requests",
                                            exc.response.status_code)
                            stats["fatal_model_error"] = f"HTTP {exc.response.status_code}"
                            break
                        LOG.warning("%s request failed: %s", phase, type(exc).__name__)
                        if attempt < self.retries:
                            await asyncio.sleep(min(2 ** attempt, 4))
                stats[phase + "_failed_pairs"] += 1

        # Bound the number of pending tasks as well as in-flight requests.
        for start in range(0, len(pairs), self.workers * 4):
            await asyncio.gather(*(one(pair) for pair in pairs[start:start + self.workers * 4]))
            if not stats.get("fatal_model_error"):
                LOG.info("%s decisions: %d/%d", phase, min(start + self.workers * 4, len(pairs)), len(pairs))
        return decisions

    async def join_tables(self, left: dict, right: dict) -> dict:
        async with self._lock:
            return await self._join(left, right)

    async def _join(self, left_ref: dict, right_ref: dict) -> dict:
        import numpy as np
        import pandas as pd

        started = time.monotonic()
        left_rows, right_rows = read_table(left_ref), read_table(right_ref)
        run_dir = self.output_dir / uuid.uuid4().hex
        run_dir.mkdir(parents=True)
        meter = UsageMeter(run_dir / "requests.jsonl")
        stats = dict(total_pairs=len(left_rows) * len(right_rows), candidates=0, auto_accepted=0,
                     threshold_rejected=0, verification_pairs=0, teacher_pairs=0,
                     api_requests=0, teacher_api_requests=0, neutral_api_requests=0,
                     cache_hits=0, request_errors=0, teacher_failed_pairs=0, neutral_failed_pairs=0)
        matches, unresolved = [], []
        timings = {}
        tau = None
        if left_rows and right_rows:
            rj, cc = self._load()
            ldf = pd.DataFrame([{f"left_{key}": val for key, val in row.items()} for row in left_rows])
            rdf = pd.DataFrame([{f"right_{key}": val for key, val in row.items()} for row in right_rows])
            t = time.monotonic()
            fused, dense, sparse = await asyncio.to_thread(self._scores, ldf, rdf, rj.v10)
            timings["scoring"] = time.monotonic() - t
            _, _, candidates = rj.fused_candidates(fused, self.k)
            # The source caps top-k at n-1, yielding no candidates on a singleton axis.
            # Restore the complete axis when k covers it, including 1xN and Nx1 tables.
            if self.k >= len(right_rows) or self.k >= len(left_rows):
                candidates.update((i, j) for i in range(len(left_rows)) for j in range(len(right_rows)))
            auto = set(rj.v10.mutual_nn_agree(dense, sparse))
            candidates |= auto
            stats.update(candidates=len(candidates), auto_accepted=len(auto))
            audit = set(np.random.RandomState(self.seed).permutation(len(left_rows))[:self.audit_rows].tolist())
            teacher_pairs = sorted(p for p in candidates if p[0] in audit)
            stats["teacher_pairs"] = len(teacher_pairs)
            cache: dict = {}
            if self.cache_dir and (self.cache_dir / "company-decisions-v1.json").exists():
                cache = json.loads((self.cache_dir / "company-decisions-v1.json").read_text())
                if not isinstance(cache, dict):
                    raise ValueError("Decision cache must contain an object")
            async with httpx.AsyncClient(timeout=self.request_timeout, event_hooks=meter.hooks) as http:
                t = time.monotonic()
                teacher = await self._decide(teacher_pairs, left_rows, right_rows, "teacher", cc, http, cache, stats)
                timings["teacher"] = time.monotonic() - t
                # Incomplete calibration disables free rejection; missing teacher calls are not negatives.
                if teacher and not stats["teacher_failed_pairs"]:
                    pairs = sorted(teacher)
                    threshold = rj.M.tau_reject(np.array([fused[i, j] for i, j in pairs]),
                                              np.array([teacher[p] for p in pairs]),
                                              np.array([i for i, _ in pairs]), self.epsilon)
                    tau = float(threshold) if math.isfinite(threshold) else None
                reject = {p for p in candidates - auto if tau is not None and fused[p] <= tau}
                disputed = sorted(candidates - auto - reject)
                stats.update(threshold_rejected=len(reject), verification_pairs=len(disputed))
                t = time.monotonic()
                neutral = await self._decide(disputed, left_rows, right_rows, "neutral", cc, http, cache, stats)
                timings["neutral"] = time.monotonic() - t
            accepted = auto | {p for p, yes in neutral.items() if yes}
            for i, j in sorted(accepted):
                matches.append({"left_id": left_rows[i]["id"], "right_id": right_rows[j]["id"],
                                "score": float(fused[i, j]),
                                "source": "auto" if (i, j) in auto else "llm_verification"})
            unresolved = [{"left_id": left_rows[i]["id"], "right_id": right_rows[j]["id"]}
                          for i, j in disputed if (i, j) not in neutral]
        stats["failed_pairs"] = stats["teacher_failed_pairs"] + stats["neutral_failed_pairs"]
        stats["matches"] = len(matches)
        stats["unresolved_pairs"] = len(unresolved)
        stats["token_usage"] = meter.summary()
        timings["total"] = time.monotonic() - started
        stats["timings_seconds"] = timings
        status = "partial" if stats["failed_pairs"] else "success"
        meter.save(run_dir / "requests.json")
        import csv
        with (run_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["left_id", "right_id", "score", "source"])
            writer.writeheader()
            writer.writerows(matches)
        summary = (
            f"Full search space (total_pairs): {stats['total_pairs']}; "
            f"selected candidates (candidates): {stats['candidates']}; "
            f"automatic accepts: {stats['auto_accepted']}; threshold rejections: {stats['threshold_rejected']}; "
            f"automated LLM verification pairs: {stats['verification_pairs']}; "
            f"matches: {len(matches)}; unresolved pairs: {len(unresolved)}."
        )
        result = {"status": status, "summary": summary,
                  "matches": matches, "stats": stats, "unresolved": unresolved,
                  "inputs": {"left": left_ref, "right": right_ref},
                  "matches_csv": str(run_dir / "matches.csv"), "result_json": str(run_dir / "result.json"),
                  "requests_json": str(run_dir / "requests.json"),
                  "configuration": {"algorithm": "fusedjoin", "k": self.k, "weight": self.weight,
                      "audit_rows": self.audit_rows, "epsilon": self.epsilon, "seed": self.seed,
                      "verification_model": self.model, "embedding_model": self.embedding_model,
                      "prompt": "cert_common.build_prompt(company), company-v7", "reject_threshold": tau,
                      "cache_enabled": self.cache_dir is not None, "cardinality": "set-based"}}
        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
