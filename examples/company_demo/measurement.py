"""Provider-reported usage and HTTP timing; never log prompts or credentials."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import httpx

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class UsageMeter:
    def __init__(self, journal: Path | None = None) -> None:
        self.records: list[dict] = []
        self.journal = journal

    @property
    def hooks(self) -> dict:
        return {"request": [self.on_request], "response": [self.on_response]}

    async def on_request(self, request: httpx.Request) -> None:
        meta = request.extensions.get("taiji_measurement", {})
        record = {k: v for k, v in meta.items() if k in (
            "phase", "left_id", "right_id", "block_index", "left_ids", "right_ids", "attempt",
        )}
        record.update(index=len(self.records), status_code=None, latency_seconds=None, usage=None)
        request.extensions["taiji_meter_index"] = len(self.records)
        request.extensions["taiji_meter_started"] = time.perf_counter()
        self.records.append(record)

    async def on_response(self, response: httpx.Response) -> None:
        # Hooks run before AsyncClient.post returns, so explicitly consume the body.
        await response.aread()
        request = response.request
        record = self.records[request.extensions["taiji_meter_index"]]
        record.update(status_code=response.status_code,
                      latency_seconds=time.perf_counter() - request.extensions["taiji_meter_started"])
        try:
            data = response.json()
            if isinstance(data, dict):
                record["usage"] = data.get("usage")
                record["response_model"] = data.get("model")
                record["response_id"] = data.get("id")
        except ValueError:
            pass
        if self.journal is not None:
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _aggregate(records: list[dict]) -> dict:
        valid = [r for r in records if isinstance(r["usage"], dict)
                 and all(type(r["usage"].get(k)) is int and r["usage"][k] >= 0 for k in TOKEN_FIELDS)]
        known = {key: sum(r["usage"][key] for r in valid) for key in TOKEN_FIELDS}
        complete = len(valid) == len(records)
        cached = [r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
                  for r in valid if isinstance(r["usage"].get("prompt_tokens_details"), dict)]
        cached = [v for v in cached if type(v) is int and v >= 0]
        return {**{k: v if complete else None for k, v in known.items()},
                "reported_subtotal": known, "complete": complete,
                "requests_with_usage": len(valid), "requests_without_usage": len(records) - len(valid),
                "provider_cached_prompt_tokens": sum(cached) if len(cached) == len(records) else None}

    def summary(self) -> dict:
        phases = sorted({r.get("phase", "unknown") for r in self.records})
        durations = sorted(r["latency_seconds"] for r in self.records if r["latency_seconds"] is not None)
        return {"source": "provider_reported", **self._aggregate(self.records),
                "api_requests": len(self.records),
                "observed_response_models": sorted({r["response_model"] for r in self.records if r.get("response_model")}),
                "by_phase": {p: self._aggregate([r for r in self.records if r.get("phase", "unknown") == p]) for p in phases},
                "request_latency_seconds": {
                    "mean": statistics.mean(durations) if durations else None,
                    "median": statistics.median(durations) if durations else None,
                    "p95": durations[min(len(durations) - 1, int((len(durations) - 1) * .95))] if durations else None,
                }}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
