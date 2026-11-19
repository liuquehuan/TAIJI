import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from taiji import ModelConfig
from examples.company_demo.adapter import FusedJoinAdapter
from examples.company_demo import llm_baseline
from examples.company_demo.llm_baseline import make_blocks, parse_matches
from examples.company_demo.measurement import UsageMeter
from examples.company_demo.tables import table_reference

ROOT = Path(__file__).resolve().parents[1]


async def test_usage_includes_retries_and_missing_usage_is_unknown(tmp_path):
    meter = UsageMeter(tmp_path / "requests.jsonl")
    responses = [
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                   "prompt_tokens_details": {"cached_tokens": 4}}, "model": "observed-model"},
        {"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
        {},
    ]
    def handle(request):
        return httpx.Response(200 if responses else 503, json=responses.pop(0))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), event_hooks=meter.hooks) as http:
        for i in range(2):
            await http.post("https://model.invalid/v1", headers={"Authorization": "Bearer secret-key"},
                            json={"prompt": "private-description"}, extensions={"taiji_measurement": {"phase": "teacher", "attempt": i + 1}})
        assert meter.summary()["total_tokens"] == 18
        assert meter.summary()["provider_cached_prompt_tokens"] is None  # One response omitted this field.
        await http.post("https://model.invalid/v1", extensions={"taiji_measurement": {"phase": "neutral"}})
    summary = meter.summary()
    assert summary["api_requests"] == 3 and summary["requests_without_usage"] == 1
    assert summary["complete"] is False and summary["total_tokens"] is None
    assert summary["reported_subtotal"]["total_tokens"] == 18
    assert summary["by_phase"]["teacher"]["total_tokens"] == 18
    assert all(r["latency_seconds"] >= 0 for r in meter.records)
    text = (tmp_path / "requests.jsonl").read_text()
    assert "secret-key" not in text and "private-description" not in text


@pytest.mark.parametrize("nl,nr,size", [(100, 100, 10), (23, 17, 10), (0, 10, 3)])
def test_blocks_cover_entire_cartesian_product_once(nl, nr, size):
    left, right = list(range(nl)), list(range(nr))
    blocks = make_blocks(left, right, size)
    covered = [(li, ri) for ls, rs in blocks for li in ls for ri in rs]
    assert len(covered) == nl * nr
    assert set(covered) == {(li, ri) for li in left for ri in right}
    if nl == nr == 100:
        assert len(blocks) == 100


def write_tables(tmp_path):
    for side in ("left", "right"):
        (tmp_path / (side + ".csv")).write_text(
            f"{side}_id,{side}_name,{side}_description\n001,NAME_MUST_NOT_BE_SENT,Company alpha\n002,NAME_MUST_NOT_BE_SENT,Company beta\n")
    return [table_reference(tmp_path / (s + ".csv")) for s in ("left", "right")]


def mock_clients(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


async def test_batched_baseline_usage_complete_and_preserves_ids(tmp_path, monkeypatch):
    refs = write_tables(tmp_path)
    def respond(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        assert "NAME_MUST_NOT_BE_SENT" not in prompt
        block = json.loads(prompt.rsplit("\n", 1)[-1])
        matches = [{"left_id": "001", "right_id": "002"}] if block["left"][0]["id"] == "001" and block["right"][0]["id"] == "002" else []
        return httpx.Response(200, json={"model": "test", "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"matches": matches})}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})
    mock_clients(monkeypatch, respond)
    result = await llm_baseline.run_llm_baseline(*refs, model=ModelConfig("test", "secret", "https://model.invalid/v1"),
        predicate="Same real-world company", output_dir=tmp_path / "out", block_size=1, workers=2)
    assert result["status"] == "success" and result["stats"]["completed_blocks"] == 4
    assert [(m["left_id"], m["right_id"]) for m in result["matches"]] == [("001", "002")]
    assert result["stats"]["token_usage"]["total_tokens"] == 56
    assert result["stats"]["api_requests"] == 4 and result["stats"]["cache_hits"] == 0
    assert len(json.loads(Path(result["requests_json"]).read_text())) == 4


async def test_baseline_retry_tokens_count_even_for_invalid_answers(tmp_path, monkeypatch):
    refs = write_tables(tmp_path)
    calls = 0
    def respond(request):
        nonlocal calls
        calls += 1
        answer = "not JSON" if calls == 1 else '{"matches": []}'
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}})
    mock_clients(monkeypatch, respond)
    result = await llm_baseline.run_llm_baseline(*refs, model=ModelConfig("test", "secret", "https://model.invalid/v1"),
        predicate="Same company", output_dir=tmp_path / "out", block_size=10, workers=1, retries=1)
    assert result["status"] == "success" and calls == 2
    assert result["stats"]["token_usage"]["total_tokens"] == 210
    assert result["stats"]["request_errors"] == 1


async def test_failed_baseline_blocks_remain_unresolved(tmp_path, monkeypatch):
    refs = write_tables(tmp_path)
    def reject(request):
        return httpx.Response(401, json={"error": "unauthorized"})
    mock_clients(monkeypatch, reject)
    result = await llm_baseline.run_llm_baseline(*refs, model=ModelConfig("test", "secret", "https://model.invalid/v1"),
        predicate="Same company", output_dir=tmp_path / "out", block_size=1, workers=1)
    assert result["status"] == "partial" and result["matches"] == []
    assert result["stats"]["unresolved_pairs"] == 4 and result["stats"]["api_requests"] == 1
    assert result["stats"]["token_usage"]["total_tokens"] is None
    with pytest.raises(ValueError, match="outside"):
        parse_matches('{"matches": [{"left_id": "invented", "right_id": "001"}]}', [{"id": "001"}], [{"id": "001"}])


async def test_fusedjoin_exports_actual_usage_without_decision_cache(tmp_path, monkeypatch):
    refs = write_tables(tmp_path)
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model="test", api_key="secret",
        base_url="https://model.invalid/v1", output_dir=tmp_path / "out", cache_dir=None, audit_rows=0)
    class Embedding:
        def encode(self, texts, **kwargs):
            return np.eye(len(texts))
    adapter._embedding = Embedding()
    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "No"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})
    mock_clients(monkeypatch, respond)
    result = await adapter.join_tables(*refs)
    assert result["status"] == "success"
    assert result["stats"]["api_requests"] == 2 and result["stats"]["cache_hits"] == 0
    assert result["stats"]["token_usage"]["total_tokens"] == 22
    assert result["stats"]["token_usage"]["by_phase"]["neutral"]["total_tokens"] == 22
