import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import taiji

from examples.company_demo.adapter import FusedJoinAdapter
from examples.company_demo.evaluate import evaluate
from examples.company_demo.tables import read_table, table_reference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "company_n100_seed20260908"


async def test_fusedjoin_mcp_uses_environment_defaults(tmp_path, monkeypatch):
    # Start the multi-table data service plus the operator without VERIFY_*
    # or per-operator cache overrides.
    # Empty inputs exercise startup, routing and result export without model requests.
    for key in ("VERIFY_MODEL", "VERIFY_API_KEY", "VERIFY_BASE_URL", "TAIJI_CACHE_DIR", "TAIJI_JOIN_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    paths = {}
    for side in ("left", "right"):
        # Each source directory has only its own table, with an arbitrary filename.
        source = tmp_path / (side + "_source")
        source.mkdir()
        paths[side] = source / "companies.csv"
        paths[side].write_text(f"{side}_id,{side}_name,{side}_description\n")
    async with taiji.Client({"model": "test", "api_key": "secret"}) as client:
        await client.add_mcp("company", {"command": sys.executable, "cwd": str(ROOT),
            "args": ["-m", "examples.company_mcp"]})
        assert (await client.call_tool("company", "list_tables", {})).result["tables"] == []
        for side in ("left", "right"):
            registration = await client.call_tool("company", "register_table", {
                "name": "company_" + side, "path": str(paths[side]),
            })
            assert not registration.is_error
        listing = (await client.call_tool("company", "list_tables", {})).result
        assert {t["name"] for t in listing["tables"]} == {"company_left", "company_right"}
        duplicate = await client.call_tool("company", "register_table", {
            "name": "company_left", "path": str(paths["right"]),
        })
        assert duplicate.is_error
        missing = await client.call_tool("company", "register_table", {
            "name": "missing", "path": str(tmp_path / "missing.csv"),
        })
        assert missing.is_error
        assert (await client.call_tool("company", "get_table", {"name": "missing"})).is_error
        assert (await client.call_tool("company", "list_tables", {})).result == listing
        await client.add_mcp("fusedjoin", {"command": sys.executable, "cwd": str(ROOT),
            "args": ["-m", "examples.fusedjoin_mcp"], "env": {
                "OPENAI_MODEL": "verification-fallback", "OPENAI_API_KEY": "test-secret",
                "OPENAI_BASE_URL": "https://model.invalid/v1", "TAIJI_OUTPUT_DIR": str(tmp_path / "out"),
            }})
        refs = {}
        for side in ("left", "right"):
            response = (await client.call_tool("company", "get_table", {"name": "company_" + side})).result
            assert response["name"] == "company_" + side
            assert "side" not in response
            assert "left" not in response and "right" not in response
            refs[side] = response["table"]
            assert Path(refs[side]["path"]) == paths[side]
            assert refs[side]["fields"]["id"] == side + "_id"
            preview = (await client.call_tool("company", "preview_table", {"name": "company_" + side})).result
            assert preview["table"] == refs[side] and preview["rows"] == []
        assert refs["left"]["path"] != refs["right"]["path"]
        result = (await client.call_tool("fusedjoin", "join_tables", refs)).result
    assert result["status"] == "success" and result["matches"] == []
    assert result["stats"]["api_requests"] == 0
    assert result["configuration"]["verification_model"] == "verification-fallback"
    assert result["configuration"]["cache_enabled"] is True
    assert Path(result["matches_csv"]).exists()


@pytest.mark.parametrize("prefix", ["", "left_", "right_", "catalog_"])
def test_table_fields_inferred_from_headers(tmp_path, prefix):
    path = tmp_path / "arbitrary_name.csv"
    path.write_text(f"{prefix}id,{prefix}name,{prefix}description\n001,Alpha,Company alpha\n")
    ref = table_reference(path)
    assert ref["fields"] == {key: prefix + key for key in ("id", "name", "description")}
    assert read_table(ref) == [{"id": "001", "name": "Alpha", "description": "Company alpha"}]


def test_ambiguous_headers_need_explicit_fields(tmp_path):
    path = tmp_path / "left.csv"
    path.write_text("left_id,left_name,left_description,right_id,right_name,right_description\n"
                    "001,Alpha,Company alpha,002,Beta,Company beta\n")
    with pytest.raises(ValueError, match="uniquely infer"):
        table_reference(path)
    fields = {key: "right_" + key for key in ("id", "name", "description")}
    assert read_table(table_reference(path, fields)) == [{"id": "002", "name": "Beta", "description": "Company beta"}]


async def test_registration_has_no_side_and_accepts_neutral_or_custom_schema(tmp_path):
    prefixed = tmp_path / "source.csv"
    prefixed.write_text("right_id,right_name,right_description\n001,Alpha,Company alpha\n")
    custom = tmp_path / "other.csv"
    custom.write_text("uid,title,text\n002,Beta,Company beta\n")
    async with taiji.Client({"model": "test", "api_key": "secret"}) as client:
        await client.add_mcp("company", {"command": sys.executable, "cwd": str(ROOT),
            "args": ["-m", "examples.company_mcp"]})
        schema = next(t["input_schema"] for t in client.list_tools() if t["name"] == "register_table")
        assert set(schema["required"]) == {"name", "path"}
        assert "side" not in schema["properties"]
        # The registration name does not control the source column prefix.
        registered = await client.call_tool("company", "register_table", {
            "name": "catalog_a", "path": str(prefixed),
        })
        assert not registered.is_error and "side" not in registered.result
        assert registered.result["table"]["fields"]["id"] == "right_id"
        assert registered.result["table"] == (await client.call_tool("company", "get_table", {"name": "catalog_a"})).result["table"]
        fields = {"id": "uid", "name": "title", "description": "text"}
        mapped = await client.call_tool("company", "register_table", {
            "name": "catalog_b", "path": str(custom), "fields": fields,
        })
        assert not mapped.is_error and mapped.result["table"]["fields"] == fields
        assert all("side" not in entry for entry in (await client.call_tool("company", "list_tables", {})).result["tables"])
        rows = (await client.call_tool("company", "preview_table", {"name": "catalog_b"})).result["rows"]
        assert rows == [{"id": "002", "name": "Beta", "description": "Company beta"}]


def test_table_ids_and_changed_reference(tmp_path):
    path = tmp_path / "left.csv"
    path.write_text('left_id,left_name,left_description\n001,Name,\n002,,Description\n')
    ref = table_reference(path)
    assert read_table(ref)[0] == {"id": "001", "name": "Name", "description": ""}
    path.write_text('left_id,left_name,left_description\n001,Changed,\n002,,Description\n')
    with pytest.raises(ValueError, match="content changed"):
        read_table(ref)
    path.write_text('left_id,left_name,left_description\n001,Name,\n001,Again,\n')
    with pytest.raises(ValueError, match="unique"):
        table_reference(path)


def test_heldout_excludes_calibration_negatives_and_positives(tmp_path):
    # Supply all 80 benchmark positives plus all 200 calibration pairs.
    from examples.company_demo.evaluate import load_pairs
    positives = load_pairs(DATA / "ground_truth.csv")
    calibration = load_pairs(DATA / "train.csv")
    path = tmp_path / "matches.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left_id", "right_id"])
        writer.writerows(positives | calibration)
    metrics = evaluate(path, DATA)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1
    assert metrics["evaluation_universe"] == 9800
    assert metrics["excluded_calibration_pairs"] == 200
    assert metrics["heldout_positive_pairs"] == 60


async def test_company_prompt_cache_and_unresolved_failures(tmp_path):
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model="test", api_key="secret",
                               base_url="https://model.invalid/v1", cache_dir=tmp_path,
                               retries=1, workers=2)
    _, cc = adapter._load()
    rows = [{"id": "001", "name": "NAME SHOULD NOT APPEAR", "description": "FULL COMPANY DESCRIPTION"},
            {"id": "002", "name": "", "description": "Other"}]
    calls = 0
    def respond(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        assert "FULL COMPANY DESCRIPTION" in prompt and "NAME SHOULD NOT APPEAR" not in prompt
        answer = "Yes" if "Other" not in prompt else "cannot determine"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})
    stats = dict(cache_hits=0, api_requests=0, teacher_api_requests=0, neutral_api_requests=0,
                 request_errors=0, teacher_failed_pairs=0, neutral_failed_pairs=0)
    cache = {}
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        decisions = await adapter._decide([(0, 0), (0, 1)], rows, rows, "teacher", cc, http, cache, stats)
        assert decisions == {(0, 0): True}  # A malformed answer never becomes False.
        assert stats["teacher_failed_pairs"] == 1 and stats["api_requests"] == 3
        await adapter._decide([(0, 0)], rows, rows, "teacher", cc, http, cache, stats)
        assert stats["cache_hits"] == 1 and calls == 3
        await adapter._decide([(0, 0)], rows, rows, "neutral", cc, http, cache, stats)
        assert calls == 4  # Teacher and neutral decisions have distinct namespaces.
    assert "secret" not in (tmp_path / "company-decisions-v1.json").read_text()


async def test_real_pipeline_primitives_without_external_models(tmp_path, monkeypatch):
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model="test", api_key="secret",
                               base_url="https://model.invalid/v1", output_dir=tmp_path / "out", audit_rows=0)
    # Exercise the real fusion, candidate and auto-accept primitives with a controlled embedding.
    class Embedding:
        def encode(self, texts, **kwargs):
            return np.eye(len(texts))
    adapter._embedding = Embedding()
    rj, _ = adapter._load()
    def forbid_labels(*args, **kwargs):
        raise AssertionError("Research data/label loader must never run in the operator")
    monkeypatch.setattr(rj.v10, "load_data", forbid_labels)
    for side in ("left", "right"):
        (tmp_path / (side + ".csv")).write_text(
            f"{side}_id,{side}_name,{side}_description\n001,Alpha,Alpha company\n002,Beta,Beta company\n")
    async def decisions(pairs, left, right, phase, cc, http, cache, stats):
        return {p: False for p in pairs}
    monkeypatch.setattr(adapter, "_decide", decisions)
    result = await adapter.join_tables(*(table_reference(tmp_path / (s + ".csv")) for s in ("left", "right")))
    assert result["status"] == "success" and len(result["matches"]) == 2
    assert result["stats"]["total_pairs"] == result["stats"]["candidates"] == 4
    assert {m["left_id"] for m in result["matches"]} == {"001", "002"}
    assert all(m["source"] == "auto" for m in result["matches"])
    assert Path(result["matches_csv"]).exists()


async def test_failed_calibration_disables_rejection_and_keeps_pairs_unresolved(tmp_path, monkeypatch):
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model="test", api_key="secret",
                               base_url="https://model.invalid/v1", output_dir=tmp_path / "out", audit_rows=2)
    rj, _ = adapter._load()
    monkeypatch.setattr(adapter, "_scores", lambda *args: (np.array([[.1, .2], [.3, .4]]), np.eye(2), np.eye(2)))
    monkeypatch.setattr(rj.v10, "mutual_nn_agree", lambda *args: set())
    for side in ("left", "right"):
        (tmp_path / (side + ".csv")).write_text(
            f"{side}_id,{side}_name,{side}_description\n001,Alpha,Alpha company\n002,Beta,Beta company\n")
    async def decisions(pairs, left, right, phase, cc, http, cache, stats):
        if phase == "teacher":
            stats["teacher_failed_pairs"] += 1
            return {p: True for p in pairs[1:]}
        stats["neutral_failed_pairs"] += 1
        return {p: False for p in pairs[1:]}
    monkeypatch.setattr(adapter, "_decide", decisions)
    result = await adapter.join_tables(*(table_reference(tmp_path / (s + ".csv")) for s in ("left", "right")))
    assert result["status"] == "partial"
    assert result["stats"]["verification_pairs"] == 4 and result["stats"]["threshold_rejected"] == 0
    assert len(result["unresolved"]) == 1 and result["matches"] == []


async def test_authentication_failure_stops_subsequent_requests(tmp_path):
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model="test", api_key="secret",
                               base_url="https://model.invalid/v1", retries=2, workers=1)
    _, cc = adapter._load()
    rows = [{"id": str(i), "name": "Company", "description": "Company description"} for i in range(5)]
    stats = dict(cache_hits=0, api_requests=0, teacher_api_requests=0, neutral_api_requests=0,
                 request_errors=0, teacher_failed_pairs=0, neutral_failed_pairs=0)
    cache = {}
    requests = 0
    def reject(request):
        nonlocal requests
        requests += 1
        return httpx.Response(401, text="Sensitive provider response")
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as http:
        pairs = [(i, j) for i in range(5) for j in range(5)]
        teacher = await adapter._decide(pairs, rows, rows, "teacher", cc, http, cache, stats)
        neutral = await adapter._decide(pairs, rows, rows, "neutral", cc, http, cache, stats)
    assert requests == 1 and stats["api_requests"] == 1
    assert stats["fatal_model_error"] == "HTTP 401"
    assert stats["teacher_failed_pairs"] == stats["neutral_failed_pairs"] == 25
    assert not teacher and not neutral and not cache


@pytest.mark.parametrize("count", [0, 1])
async def test_empty_and_singleton_tables(tmp_path, monkeypatch, count):
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model="test", api_key="secret",
                               base_url="https://model.invalid/v1", output_dir=tmp_path / "out", audit_rows=0)
    if count:
        monkeypatch.setattr(adapter, "_scores", lambda *args: (np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))))
    for side in ("left", "right"):
        text = f"{side}_id,{side}_name,{side}_description\n"
        if count:
            text += "001,Alpha,Company alpha\n"
        (tmp_path / (side + ".csv")).write_text(text)
    result = await adapter.join_tables(*(table_reference(tmp_path / (s + ".csv")) for s in ("left", "right")))
    assert result["status"] == "success" and len(result["matches"]) == count
    assert result["stats"]["candidates"] == count and result["stats"]["api_requests"] == 0
