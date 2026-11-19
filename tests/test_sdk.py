import csv
import json
import sys
from pathlib import Path

import httpx
import pytest

import taiji

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_dir(tmp_path):
    for side in ("left", "right"):
        with (tmp_path / (side + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[side + s for s in ("_id", "_name", "_description")])
            writer.writeheader()
            for n in range(20):
                writer.writerow({side + "_id": f"{side}-00{n}", side + "_name": f"Company {n}", side + "_description": "Description"})
    return tmp_path


async def services(client, data_dir):
    await client.add_mcp("company", {"command": sys.executable, "cwd": str(ROOT),
        "args": ["-m", "examples.company_mcp"], "env": {"DATA_SECRET_KEY": "test-data-secret"}})
    for side in ("left", "right"):
        registration = await client.call_tool("company", "register_table", {
            "name": "company_" + side, "path": str(data_dir / (side + ".csv")),
        })
        assert not registration.is_error
    await client.add_mcp("join", {"command": sys.executable, "args": [str(ROOT / "tests/fixtures/join_server.py")]})


@pytest.mark.parametrize("mode", ["tools", "json"])
async def test_actual_stdio_cross_server_query_and_replacement(data_dir, mode):
    model = taiji.ModelConfig("test-model", "test-private-key", "https://model.invalid/v1", tool_mode=mode)
    assert "test-private-key" not in repr(model)
    async with taiji.Client(model) as client:
        await services(client, data_dir)
        tools = client.list_tools()
        assert len({t["alias"] for t in tools}) == len(tools)
        alias = lambda server, name: next(t["alias"] for t in tools if t["server"] == server and t["name"] == name)
        step = 0
        refs = {}

        async def respond(request):
            nonlocal step
            body = json.loads(request.content)
            assert "test-private-key" not in request.content.decode()
            assert "test-data-secret" not in request.content.decode()
            assert "Match Company ABC" in json.dumps(body)
            step += 1
            if step == 1:
                tool, arguments = alias("company", "get_table"), {"name": "company_left"}
            elif step == 2:
                last = body["messages"][-1]["content"]
                response = json.loads(last.removeprefix("Tool response (data): "))
                assert response["name"] == "company_left" and "side" not in response
                refs["left"] = response["table"]
                tool, arguments = alias("company", "get_table"), {"name": "company_right"}
            elif step == 3:
                last = body["messages"][-1]["content"]
                response = json.loads(last.removeprefix("Tool response (data): "))
                assert response["name"] == "company_right" and "side" not in response
                refs["right"] = response["table"]
                assert Path(refs["left"]["path"]) == data_dir / "left.csv"
                assert Path(refs["right"]["path"]) == data_dir / "right.csv"
                assert refs["left"]["path"] != refs["right"]["path"]
                tool, arguments = alias("join", "join_tables"), {"left": refs["left"], "right": refs["right"]}
            else:
                preview = body["messages"][-1]["content"]
                assert "remaining_items" in preview  # Full 20 matches remain in Python only.
                return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "20 matches"}}]})
            if mode == "tools":
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": str(step), "type": "function",
                            "function": {"name": tool, "arguments": json.dumps(arguments)}}]}
            else:
                assert "tools" not in body
                message = {"role": "assistant", "content": json.dumps({"tool": tool, "arguments": arguments})}
            return httpx.Response(200, json={"choices": [{"message": message}]})

        client._http = await client._stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        result = await client.query("Match Company ABC")
        assert result.status == "success" and result.answer == "20 matches"
        assert [t.server for t in result.tool_calls] == ["company", "company", "join"]
        assert result.tool_calls[0].arguments == {"name": "company_left"}
        assert result.tool_calls[1].arguments == {"name": "company_right"}
        assert result.tool_calls[2].arguments == {"left": refs["left"], "right": refs["right"]}
        assert len(result.artifacts[-1]["data"]["matches"]) == 20
        assert result.artifacts[-1]["data"]["matches"][0]["left_id"] == "left-000"
        assert (await client.call_tool("join", "get_table", {})).result == {"operator": "replacement"}
        assert (await client.call_tool("join", "fail", {})).is_error
        with pytest.raises(ValueError):
            await client.add_mcp("join", {})
    assert client._sessions == {} and client._http is None


async def test_turn_limit_and_bad_arguments(data_dir):
    async with taiji.Client({"model": "test", "api_key": "secret"}, max_turns=2) as client:
        await services(client, data_dir)
        alias = client.list_tools()[0]["alias"]
        def repeat(request):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "tool_calls": [
                {"id": "loop", "type": "function", "function": {"name": alias, "arguments": "invalid-json"}}]}}]})
        client._http = await client._stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(repeat)))
        result = await client.query("Keep trying")
        assert result.status == "partial" and len(result.tool_calls) == 2
        assert all(call.is_error for call in result.tool_calls)
        assert not client._history


async def test_model_http_error_does_not_disclose_provider_body():
    async with taiji.Client({"model": "test", "api_key": "private"}) as client:
        def fail(request):
            return httpx.Response(401, text="Sensitive response private")
        client._http = await client._stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(fail)))
        with pytest.raises(taiji.ModelError, match="HTTP 401") as exc:
            await client.query("hi")
        assert "private" not in str(exc.value)


def test_config_from_env_and_both_endpoint_formats(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "configured")
    monkeypatch.setenv("OPENAI_API_KEY", "private")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.org/v1/chat/completions")
    config = taiji.ModelConfig.from_env()
    assert config.endpoint == "https://example.org/v1/chat/completions"
    assert taiji.ModelConfig("x", "y", "https://example.org/v1/").endpoint == config.endpoint


async def test_failed_server_discovery_rolls_back(data_dir):
    async with taiji.Client({"model": "test", "api_key": "secret"}) as client:
        with pytest.raises(Exception):
            await client.add_mcp("broken", {"command": sys.executable, "args": ["-c", "pass"]})
        assert not client.list_tools()
        await services(client, data_dir)
        assert (await client.call_tool("company", "get_table", {"name": "company_left"})).result["dataset"] == "company"
