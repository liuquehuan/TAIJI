"""Advanced validation: python -m examples.company_validation --compare --evaluate."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import taiji

ROOT = Path(__file__).resolve().parents[1]


def configurations(output_dir: Path, *, audit_rows: int = 20,
                   cache: bool = True) -> tuple[taiji.ModelConfig, dict, dict]:
    model = taiji.ModelConfig.from_env()
    data_config = {"command": sys.executable, "cwd": str(ROOT),
                   "args": ["-m", "examples.company_mcp"]}
    join_config = {"command": sys.executable, "cwd": str(ROOT),
                   "args": ["-m", "examples.fusedjoin_mcp"], "timeout": 3600,
                   "env": {
                       "VERIFY_MODEL": os.getenv("VERIFY_MODEL", model.model),
                       "VERIFY_API_KEY": os.getenv("VERIFY_API_KEY", model.api_key),
                       "VERIFY_BASE_URL": os.getenv("VERIFY_BASE_URL", model.base_url),
                       "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                       "EMBEDDING_CACHE": os.getenv("EMBEDDING_CACHE", str(ROOT / ".model-cache")),
                       "TAIJI_OUTPUT_DIR": str(output_dir.resolve()),
                       "TAIJI_CACHE_DIR": str(ROOT / ".taiji-cache") if cache else "",
                       "TAIJI_JOIN_CONFIG": json.dumps({"audit_rows": audit_rows}),
                   }}
    return model, data_config, join_config


async def run(args: argparse.Namespace) -> None:
    model, data_config, join_config = configurations(
        args.output_dir, audit_rows=args.audit_rows, cache=not args.no_cache)
    async with taiji.Client(model=model) as client:
        await client.add_mcp("company", data_config)
        for side in ("left", "right"):
            registration = await client.call_tool("company", "register_table", {
                "name": "company_" + side, "path": str((args.data_dir / (side + ".csv")).resolve()),
            })
            if registration.is_error:
                raise RuntimeError(f"Company {side} table registration failed")
        await client.add_mcp("fusedjoin", join_config)
        direct = None
        if args.direct or args.compare:
            left = (await client.call_tool("company", "get_table", {"name": "company_left"})).result["table"]
            right = (await client.call_tool("company", "get_table", {"name": "company_right"})).result["table"]
            direct_call = await client.call_tool("fusedjoin", "join_tables", {"left": left, "right": right})
            if direct_call.is_error:
                raise RuntimeError("Direct join failed; inspect the MCP server stderr")
            direct = direct_call.result
            if direct["stats"].get("fatal_model_error"):
                raise RuntimeError(
                    f"Verification model configuration/authentication failed ({direct['stats']['fatal_model_error']}). "
                    f"Update the local model configuration. Partial run saved to {direct['result_json']}"
                )
        if args.direct:
            result = taiji.QueryResult(answer="Direct tool execution", artifacts=[{"server": "fusedjoin", "tool": "join_tables", "data": direct}], status=direct["status"])
        else:
            result = await client.query(args.query)
        joins = [a["data"] for a in result.artifacts if a["tool"] == "join_tables"
                 and isinstance(a["data"], dict) and "matches_csv" in a["data"]]
        if not joins:
            raise RuntimeError("Query did not produce a join result; inspect model/tool configuration")
        joined = joins[-1]
        record = result.to_dict()
        if args.compare and direct is not None:
            pairs = lambda data: {(m["left_id"], m["right_id"]) for m in data["matches"]}
            equivalent = (pairs(direct) == pairs(joined)
                          and direct["unresolved"] == joined["unresolved"]
                          and direct["inputs"] == joined["inputs"]
                          and direct["configuration"] == joined["configuration"])
            record["direct_comparison"] = {"equivalent": equivalent, "direct_result_json": direct["result_json"],
                                           "direct_stats": direct["stats"]}
            if not equivalent:
                raise RuntimeError("Direct and natural-language tool executions differ")
        # Evaluation happens after querying, outside the model and operator processes.
        if args.evaluate:
            from examples.company_demo.evaluate import evaluate
            record["evaluation"] = evaluate(Path(joined["matches_csv"]), args.data_dir)
        path = Path(joined["result_json"]).with_name("query.json")
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(result.answer)
        print(f"Status: {result.status}; matches: {joined['matches_csv']}; trace: {path}")
        if "evaluation" in record:
            print(json.dumps(record["evaluation"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "company_n100_seed20260908")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/company")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--query", default="使用 fusedjoin 匹配 company 数据源中已注册的 company_left 和 company_right 两张公司表，找出两张表中属于同一家现实公司的记录。回答仅包含匹配数量、未决数量、匹配结果文件路径和完整结果文件路径四项。")
    parser.add_argument("--audit-rows", type=int, default=20)
    parser.add_argument("--direct", action="store_true", help="Call tools without the planner model")
    parser.add_argument("--compare", action="store_true", help="Compare direct and natural-language execution")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate separately after execution")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    # Explicit demo convenience only; the SDK never implicitly reads credential files.
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
