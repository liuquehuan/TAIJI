"""Compare cold fusedjoin matching with exhaustive batched LLM matching."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

import taiji
from examples.company_demo.adapter import FusedJoinAdapter
from examples.company_demo.evaluate import evaluate
from examples.company_demo.llm_baseline import run_llm_baseline
from examples.company_demo.tables import table_reference

ROOT = Path(__file__).resolve().parents[1]


def comparison_row(method: str, result: dict, metrics: dict, latency: float) -> dict:
    stats = result["stats"]
    usage = stats["token_usage"]
    return {"method": method, "status": result["status"],
            **{key: metrics[key] for key in ("precision", "recall", "f1", "true_positives", "false_positives", "false_negatives")},
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"], "usage_complete": usage["complete"],
            "provider_cached_prompt_tokens": usage["provider_cached_prompt_tokens"],
            "latency_seconds": latency, "api_requests": stats["api_requests"],
            "application_cache_hits": stats["cache_hits"], "matches": len(result["matches"]),
            "unresolved_pairs": stats["unresolved_pairs"], "result_json": result["result_json"],
            "matches_csv": result["matches_csv"], "requests_json": result["requests_json"]}


def save_comparison(run_dir: Path, protocol: dict, rows: list[dict]) -> None:
    data = {"protocol": protocol, "results": rows,
            "complete": len(rows) == len(protocol["methods"]),
            "accuracy_runs_successful": all(r["status"] == "success" for r in rows),
            "token_usage_complete": all(r["usage_complete"] for r in rows)}
    (run_dir / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Company：fusedjoin 与纯 LLM 分块匹配实测", "",
             f"模型：`{protocol['model']}`；并发：{protocol['workers']}；分块大小：{protocol['block_size']} × {protocol['block_size']}。",
             "", "| 方法 | 状态 | Precision | Recall | F1 | 输入 token | 输出 token | 总 token | 匹配耗时 / 秒 | API 请求 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        tokens = [str(row[k]) if row[k] is not None else "usage 缺失" for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        lines.append(f"| {row['method']} | {row['status']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} | "
                     + " | ".join(tokens) + f" | {row['latency_seconds']:.3f} | {row['api_requests']} |")
    if rows:
        cached = "；".join(f"{r['method']}={r['provider_cached_prompt_tokens'] if r['provider_cached_prompt_tokens'] is not None else '接口未提供'}"
                           for r in rows)
        lines += ["", "本次服务端 cached_tokens：" + cached + "。这些 token 包含在输入/总 token 中；服务端缓存可能影响延迟。"]
    by_method = {r["method"]: r for r in rows}
    if "fusedjoin" in by_method and "llm_batch" in by_method:
        fused, baseline = by_method["fusedjoin"], by_method["llm_batch"]
        if fused["total_tokens"] is not None and baseline["total_tokens"]:
            reduction = 1 - fused["total_tokens"] / baseline["total_tokens"]
            lines += ["", f"fusedjoin 总 token 相对纯 LLM {'减少' if reduction >= 0 else '增加'} {abs(reduction):.2%}；"
                      f"两者匹配耗时比（fusedjoin / LLM）为 {fused['latency_seconds'] / baseline['latency_seconds']:.2f}。"
                      "准确率取舍见上表，结果适用于当前逐对校准实现与该分块基线的方案比较。"]
        stats = json.loads(Path(fused["result_json"]).read_text())["stats"]
        teacher = stats["token_usage"]["by_phase"].get("teacher", {})
        lines += ["", f"fusedjoin 阶段分解：打分/embedding {stats['timings_seconds'].get('scoring', 0):.3f} 秒；"
                  f"teacher {stats['teacher_api_requests']} 次请求、{teacher.get('total_tokens')} token、"
                  f"{stats['timings_seconds'].get('teacher', 0):.3f} 秒；neutral {stats['neutral_api_requests']} 次请求、"
                  f"{stats['timings_seconds'].get('neutral', 0):.3f} 秒。"]
    lines += ["", "计量范围：匹配函数的完整墙钟耗时，包括读表、计算、API 请求与结果导出。共同的 Python 模块导入、表引用创建、自然语言规划、MCP 启动和离线评估不计入；fusedjoin 的 embedding 加载/编码、teacher 校准和 neutral 验证计入。模型权重已下载，不计首次权重下载。",
              "", "两边均禁用应用层判断缓存，使用相同模型、temperature=0、并发和重试次数。服务提供方可能自动缓存输入前缀，接口返回的 cached_tokens 单独记录；总 token 不扣除这些 token，也不把 token 总量等同于账单金额。",
              "", "纯 LLM 将左右表分别分块，遍历所有块组合，覆盖完整笛卡尔积；没有 embedding、候选筛选或自动接受。两边采用 company-v7 的相同公司匹配判据及完整描述。fusedjoin 逐对输出 Yes/No，纯 LLM 每块输出匹配 ID 列表，输入组织和输出格式存在差异。允许一对多/多对多。",
              "", "评估在匹配结束后独立读取标签：同时剔除全部 200 个正负校准对，在剩余 9,800 对上评估 60 个标注正例。未标注配对按负例计属于该 benchmark 的约定。失败请求/未完成块单独标为未决；partial 运行不视为完整准确率比较。",
              "", "Token 为每个实际 HTTP 请求返回的 usage 之和，包含重试与无法解析的回答；有请求未返回有效 usage 时，总量标记为缺失，已知小计保留在逐方法 result.json。历史 fusedjoin 结果没有 usage，未将估算值当作历史实测。",
              "", "字段语义参考 [OpenAI Chat Completions 官方文档](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)；本实验实际计数来自配置的第三方接口响应。",
              "", "本次为单次实测，API 延迟、服务端缓存和模型输出可能随运行变化。完整配置、输入哈希、逐请求 token/延迟、逐方法匹配 CSV 和评估文件均保留在本目录。",
              "", "逐方法产物：", ""]
    lines += [f"- {r['method']}：`{r['result_json']}`；请求：`{r['requests_json']}`。" for r in rows]
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> Path | None:
    model = taiji.ModelConfig(
        model=os.getenv("VERIFY_MODEL") or os.environ["OPENAI_MODEL"],
        api_key=os.getenv("VERIFY_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("VERIFY_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1", timeout=args.request_timeout,
    )
    refs = {side: table_reference(args.data_dir / (side + ".csv")) for side in ("left", "right")}
    methods = ["fusedjoin", "llm_batch"] if args.method == "both" else [args.method]
    pairs = refs["left"]["row_count"] * refs["right"]["row_count"]
    nblocks = ((refs["left"]["row_count"] + args.block_size - 1) // args.block_size
               * ((refs["right"]["row_count"] + args.block_size - 1) // args.block_size))
    print(f"Methods: {methods}; full domain: {pairs} pairs; LLM blocks: {nblocks}; workers: {args.workers}", flush=True)
    if args.dry_run:
        return None
    run_dir = args.output_dir.resolve() / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    adapter = FusedJoinAdapter(source_dir=ROOT / "fusedjoin", model=model.model, api_key=model.api_key,
        base_url=model.base_url, embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_cache=os.getenv("EMBEDDING_CACHE", str(ROOT / ".model-cache")), cache_dir=None,
        output_dir=run_dir / "fusedjoin", workers=args.workers, retries=args.retries,
        request_timeout=args.request_timeout)
    # Prepare the shared original company predicate outside both matching timers.
    _, cc = adapter._load()
    predicate = cc.build_prompt("company", "", "", "", "", None, None).split("\nLeft:", 1)[0]
    protocol = {"model": model.model, "endpoint": model.endpoint, "methods": methods,
        "workers": args.workers, "block_size": args.block_size, "baseline_blocks": nblocks,
        "total_pairs": pairs, "retries": args.retries, "request_timeout": args.request_timeout,
        "temperature": 0, "application_decision_cache": "disabled for both methods",
        "scope": "matching stage; shared setup, planner, MCP setup and evaluation excluded",
        "fusedjoin_parameters": {"k": 10, "weight": .5, "audit_rows": 20, "epsilon": .05, "seed": 0},
        "inputs": refs, "predicate": predicate, "predicate_sha256": hashlib.sha256(predicate.encode()).hexdigest(),
        "platform": platform.platform(), "python": platform.python_version(),
        "libraries": {p: importlib.metadata.version(p) for p in ("httpx", "numpy", "pandas", "sentence-transformers")}}
    rows = []
    save_comparison(run_dir, protocol, rows)
    print(f"Artifacts: {run_dir}", flush=True)
    for method in methods:
        started = time.perf_counter()
        if method == "fusedjoin":
            result = await adapter.join_tables(refs["left"], refs["right"])
        else:
            result = await run_llm_baseline(refs["left"], refs["right"], model=model, predicate=predicate,
                output_dir=run_dir / "llm_batch", block_size=args.block_size, workers=args.workers, retries=args.retries)
        latency = time.perf_counter() - started
        metrics = evaluate(Path(result["matches_csv"]), args.data_dir)
        Path(result["result_json"]).with_name("evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        row = comparison_row(method, result, metrics, latency)
        rows.append(row)
        save_comparison(run_dir, protocol, rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if result["stats"].get("fatal_model_error"):
            raise RuntimeError(f"Model configuration error: {result['stats']['fatal_model_error']}; partial artifacts: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "company_n100_seed20260908")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/company_benchmark")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--method", choices=("both", "fusedjoin", "llm_batch"), default="both")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.block_size < 1 or args.workers < 1 or args.retries < 0:
        parser.error("block-size and workers must be positive; retries must be nonnegative")
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO)
    for name in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    path = asyncio.run(run(args))
    if path:
        print(f"Report: {path / 'REPORT.md'}")


if __name__ == "__main__":
    main()
