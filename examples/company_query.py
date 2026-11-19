import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

import taiji

ROOT = Path(__file__).resolve().parents[1]


async def main() -> taiji.QueryResult:
    load_dotenv(ROOT / ".env")
    model = taiji.ModelConfig.from_env()

    data_dir = ROOT / "company_n100_seed20260908"
    stdio_config = {"command": sys.executable, "cwd": str(ROOT)}
    data_config = {**stdio_config, "args": ["-m", "examples.company_mcp"]}
    join_config = {**stdio_config, "args": ["-m", "examples.fusedjoin_mcp"], "timeout": 3600}

    async with taiji.Client(model=model) as client:
        await client.add_mcp("company", data_config)
        await client.call_tool("company", "register_table", {
            "name": "company_left", "path": str(data_dir / "left.csv"),
        })
        await client.call_tool("company", "register_table", {
            "name": "company_right", "path": str(data_dir / "right.csv"),
        })
        await client.add_mcp("fusedjoin", join_config)
        result = await client.query(
            "请找出 company 左右表中属于同一家现实公司的记录"
            "并报告匹配数量、未决数量和结果文件路径。"
        )

    print(result.answer)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from examples.company_validation import main as validate
        validate()
    else:
        asyncio.run(main())
