"""Independent fusedjoin stdio MCP service, configured only through its own env."""
import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .company_demo.adapter import FusedJoinAdapter


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    for name in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(os.getenv("TAIJI_JOIN_CONFIG", "{}"))
    cache_dir = os.getenv("TAIJI_CACHE_DIR", str(repo / ".taiji-cache"))
    adapter = FusedJoinAdapter(
        source_dir=Path(config.pop("source_dir", repo / "fusedjoin")),
        model=os.getenv("VERIFY_MODEL") or os.environ["OPENAI_MODEL"],
        api_key=os.getenv("VERIFY_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("VERIFY_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1",
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_cache=os.getenv("EMBEDDING_CACHE", str(repo / ".model-cache")),
        output_dir=Path(os.getenv("TAIJI_OUTPUT_DIR", repo / "outputs/company")),
        cache_dir=Path(cache_dir) if cache_dir else None,
        **config,
    )
    server = FastMCP("fusedjoin")

    @server.tool()
    async def join_tables(left: dict, right: dict) -> dict:
        """Find records that refer to the same real-world company (set-based matching).

        Obtain references for the two named input tables from the data-source MCP
        tools first, keeping the left and right inputs distinct. Each
        reference has an absolute CSV 'path', 'fields' mapping id/name/description to
        column names, 'row_count' and optional sha256. Pass the complete references.
        Returns original left_id/right_id matches, score (not probability), source,
        statistics, unresolved pairs, status, matches_csv and result_json paths.
        stats.total_pairs is the full Cartesian search space; stats.candidates is
        the smaller number of selected candidates. They are distinct counts.
        verification_pairs counts automated LLM checks, NOT pending human review;
        unresolved_pairs counts decisions that could not be completed.
        Model and algorithm parameters are configured by the operator, not tool arguments.
        """
        return await adapter.join_tables(left, right)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
