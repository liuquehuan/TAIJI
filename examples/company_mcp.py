"""Company table registry: python -m examples.company_mcp."""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .company_demo.tables import read_table, table_reference


def main() -> None:
    tables: dict[str, dict] = {}
    server = FastMCP("company-data")

    @server.tool()
    def register_table(name: str, path: str, fields: dict[str, str] | None = None) -> dict:
        """Register one named company CSV table during Python setup.

        path must be an absolute CSV path. Fields are inferred from id/name/description
        columns, optionally sharing a prefix. For other or ambiguous schemas, supply
        fields mapping id/name/description to exact column names. Registration does
        not assign a table a left/right role. Names must be nonempty and unique. Query existing
        registrations with list_tables/get_table; register new files only when
        explicitly asked to do so. No other files or evaluation labels are loaded.
        """
        if not name.strip() or name in tables:
            raise ValueError("Table name must be nonempty and unique")
        if not Path(path).is_absolute():
            raise ValueError("Table path must be absolute")
        ref = table_reference(Path(path), fields)
        tables[name] = {"name": name, "table": ref}
        return {"dataset": "company", **tables[name]}

    @server.tool()
    def list_tables() -> dict:
        """List registered company table names, row counts and fields."""
        return {"dataset": "company", "tables": [
            {"name": name, "row_count": entry["table"]["row_count"],
             "fields": entry["table"]["fields"]}
            for name, entry in tables.items()
        ]}

    @server.tool()
    def get_table(name: str) -> dict:
        """Get one registered company table reference by name.

        Use list_tables to discover names if needed. The response includes 'table'
        with its absolute CSV path, fields, row count and hash. Get each named input
        separately and pass its 'table' to either left/right argument of join_tables
        on the operator service, choosing roles for that invocation. This service can provide both inputs
        for matching two independently registered tables.
        """
        if name not in tables:
            raise ValueError(f"Unknown registered table: {name}")
        return {"dataset": "company", **tables[name]}

    @server.tool()
    def preview_table(name: str, limit: int = 3) -> dict:
        """Preview up to five records from a registered company table by name."""
        if not 1 <= limit <= 5:
            raise ValueError("limit must be between 1 and 5")
        entry = get_table(name)
        return {**entry, "rows": read_table(entry["table"])[:limit]}

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
