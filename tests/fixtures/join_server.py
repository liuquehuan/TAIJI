"""Deterministic replacement operator for contract testing, not fusedjoin results."""
import csv
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP("replacement-join")


@server.tool()
def get_table() -> dict:
    """Deliberately shares a name with the data server to test namespaced routing."""
    return {"operator": "replacement"}


@server.tool()
def join_tables(left: dict, right: dict) -> dict:
    """Join same-name records using original IDs and the standard table-reference contract."""
    def read(ref):
        with Path(ref["path"]).open(newline="") as handle:
            return list(csv.DictReader(handle))
    lrows, rrows = read(left), read(right)
    matches = [{"left_id": l[left["fields"]["id"]], "right_id": r[right["fields"]["id"]]}
               for l in lrows for r in rrows
               if l[left["fields"]["name"]] == r[right["fields"]["name"]]]
    return {"status": "success", "matches": matches, "stats": {"matches": len(matches)}}


@server.tool()
def fail() -> dict:
    """Produce a tool error for propagation tests."""
    raise ValueError("Intentional test failure")


if __name__ == "__main__":
    try:
        server.run()
    finally:
        if os.getenv("CLOSE_MARKER"):
            Path(os.environ["CLOSE_MARKER"]).touch()
