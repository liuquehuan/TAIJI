"""Local CSV reference contract shared by the two demo plugins, not the SDK."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


def read_table(ref: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(ref["path"])
    if not path.is_absolute() or path.suffix.lower() != ".csv":
        raise ValueError("Table path must be an absolute CSV path shared by both servers")
    fields = ref["fields"]
    if set(fields) != {"id", "name", "description"} or len(set(fields.values())) != 3:
        raise ValueError("fields must map distinct id, name and description columns")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(fields.values()).issubset(reader.fieldnames or []):
            raise ValueError("Referenced columns are missing")
        rows = [{key: row.get(column) or "" for key, column in fields.items()} for row in reader]
    ids = [row["id"] for row in rows]
    if any(not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Table IDs must be nonempty and unique; IDs are preserved as strings")
    if ref.get("row_count", len(rows)) != len(rows):
        raise ValueError("Table changed since its reference was obtained")
    if ref.get("sha256") and ref["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Table content changed since discovery")
    return rows


def table_reference(path: Path, fields: dict[str, str] | None = None) -> dict[str, Any]:
    if fields is None:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle), [])
        prefixes = ["" if col == "id" else col[:-2] for col in columns
                    if col == "id" or col.endswith("_id")]
        candidates = [{key: prefix + key for key in ("id", "name", "description")}
                      for prefix in prefixes
                      if all(prefix + key in columns for key in ("id", "name", "description"))]
        if len(candidates) != 1:
            raise ValueError("Cannot uniquely infer id/name/description columns; supply an explicit fields mapping")
        fields = candidates[0]
    ref = {"path": str(path.resolve()), "format": "csv", "fields": dict(fields)}
    rows = read_table(ref)
    ref.update(row_count=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return ref
