"""Offline held-out evaluation. Never imported by either MCP server."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .tables import read_table, table_reference


def load_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"left_id", "right_id"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Pair columns missing in {path.name}")
        rows = list(reader)
    return {(r["left_id"], r["right_id"]) for r in rows}


def evaluate(matches_csv: Path, data_dir: Path) -> dict:
    ids = {}
    for side in ("left", "right"):
        ids[side] = {r["id"] for r in read_table(table_reference(data_dir / (side + ".csv")))}
    predicted = load_pairs(matches_csv)
    calibration = load_pairs(data_dir / "train.csv")  # Exclude negatives as well as positives.
    truth = load_pairs(data_dir / "test_ground_truth.csv")
    for pairs in (predicted, calibration, truth):
        if any(l not in ids["left"] or r not in ids["right"] for l, r in pairs):
            raise ValueError("Pair outside the query table domain")
    if truth & calibration:
        raise ValueError("Held-out positives overlap calibration pairs")
    heldout = predicted - calibration
    tp, fp, fn = len(heldout & truth), len(heldout - truth), len(truth - heldout)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "heldout_predictions": len(heldout), "heldout_positive_pairs": len(truth),
            "excluded_calibration_pairs": len(calibration),
            "evaluation_universe": len(ids["left"]) * len(ids["right"]) - len(calibration),
            "label_semantics": "Closed-world benchmark; unannotated held-out pairs count as negatives"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches_csv", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("company_n100_seed20260908"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(args.matches_csv, args.data_dir)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
