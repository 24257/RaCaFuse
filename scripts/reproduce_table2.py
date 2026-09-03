#!/usr/bin/env python3
"""Render the release per-seed stability table; filename is repository convenience, not manuscript numbering."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {"mimic-iii": "MIMIC-III", "mimic-iv": "MIMIC-IV", "eicu": "eICU"}


def render() -> str:
    with (ROOT / "results" / "test_results_per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = ["| Dataset | Seed | Jaccard | F1 | PRAUC | Count MAE |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {DISPLAY[row['dataset']]} | {row['seed']} | {float(row['jaccard']):.6f} | {float(row['f1']):.6f} | "
            f"{float(row['prauc']):.6f} | {float(row['count_mae']):.6f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    table = render()
    print(table)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
