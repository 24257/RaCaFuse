#!/usr/bin/env python3
"""Render the release main-result table from frozen mean±sample-SD metadata."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("mimic-iii", "mimic-iv", "eicu")


def render() -> str:
    payload = json.loads((ROOT / "configs" / "paper_final_test_targets.json").read_text(encoding="utf-8"))
    lines = ["| Dataset | Jaccard | F1 | PRAUC |", "|---|---:|---:|---:|"]
    for dataset in ORDER:
        row = payload["datasets"][dataset]
        lines.append(f"| {row['display_name']} | {row['jaccard']['paper']} | {row['f1']['paper']} | {row['prauc']['paper']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="", help="optional UTF-8 Markdown output path")
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
