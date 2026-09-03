#!/usr/bin/env python3
"""Inspect baseline provenance or run the bundled SEER auxiliary branch training path."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "baseline_registry.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registered comparison methods")
    parser.add_argument("--method", default="", help="method key from the registry")
    parser.add_argument("--dataset", default="mimic-iii", choices=["mimic-iii", "mimic-iv", "eicu"])
    parser.add_argument("--seed", type=int, default=1203, choices=[1203, 1204, 1205])
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if args.list or not args.method:
        for key, item in registry.items():
            print(f"{key}: {item['name']} [{item['availability']}]")
        return 0
    if args.method not in registry:
        parser.error(f"unknown method: {args.method}")
    item = registry[args.method]
    if args.method != "seer_auxiliary":
        print(f"{item['name']} is not vendored in this release.")
        print(f"Source: {item['source']}")
        print("Use the source implementation with the same processed data/split/metric protocol documented in this repository.")
        return 0
    command = [sys.executable, str(ROOT / "scripts" / "train_seer_auxiliary.py"), "--dataset", args.dataset, "--seed", str(args.seed), "--cuda", str(args.cuda)]
    if args.dry_run:
        command.append("--dry-run")
    print(" ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
