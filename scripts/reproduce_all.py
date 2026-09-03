#!/usr/bin/env python3
"""Check the public fixture or explicitly supplied authorized local predictions."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mimic-iii", "mimic-iv", "eicu")
SEEDS = (1203, 1204, 1205)


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-inference", action="store_true", help="rerun all 9 final inferences; requires protected processed datasets and suitable GPU/runtime")
    parser.add_argument("--predictions-root", type=Path, help="authorized local prediction directory; required for real-data replay")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    if args.with_inference and args.predictions_root is None:
        parser.error("--with-inference requires --predictions-root for private outputs")
    if args.with_inference:
        for dataset in DATASETS:
            for seed in SEEDS:
                run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--dataset", dataset, "--seed", str(seed), "--cuda", str(args.cuda), "--prediction-output", str(args.predictions_root.resolve() / dataset / f"seed{seed}.npz")])
    if args.predictions_root is not None:
        run([sys.executable, str(ROOT / "scripts" / "reproduce_results.py"), "--predictions-root", str(args.predictions_root.resolve())])
    else:
        run([sys.executable, str(ROOT / "scripts" / "verify_public.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
