#!/usr/bin/env python3
"""Train the public SEER longitudinal branch with the shared 3-dataset protocol."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from protocol import ROOT, stage_a_command, stage_b_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mimic-iii", "mimic-iv", "eicu"])
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-root", default="artifacts/retrained")
    parser.add_argument("--stage", choices=["all", "a", "b"], default="all")
    parser.add_argument("--device", default="", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.dataset
    output_root = ROOT / args.output_root / args.dataset / f"seed{args.seed}"
    stage_a = output_root / "stage_a.pth"
    stage_b = output_root / "stage_b.pth"
    paper_cache = output_root / "cache" / "paper_m3"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    if args.stage in {"all", "a"}:
        commands.append(stage_a_command(args.dataset, args.seed, data_root, stage_a))
    if args.stage in {"all", "b"}:
        if args.stage == "b" and not stage_a.is_file():
            raise FileNotFoundError(f"Stage B requires an existing Stage A checkpoint: {stage_a}")
        commands.append(stage_b_command(args.dataset, args.seed, data_root, stage_a, stage_b, paper_cache))

    env = None
    if args.device:
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.device
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    print(f"Longitudinal outputs: {stage_a} and {stage_b}")
    print("This command retrains only the longitudinal branch. Final SEER evaluation also uses the SEER auxiliary branch checkpoint seer_auxiliary_branch.model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
