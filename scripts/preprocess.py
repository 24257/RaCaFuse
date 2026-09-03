#!/usr/bin/env python3
"""Dataset-specific preprocessing dispatcher for the public SEER data layout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {
    "mimic-iii": ROOT / "scripts" / "preprocessing" / "mimic_iii.py",
    "mimic-iv": ROOT / "scripts" / "preprocessing" / "mimic_iv_carmen.py",
    "eicu": ROOT / "scripts" / "preprocessing" / "eicu.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--raw-root", help="raw dataset root (MIMIC-III/eICU)")
    parser.add_argument("--source", help="Carmen data_final_4.pkl (MIMIC-IV)")
    parser.add_argument("--output-root", default="", help="defaults to data/<dataset>")
    parser.add_argument("--ddi-source", default="")
    parser.add_argument("--ddi-source-vocab", default="")
    parser.add_argument("--ehr-source", default="")
    parser.add_argument("--overwrite", action="store_true")
    args, extra = parser.parse_known_args()

    output = Path(args.output_root) if args.output_root else ROOT / "data" / args.dataset
    command = [sys.executable, str(BUILDERS[args.dataset])]
    if args.dataset == "mimic-iv":
        if not args.source:
            parser.error("--dataset mimic-iv requires --source <Carmen data_final_4.pkl>")
        command += ["--source", args.source, "--output_root", str(output), "--allow_single_visit_patients"]
        if args.ddi_source:
            command += ["--ddi_source", args.ddi_source]
        if args.ddi_source_vocab:
            command += ["--ddi_source_vocab", args.ddi_source_vocab]
        if args.ehr_source:
            command += ["--ehr_source", args.ehr_source]
        if args.overwrite:
            command += ["--overwrite"]
    elif args.dataset == "mimic-iii":
        if not args.raw_root:
            parser.error("--dataset mimic-iii requires --raw-root")
        command += ["--raw_root", args.raw_root, "--output_root", str(output)]
        if args.overwrite:
            command += ["--overwrite"]
    else:
        if not args.raw_root:
            parser.error("--dataset eicu requires --raw-root")
        command += ["--raw_root", args.raw_root, "--output_root", str(output)]
        if args.overwrite:
            command += ["--overwrite"]
    command += extra
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
