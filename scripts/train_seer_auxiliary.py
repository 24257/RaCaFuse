#!/usr/bin/env python3
"""Retrain the SEER auxiliary branch used by final SEER."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from protocol import ddi_mask_path
ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "baselines" / "seer_auxiliary_adapter.py"
BUNDLED_RUNTIME = ROOT / "src" / "seer_auxiliary_branch"

def build_command(dataset: str, seed: int, data_root: Path, output_dir: Path, cuda: int) -> list[str]:
    cmd = [sys.executable, str(ADAPTER), "--dataset", dataset, "--datadir", str(data_root), "--records_file", "records_final.pkl", "--voc_file", "voc_final.pkl", "--ddi_file", "ddi_A_final.pkl", "--seed", str(seed), "--cuda", str(cuda), "--epoch", "40", "--early_stop", "30", "--lr", "0.0005", "--coef", "2.5", "--target_ddi", "0.06", "--dim", "64", "--dp", "0.7", "--target_start_visit", "1", "--output_dir", str(output_dir), "--run_id", f"SEER_seer_auxiliary_{dataset}_seed{seed}"]
    mask = ddi_mask_path(dataset, seed)
    if mask.is_file(): cmd += ["--ddi_mask_path", str(mask)]
    if dataset in {"mimic-iii", "mimic-iv"}:
        cmd += ["--paper_protocol"]
    else:
        cmd += ["--split_path", str(ROOT / "configs" / "splits" / "seer_auxiliary_training" / f"eicu_seed{seed}.json")]
    return cmd

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["mimic-iii", "mimic-iv", "eicu"])
    p.add_argument("--seed", type=int, default=1203, choices=[1203,1204,1205])
    p.add_argument("--data-root", default="")
    p.add_argument("--output-root", default="artifacts/retrained_seer_auxiliary")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not (BUNDLED_RUNTIME / "modules" / "__init__.py").is_file(): raise FileNotFoundError(BUNDLED_RUNTIME)
    data_root = Path(a.data_root) if a.data_root else ROOT / "data" / a.dataset
    output = ROOT / a.output_root / a.dataset / f"seed{a.seed}"
    cmd = build_command(a.dataset, a.seed, data_root, output, a.cuda)
    print(" ".join(cmd))
    if not a.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy(); env.pop("SEER_AUXILIARY_SRC", None)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
