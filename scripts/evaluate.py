#!/usr/bin/env python3
"""Evaluate released SEER checkpoints with the SEER auxiliary branch."""

from __future__ import annotations

import sys as _seer_sys
from pathlib import Path as _SeerPath
_seer_sys.path.insert(0, str(_SeerPath(__file__).resolve().parents[1] / "src"))
import seer_security

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from protocol import ROOT, PRIVATE_ARTIFACT_ROOT, ehr_cache_path, ddi_mask_path, final_split_path, longitudinal_export_command

sys.path.insert(0, str(ROOT / "src"))
from final_fusion import AffineCountCalibrator, fuse_components, fuse_with_calibrators, load_npz  # noqa: E402

BUNDLED_AUXILIARY_RUNTIME = ROOT / "src" / "seer_auxiliary_branch"


def run(command: list[str], env: dict[str, str] | None, dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def seer_auxiliary_command(dataset: str, seed: int, data_root: Path, checkpoint: Path, split: str, output_dir: Path, cuda: int) -> list[str]:
    command = [
        sys.executable, str(ROOT / "baselines" / "seer_auxiliary_adapter.py"), "--Test", "--dataset", dataset, "--datadir", str(data_root),
        "--records_file", "records_final.pkl", "--voc_file", "voc_final.pkl", "--ddi_file", "ddi_A_final.pkl",
        "--target_start_visit", "1", "--seed", str(seed), "--cuda", str(cuda), "--dim", "64", "--dp", "0.7",
        "--lr", "0.0005", "--coef", "2.5", "--target_ddi", "0.06", "--output_dir", str(output_dir),
        "--run_id", f"SEER_{dataset}_seed{seed}_{split}", "--resume_path", str(checkpoint), "--eval_split", split, "--save_predictions",
    ]
    command.append("--paper_protocol")
    if dataset == "eicu":
        if split == "train":
            manifest = ROOT / "configs" / "splits" / "seer_auxiliary_training" / f"eicu_seed{seed}.json"
        else:
            manifest = final_split_path(dataset, seed)
        if manifest is None or not manifest.is_file():
            raise RuntimeError(f"eICU AUXILIARY split manifest is missing: {manifest}")
        command.extend(["--split_path", str(manifest)])
    mask = ddi_mask_path(dataset, seed)
    if mask.is_file():
        command.extend(["--ddi_mask_path", str(mask)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mimic-iii", "mimic-iv", "eicu"])
    parser.add_argument("--seed", type=int, required=True, choices=[1203, 1204, 1205])
    parser.add_argument("--split", choices=["eval", "test"], default="test")
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--calibration-data-root",
        default="",
        help="optional data root used for both branches' train-only calibration exports; defaults to --data-root",
    )
    parser.add_argument("--seer_auxiliary-src", default="", help="optional debugging override for bundled AUXILIARY runtime")
    parser.add_argument("--cuda", type=int, default=0, help="visible CUDA device index for both branches; -1 selects CPU")
    parser.add_argument("--work-root", default="artifacts/evaluation")
    parser.add_argument("--prediction-output", default="")
    parser.add_argument("--stage-a", default="", help="optional Stage A checkpoint override")
    parser.add_argument("--stage-b", default="", help="optional Stage B checkpoint override")
    parser.add_argument("--auxiliary-checkpoint", default="", help="optional AUXILIARY-derived expert checkpoint override")
    parser.add_argument(
        "--refit-calibration",
        action="store_true",
        help="fit count calibrators from this checkpoint set's train-only outputs instead of using released frozen calibrators",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (BUNDLED_AUXILIARY_RUNTIME / "modules" / "__init__.py").is_file():
        raise FileNotFoundError(BUNDLED_AUXILIARY_RUNTIME)
    override = None
    if args.seer_auxiliary_src:
        override = Path(args.seer_auxiliary_src).expanduser().resolve()
        if not (override / "modules" / "__init__.py").is_file():
            parser.error(f"invalid --seer_auxiliary-src: {override}")

    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.dataset
    calibration_data_root = Path(args.calibration_data_root) if args.calibration_data_root else data_root
    checkpoint_root = PRIVATE_ARTIFACT_ROOT / "checkpoints" / args.dataset / f"seed{args.seed}"
    stage_a = Path(args.stage_a).expanduser() if args.stage_a else checkpoint_root / "stage_a.pth"
    stage_b = Path(args.stage_b).expanduser() if args.stage_b else checkpoint_root / "stage_b.pth"
    auxiliary_checkpoint = Path(args.auxiliary_checkpoint).expanduser() if args.auxiliary_checkpoint else checkpoint_root / "seer_auxiliary_branch.model"
    for path in (stage_a, stage_b, auxiliary_checkpoint):
        if not args.dry_run and not path.is_file():
            raise FileNotFoundError(path)
    if args.dataset == "eicu":
        manifest = final_split_path(args.dataset, args.seed)
        if manifest is None or not manifest.is_file():
            raise FileNotFoundError(manifest)
    if not args.dry_run:
        if not ddi_mask_path(args.dataset, args.seed).is_file() or not any(ehr_cache_path(args.dataset, args.seed).glob("*.pkl")):
            raise FileNotFoundError("Frozen reference inputs are not public. Set SEER_PRIVATE_ARTIFACT_ROOT to your authorized local artifact tree; see data/README.md.")
        seer_security.require_trusted_pickle()
        for name in ("records_final.pkl", "voc_final.pkl", "ddi_A_final.pkl", "ehr.pkl"):
            if not (data_root / name).is_file():
                raise FileNotFoundError(data_root / name)

    work = ROOT / args.work_root / args.dataset / f"seed{args.seed}" / args.split
    long_dir, auxiliary_dir = work / "longitudinal", work / "seer_auxiliary"
    cache = work / "paper_m3_cache"
    calibration_cache = work / "paper_m3_cache_calibration"
    long_target = long_dir / f"{args.split}.npz"
    if not args.dry_run:
        long_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if override is None:
        env.pop("SEER_AUXILIARY_SRC", None)
    else:
        env["SEER_AUXILIARY_SRC"] = str(override)

    run(longitudinal_export_command(args.dataset, args.seed, data_root, stage_a, stage_b, args.split, long_target, cache, cuda=args.cuda), env, args.dry_run)
    auxiliary_target = auxiliary_dir / args.split
    run(seer_auxiliary_command(args.dataset, args.seed, data_root, auxiliary_checkpoint, args.split, auxiliary_target, args.cuda), env, args.dry_run)

    long_train = long_dir / "train.npz"
    auxiliary_train = auxiliary_dir / "train"
    if args.refit_calibration:
        run(longitudinal_export_command(args.dataset, args.seed, calibration_data_root, stage_a, stage_b, "train", long_train, calibration_cache, cuda=args.cuda), env, args.dry_run)
        run(seer_auxiliary_command(args.dataset, args.seed, calibration_data_root, auxiliary_checkpoint, "train", auxiliary_train, args.cuda), env, args.dry_run)
    if args.dry_run:
        mode = "fresh train-only calibration refit" if args.refit_calibration else "released frozen calibration"
        print(f"DRY RUN: bundled auxiliary runtime, split policy, commands, and {mode} validated; external checkpoint/input existence was not checked and protected data were not opened.")
        return 0

    with (data_root / "ddi_A_final.pkl").open("rb") as handle:
        ddi = np.asarray(seer_security.load_pickle(handle))
    if args.refit_calibration:
        arrays, metrics = fuse_components(
            load_npz(long_train),
            load_npz(auxiliary_train / "predictions.npz"),
            load_npz(long_target),
            load_npz(auxiliary_target / "predictions.npz"),
            ddi,
        )
    else:
        calibration = json.loads((ROOT / "configs" / "count_calibration.json").read_text(encoding="utf-8"))
        calibration_row = calibration[args.dataset][str(args.seed)]
        long_cal = AffineCountCalibrator(**calibration_row["longitudinal"])
        auxiliary_cal = AffineCountCalibrator(**calibration_row["seer_auxiliary"])
        arrays, metrics = fuse_with_calibrators(
            load_npz(long_target), load_npz(auxiliary_target / "predictions.npz"), ddi, long_cal, auxiliary_cal,
        )
    if args.dataset == "eicu":
        metrics["ddi_rate"] = None
        metrics["ddi_reporting"] = "N/A: packaged eICU DDI matrix contains no nonzero interactions"
    output = Path(args.prediction_output) if args.prediction_output else work / "seer_prediction.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, "split": args.split, "prediction": str(output), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
