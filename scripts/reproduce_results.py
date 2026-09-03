#!/usr/bin/env python3
"""Recompute and regression-check all nine authorized local SEER reference predictions."""

from __future__ import annotations

import argparse
import math
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from final_fusion import load_npz, set_metrics  # noqa: E402

DATASETS = ("mimic-iii", "mimic-iv", "eicu")
SEEDS = (1203, 1204, 1205)
METRICS = ("jaccard", "f1", "prauc", "count_mae")
PAPER_PERCENT_TARGETS = {
    "mimic-iii": ("MIMIC-III", "57.73 ± 0.09", "72.48 ± 0.08", "81.77 ± 0.05"),
    "mimic-iv": ("MIMIC-IV", "53.20 ± 0.06", "68.41 ± 0.03", "77.97 ± 0.06"),
    "eicu": ("eICU", "54.52 ± 0.16", "68.89 ± 0.13", "82.58 ± 0.20"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recompute(path: Path) -> dict[str, float]:
    arrays = load_npz(path)
    required = {"y_true", "y_pred", "ranking_score", "predicted_count"}
    missing = required.difference(arrays)
    if missing:
        raise RuntimeError(f"missing arrays {sorted(missing)}: {path}")
    metrics = set_metrics(arrays["y_true"], arrays["y_pred"], arrays["ranking_score"])
    expected_count = np.asarray(arrays["predicted_count"], dtype=np.int64)
    observed_count = np.asarray(arrays["y_pred"]).sum(axis=1).astype(np.int64)
    if not np.array_equal(expected_count, observed_count):
        raise RuntimeError(f"predicted_count mismatch: {path}")
    return {metric: float(metrics[metric]) for metric in METRICS}


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label}: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True, help="local authorized prediction directory containing <dataset>/seed<seed>.npz")
    args = parser.parse_args()
    per_seed_rows = read_csv(ROOT / "results" / "test_results_per_seed.csv")
    per_seed = {(row["dataset"], int(row["seed"])): row for row in per_seed_rows}
    expected_keys = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(per_seed) != expected_keys:
        raise RuntimeError("per-seed CSV is not exactly 3 datasets x 3 seeds")

    recomputed: dict[tuple[str, int], dict[str, float]] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = args.predictions_root / dataset / f"seed{seed}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            got = recompute(path)
            recomputed[(dataset, seed)] = got
            row = per_seed[(dataset, seed)]
            for metric in METRICS:
                close(got[metric], float(row[metric]), 2e-6, f"per-seed {dataset}/seed{seed}/{metric}")

    mean_sd_rows = {row["dataset"]: row for row in read_csv(ROOT / "results" / "test_results_mean_sd.csv")}
    final_json = json.loads((ROOT / "results" / "FINAL_TEST_RESULTS.json").read_text(encoding="utf-8"))
    final_aggregate = final_json["aggregate_mean_sample_sd"]

    print("Paper result regression (three-seed arithmetic mean ± sample SD):")
    for dataset in DATASETS:
        aggregate: dict[str, tuple[float, float]] = {}
        for metric in METRICS:
            values = [recomputed[(dataset, seed)][metric] for seed in SEEDS]
            mean = statistics.mean(values)
            sample_sd = statistics.stdev(values)
            aggregate[metric] = (mean, sample_sd)
            csv_row = mean_sd_rows[dataset]
            close(mean, float(csv_row[f"{metric}_mean"]), 2e-6, f"mean CSV {dataset}/{metric}")
            close(sample_sd, float(csv_row[f"{metric}_sample_sd"]), 2e-6, f"SD CSV {dataset}/{metric}")
            if metric in {"jaccard", "f1", "prauc"}:
                target = final_aggregate[dataset][metric]
                close(mean, float(target["mean"]), 2e-6, f"FINAL JSON mean {dataset}/{metric}")
                close(sample_sd, float(target["sample_sd"]), 2e-6, f"FINAL JSON SD {dataset}/{metric}")

        display, expected_j, expected_f1, expected_prauc = PAPER_PERCENT_TARGETS[dataset]
        formatted = (
            f"{aggregate['jaccard'][0] * 100:.2f} ± {aggregate['jaccard'][1] * 100:.2f}",
            f"{aggregate['f1'][0] * 100:.2f} ± {aggregate['f1'][1] * 100:.2f}",
            f"{aggregate['prauc'][0] * 100:.2f} ± {aggregate['prauc'][1] * 100:.2f}",
        )
        expected = (expected_j, expected_f1, expected_prauc)
        if formatted != expected:
            raise RuntimeError(f"paper-format regression failed for {dataset}: {formatted} != {expected}")
        print(f"{display}: Jaccard {formatted[0]} | F1 {formatted[1]} | PRAUC {formatted[2]}")

    print("PASS: all 9 authorized local predictions reproduce the paper table, per-seed CSV, mean/sample-SD CSV, FINAL_TEST_RESULTS.json, and count MAE.")
    print("DDI is not recomputed from NPZ alone; eICU DDI remains N/A because its packaged matrix has no nonzero interactions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
