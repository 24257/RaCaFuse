#!/usr/bin/env python3
"""Run transparent final-fusion diagnostics from supplied branch target components.

This is a diagnostic utility, not a claim that a given setting is a manuscript-reported ablation.
"""
from __future__ import annotations

import sys as _seer_sys
from pathlib import Path as _SeerPath
_seer_sys.path.insert(0, str(_SeerPath(__file__).resolve().parents[1] / "src"))
import seer_security
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from final_fusion import AffineCountCalibrator, fuse_with_calibrators, load_npz  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mimic-iii", "mimic-iv", "eicu"])
    parser.add_argument("--seed", type=int, required=True, choices=[1203, 1204, 1205])
    parser.add_argument("--longitudinal", required=True, help="target longitudinal component NPZ")
    parser.add_argument("--seer_auxiliary", required=True, help="target SEERAuxiliary-derived component NPZ")
    parser.add_argument("--ddi", required=True, help="aligned ddi_A_final.pkl")
    parser.add_argument("--score-long-weight", type=float, default=0.3)
    parser.add_argument("--count-long-weight", type=float, default=0.4)
    parser.add_argument("--count-offset", type=int, default=1)
    parser.add_argument("--ddi-penalty", type=float, default=0.04)
    parser.add_argument("--ddi-window", type=int, default=60)
    args = parser.parse_args()
    calibration = json.loads((ROOT / "configs" / "count_calibration.json").read_text(encoding="utf-8"))
    row = calibration[args.dataset][str(args.seed)]
    with Path(args.ddi).open("rb") as handle:
        ddi = np.asarray(seer_security.load_pickle(handle))
    arrays, metrics = fuse_with_calibrators(
        load_npz(Path(args.longitudinal)), load_npz(Path(args.seer_auxiliary)), ddi,
        AffineCountCalibrator(**row["longitudinal"]), AffineCountCalibrator(**row["seer_auxiliary"]),
        score_long_weight=args.score_long_weight, count_long_weight=args.count_long_weight,
        count_offset=args.count_offset, ddi_penalty=args.ddi_penalty, ddi_window=args.ddi_window,
    )
    if args.dataset == "eicu":
        metrics["ddi_rate"] = None
    metrics["predicted_rows"] = int(arrays["y_pred"].shape[0])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
