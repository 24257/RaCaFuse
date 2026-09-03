#!/usr/bin/env python3
"""Create a tiny non-clinical NPZ fixture for SEER metric/fusion smoke tests."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/sample/synthetic_predictions.npz")
    args = parser.parse_args()
    y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.int8)
    ranking_score = np.array([[2.0, 0.1, 1.5, -0.5], [0.0, 1.4, -0.2, 1.2], [1.3, 1.1, 0.2, -0.4]], dtype=np.float32)
    y_pred = (ranking_score > 0.8).astype(np.int8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, y_true=y_true, y_pred=y_pred, ranking_score=ranking_score, predicted_count=y_pred.sum(axis=1).astype(np.int32))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
