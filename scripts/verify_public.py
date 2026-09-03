#!/usr/bin/env python3
"""Check public-package boundaries and a synthetic numerical smoke test."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from final_fusion import load_npz, set_metrics


def check_public_tree(root=ROOT):
    sample = root / "data" / "sample" / "synthetic_predictions.npz"
    violations = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in {".git", ".venv", "venv", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if (path.suffix.lower() in {".pkl", ".pickle", ".pth", ".pt", ".model", ".safetensors", ".npz", ".npy", ".pem", ".key"} and path != sample
                or path.name == ".env" or path.name.startswith(".env.")
                or len(rel.parts) > 1 and rel.parts[0] == "data" and rel.parts[1] in {"mimic-iii", "mimic-iv", "eicu"}):
            violations.append(str(rel))
    if violations:
        raise RuntimeError("Non-public files detected; exclude them before packaging: " + ", ".join(violations))


def main():
    check_public_tree()
    arrays = load_npz(ROOT / "data" / "sample" / "synthetic_predictions.npz")
    metrics = set_metrics(arrays["y_true"], arrays["y_pred"], arrays["ranking_score"])
    np.testing.assert_array_equal(arrays["predicted_count"], arrays["y_pred"].sum(axis=1))
    for key in ("jaccard", "f1", "prauc"):
        if not np.isclose(metrics[key], 1.0):
            raise RuntimeError(f"synthetic metric mismatch: {key}")
    print("PASS: public artifact boundary and synthetic metrics. This does not reproduce clinical results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
