from pathlib import Path
import subprocess
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PublicUtilitiesTest(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=True)

    def test_main_table_contains_targets(self):
        out = self.run_script("reproduce_table1.py").stdout
        self.assertIn("57.73 ± 0.09", out)
        self.assertIn("82.58 ± 0.20", out)

    def test_seed_table_has_nine_rows(self):
        out = self.run_script("reproduce_table2.py").stdout
        self.assertEqual(sum(line.startswith("| MIMIC-") or line.startswith("| eICU") for line in out.splitlines()), 9)

    def test_baseline_registry_lists_local_expert(self):
        out = self.run_script("run_baselines.py", "--list").stdout
        self.assertIn("seer_auxiliary", out)
        self.assertIn("bundled-adapted-runtime", out)

    def test_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.npz"
            subprocess.run([sys.executable, str(ROOT / "data" / "make_synthetic_fixture.py"), "--output", str(path)], cwd=ROOT, check=True)
            with np.load(path) as payload:
                self.assertEqual(payload["y_true"].shape, (3, 4))
                np.testing.assert_array_equal(payload["predicted_count"], payload["y_pred"].sum(axis=1))

    def test_refit_calibration_uses_calibration_root_for_both_branches(self):
        target_root = str(Path("/tmp/seer-target-data"))
        calibration_root = str(Path("/tmp/seer-calibration-data"))
        out = self.run_script(
            "evaluate.py",
            "--dataset", "mimic-iii",
            "--seed", "1203",
            "--data-root", target_root,
            "--calibration-data-root", calibration_root,
            "--refit-calibration",
            "--dry-run",
        ).stdout
        auxiliary_lines = [line for line in out.splitlines() if "seer_auxiliary_adapter.py" in line]
        self.assertEqual(len(auxiliary_lines), 2)
        test_line = next(line for line in auxiliary_lines if "--eval_split test" in line)
        train_line = next(line for line in auxiliary_lines if "--eval_split train" in line)
        self.assertIn(f"--datadir {target_root}", test_line)
        self.assertIn(f"--datadir {calibration_root}", train_line)


if __name__ == "__main__":
    unittest.main()
