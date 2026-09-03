from pathlib import Path
import subprocess
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import unittest


class ReproduceResultsTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SEER_TEST_PREDICTIONS_ROOT"), "requires authorized local prediction files")
    def test_paper_targets(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "reproduce_results.py"), "--predictions-root", os.environ["SEER_TEST_PREDICTIONS_ROOT"]],
            cwd=root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        for expected in (
            "57.73 ± 0.09", "72.48 ± 0.08", "81.77 ± 0.05",
            "53.20 ± 0.06", "68.41 ± 0.03", "77.97 ± 0.06",
            "54.52 ± 0.16", "68.89 ± 0.13", "82.58 ± 0.20",
        ):
            self.assertIn(expected, proc.stdout)


if __name__ == "__main__":
    unittest.main()
