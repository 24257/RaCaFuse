import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_fusion  # noqa: E402


class FinalFusionTest(unittest.TestCase):
    def test_frozen_defaults(self):
        params = inspect.signature(final_fusion.fuse_components).parameters
        self.assertEqual(params["score_long_weight"].default, 0.3)
        self.assertEqual(params["count_long_weight"].default, 0.4)
        self.assertEqual(params["count_offset"].default, 1)
        self.assertEqual(params["ddi_penalty"].default, 0.04)
        self.assertEqual(params["ddi_window"].default, 60)


if __name__ == "__main__":
    unittest.main()
