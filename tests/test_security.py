import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import seer_security
from final_fusion import load_npz
from reproduce_results import close
from verify_public import check_public_tree


class SecurityTest(unittest.TestCase):
    def test_object_npz_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            np.savez(path, payload=np.array([{"value": 1}], dtype=object))
            with self.assertRaises(ValueError):
                load_npz(path)

    def test_pickle_denied_before_reader_runs(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PermissionError):
                seer_security.load_pickle(io.BytesIO(b"not a pickle"))
            with self.assertRaises(PermissionError):
                seer_security.read_pandas_pickle("untrusted.pkl")

    def test_checkpoint_has_no_unsafe_fallback(self):
        loader = Mock(side_effect=ValueError("unsupported object"))
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(load=loader)}):
            with self.assertRaises(ValueError):
                seer_security.load_checkpoint("untrusted.pth", map_location="cpu")
        loader.assert_called_once_with("untrusted.pth", map_location="cpu", weights_only=True)

    def test_nonfinite_metric_cannot_pass(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(RuntimeError):
                close(value, 0.5, 1e-6, "metric")

    def test_release_check_detects_private_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "patient_labels.npz").write_bytes(b"placeholder")
            with self.assertRaises(RuntimeError):
                check_public_tree(root)


if __name__ == "__main__":
    unittest.main()
