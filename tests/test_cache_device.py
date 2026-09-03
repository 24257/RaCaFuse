import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cache_identity import content_digest, file_digest


class CacheDeviceTest(unittest.TestCase):
    def test_graph_cache_tracks_records_and_vocabulary(self):
        tree = ast.parse((ROOT / "src" / "trainer.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cached_directed_weighted_ehr_adj")
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "Path": Path, "args": types.SimpleNamespace(ehr_graph_cache_dir=directory),
                "dataset": "synthetic", "dataset_cache_fingerprint": content_digest({"vocab": ["A", "B"]}),
                "stable_digest": content_digest,
                "seer_security": types.SimpleNamespace(load_pickle=json.load),
                "dill": types.SimpleNamespace(dump=json.dump),
            }
            # Exercise the production function without importing the training CLI
            # or loading clinical artifacts. The fake serializer uses bytes too.
            namespace["dill"].dump = lambda value, handle: handle.write(json.dumps(value).encode())
            builder = Mock(side_effect=lambda records, size: records)
            namespace["create_directed_weighted_ehr_adj"] = builder
            exec(compile(ast.Module(body=[function], type_ignores=[]), "cache_function", "exec"), namespace)
            cache = namespace["cached_directed_weighted_ehr_adj"]
            records = [[[[0], [0], [0]]]]
            (Path(directory) / "ehr_graph_plus-legacy.pkl").write_bytes(b"legacy")
            cache(records, 2, [0])
            cache(records, 2, [0])
            self.assertEqual(builder.call_count, 1)
            changed = copy.deepcopy(records)
            changed[0][0][2] = [1]
            self.assertEqual(cache(changed, 2, [0]), changed)
            self.assertEqual(builder.call_count, 2)
            namespace["dataset_cache_fingerprint"] = content_digest({"vocab": ["B", "A"]})
            cache(changed, 2, [0])
            self.assertEqual(builder.call_count, 3)

    def test_file_identity_detects_same_size_same_timestamp_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"abcd")
            stat = path.stat()
            before = file_digest(path)
            path.write_bytes(b"abce")
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(before, file_digest(path))

    def test_both_branches_and_calibration_receive_device(self):
        for cuda in (1, -1):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--dataset", "mimic-iii", "--seed", "1203", "--cuda", str(cuda), "--refit-calibration", "--dry-run"],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
            )
            commands = [line for line in result.stdout.splitlines() if "trainer.py" in line or "seer_auxiliary_adapter.py" in line]
            self.assertEqual(len(commands), 4)
            for command in commands:
                self.assertIn(f"--cuda {cuda}", command)

    def test_trainer_selects_requested_device(self):
        tree = ast.parse((ROOT / "src" / "trainer.py").read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "device" for t in node.targets))
        for cuda, available, expected in ((1, True, "cuda:1"), (-1, True, "cpu"), (0, False, "cpu")):
            namespace = {"args": types.SimpleNamespace(cuda=cuda), "torch": types.SimpleNamespace(device=lambda value: value, cuda=types.SimpleNamespace(is_available=lambda: available))}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), "device_selection", "exec"), namespace)
            self.assertEqual(namespace["device"], expected)


if __name__ == "__main__":
    unittest.main()
