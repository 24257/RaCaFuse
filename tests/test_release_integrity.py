from pathlib import Path
import re
import importlib.util
import subprocess
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTest(unittest.TestCase):
    def test_public_release_excludes_restricted_artifacts(self):
        self.assertTrue((ROOT / "src" / "seer_auxiliary_branch" / "modules" / "__init__.py").is_file())
        sys.path.insert(0, str(ROOT / "scripts"))
        from verify_public import check_public_tree
        check_public_tree()

    @unittest.skipUnless(all(importlib.util.find_spec(m) is not None for m in ("torch", "dill", "sklearn")), "training dependencies are not installed")
    def test_adapter_help_uses_bundled_runtime(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "baselines" / "seer_auxiliary_adapter.py"), "--help"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_no_private_paths_in_runtime_code_or_configs(self):
        forbidden = (".omx/logs", "SEER_FINAL_TEST_SUPPLEMENT", "C:/Users")
        numeric_mount = re.compile(r"/(?:\d{6,})/")
        roots = [ROOT / "scripts", ROOT / "src", ROOT / "baselines", ROOT / "configs"]
        suffixes = {".py", ".json", ".yaml", ".yml", ".txt"}
        for base in roots:
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in suffixes:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    self.assertIsNone(
                        numeric_mount.search(text),
                        f"absolute numeric mount path in {path.relative_to(ROOT)}",
                    )
                    for token in forbidden:
                        self.assertNotIn(token, text, f"{token} in {path.relative_to(ROOT)}")


if __name__ == "__main__":
    unittest.main()
