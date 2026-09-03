from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import protocol  # noqa: E402


class ProtocolTest(unittest.TestCase):
    @staticmethod
    def option(command, name):
        index = command.index(name)
        return command[index + 1]

    def test_engine_is_public_src_trainer(self):
        self.assertEqual(protocol.ENGINE, ROOT / "src" / "trainer.py")

    def test_shared_training_parameters_are_identical(self):
        for dataset in ("mimic-iii", "mimic-iv", "eicu"):
            command = protocol.stage_a_command(
                dataset, 1203, ROOT / "data" / dataset, ROOT / "artifacts" / dataset / "stage_a.pth"
            )
            self.assertEqual(self.option(command, "--batch"), "32")
            self.assertEqual(self.option(command, "--weight_decay"), "1e-06")
            self.assertEqual(self.option(command, "--bce_weight"), "0.7")
            self.assertEqual(self.option(command, "--soft_jaccard_weight"), "0.2")
            self.assertEqual(self.option(command, "--pairwise_rank_weight"), "0.02")
            self.assertIn("--validation_only_no_test", command)
            self.assertEqual(self.option(command, "--split_path"), str(protocol.training_split_path(dataset, 1203)))
        eicu = protocol.stage_a_command(
            "eicu", 1203, ROOT / "data" / "eicu", ROOT / "artifacts" / "eicu" / "stage_a.pth"
        )
        self.assertIn("--disable_procedure_branch", eicu)

    def test_final_split_semantics(self):
        for dataset in ("mimic-iii", "mimic-iv"):
            command = protocol.longitudinal_export_command(
                dataset, 1203, ROOT / "data" / dataset, Path("a.pth"), Path("b.pth"), "test", Path("out.npz"), Path("cache")
            )
            self.assertNotIn("--split_path", command)
        eicu = protocol.longitudinal_export_command(
            "eicu", 1203, ROOT / "data" / "eicu", Path("a.pth"), Path("b.pth"), "test", Path("out.npz"), Path("cache")
        )
        self.assertEqual(self.option(eicu, "--split_path"), str(protocol.final_split_path("eicu", 1203)))
        self.assertNotIn("--disable_procedure_branch", eicu)


if __name__ == "__main__":
    unittest.main()
