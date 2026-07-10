import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase0QueueConfigTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "configs" / "phase0_two_hour_qwen.json"

    def test_matrix_freezes_order_budget_and_model_revisions(self):
        matrix = json.loads(self.path.read_text())

        self.assertEqual(matrix["schema_version"], "icml2027-phase0-two-hour-matrix-v1")
        self.assertEqual(matrix["queue_wall_seconds"], 6600)
        self.assertEqual(matrix["cell_wall_seconds"], 1500)
        self.assertEqual(
            [cell["key"] for cell in matrix["cells"]],
            ["3b_web", "3b_math", "7b_web", "7b_math"],
        )
        models = {item["key"]: item for item in matrix["models"]}
        self.assertEqual(
            models["qwen3b"]["revision"],
            "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        )
        self.assertEqual(
            models["qwen7b"]["revision"],
            "d149729398750b98c0af14eb82c78cfe92750796",
        )

    def test_cells_freeze_pilot_sampling_and_never_enable_legacy(self):
        matrix = json.loads(self.path.read_text())

        self.assertNotIn("allow_legacy_protocol", matrix)
        for cell in matrix["cells"]:
            self.assertEqual(cell["stride"], 4)
            self.assertEqual(cell["min_context"], 16)
            self.assertEqual(cell["max_length"], 512)
            self.assertEqual(cell["min_true_count"], 20)
            self.assertEqual(cell["seed"], 1729)
            if cell["model_key"] == "qwen3b":
                self.assertEqual(cell["batch_size"], 2)
                self.assertEqual(cell["max_positions"], 80_000)
            else:
                self.assertEqual(cell["batch_size"], 1)
                self.assertEqual(cell["max_positions"], 50_000)

    def test_all_artifact_paths_are_relative(self):
        matrix = json.loads(self.path.read_text())

        for value in (matrix["frequency_manifest"], matrix["frequency_document_jsonl"]):
            self.assertFalse(Path(value).is_absolute())
        for item in matrix["models"]:
            self.assertFalse(Path(item["frequency_dir"]).is_absolute())
        for domain in matrix["domains"]:
            for key, value in domain.items():
                if key != "key":
                    self.assertFalse(Path(value).is_absolute())


if __name__ == "__main__":
    unittest.main()
