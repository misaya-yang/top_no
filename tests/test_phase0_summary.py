import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from summarize_phase0_queue import summarize_cells  # noqa: E402


class Phase0SummaryTests(unittest.TestCase):
    def cell(self, key, model, domain, *, effect, direction=1.0, perm_effect=0.05):
        signed = effect * direction
        return {
            "schema_version": "icml2027-phase0-summary-v1",
            "evidence_grade": "E-pilot",
            "paper_citable": False,
            "cell_key": key,
            "model_key": model,
            "domain_key": domain,
            "completion_status": "COMPLETE",
            "n_documents": 100,
            "n_positions": 10_000,
            "analysis": {
                "informative": True,
                "max_abs_shift": effect,
                "rare_minus_reference_shift": signed,
                "non_additive": False,
            },
            "permutation_analysis": {
                "informative": True,
                "max_abs_shift": perm_effect,
            },
            "half_analysis": [
                {"informative": True, "rare_minus_reference_shift": signed * 0.9},
                {"informative": True, "rare_minus_reference_shift": signed * 1.1},
            ],
        }

    def four_cells(self, *, effect, perm_effect):
        return [
            self.cell("3b_web", "qwen3b", "web", effect=effect, perm_effect=perm_effect),
            self.cell("3b_math", "qwen3b", "math", effect=effect, perm_effect=perm_effect),
            self.cell("7b_web", "qwen7b", "web", effect=effect, perm_effect=perm_effect),
            self.cell("7b_math", "qwen7b", "math", effect=effect, perm_effect=perm_effect),
        ]

    def test_four_replicated_cells_pass_plan_a(self):
        memo = summarize_cells(self.four_cells(effect=0.45, perm_effect=0.05))

        self.assertEqual(memo["verdict"], "PLAN_A_PILOT")
        self.assertTrue(memo["cross_domain_sign_agreement"])
        self.assertTrue(memo["cross_scale_sign_agreement"])

    def test_two_domains_with_small_effect_support_plan_b(self):
        cells = [
            self.cell("3b_web", "qwen3b", "web", effect=0.08),
            self.cell("3b_math", "qwen3b", "math", effect=0.09),
        ]

        memo = summarize_cells(cells)

        self.assertEqual(memo["verdict"], "PLAN_B_PILOT")

    def test_one_cell_is_insufficient(self):
        memo = summarize_cells(
            [self.cell("3b_web", "qwen3b", "web", effect=0.45)]
        )

        self.assertEqual(memo["verdict"], "INSUFFICIENT")

    def test_permutation_control_that_matches_effect_blocks_plan_a(self):
        memo = summarize_cells(self.four_cells(effect=0.45, perm_effect=0.40))

        self.assertEqual(memo["verdict"], "INSUFFICIENT")

    def test_half_sign_instability_blocks_plan_a(self):
        cells = self.four_cells(effect=0.45, perm_effect=0.05)
        cells[0]["half_analysis"][1]["rare_minus_reference_shift"] = -0.45

        memo = summarize_cells(cells)

        self.assertEqual(memo["verdict"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
