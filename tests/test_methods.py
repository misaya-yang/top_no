import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from methods import (  # noqa: E402
        PAPER_REQUIRED_METHOD_KEYS,
        calibrate_method,
        get_method_definition,
        implemented_method_keys,
        method_registry,
        missing_paper_method_keys,
        prediction_set_mask,
    )
    from samplers import get_keep_mask  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class MethodRegistryTests(unittest.TestCase):
    def test_registry_has_stable_unique_keys_and_explicit_implementation_status(self):
        expected = {
            "c_margin",
            "c_logprob",
            "c_zmargin",
            "aps",
            "raps",
            "ts_aps",
            "cns",
            "entropy_mondrian_margin",
            "frequency_mondrian_margin",
            "learned_h",
            "learned_g",
            "c_nu",
        }
        definitions = method_registry()

        self.assertEqual({item.key for item in definitions}, expected)
        self.assertEqual(len(definitions), len(expected))
        self.assertEqual(
            implemented_method_keys(),
            {
                "c_margin",
                "aps",
                "c_nu",
                "entropy_mondrian_margin",
                "frequency_mondrian_margin",
            },
        )
        self.assertTrue(PAPER_REQUIRED_METHOD_KEYS < expected)
        self.assertFalse(PAPER_REQUIRED_METHOD_KEYS <= implemented_method_keys())
        self.assertEqual(
            missing_paper_method_keys(),
            PAPER_REQUIRED_METHOD_KEYS - implemented_method_keys(),
        )

    def test_lookup_rejects_unknown_and_calibration_rejects_unimplemented_method(self):
        with self.assertRaisesRegex(ValueError, "unknown method key"):
            get_method_definition("conformal_nu_k10")
        with self.assertRaisesRegex(NotImplementedError, "raps"):
            calibrate_method(
                "raps",
                torch.zeros((1, 2)),
                torch.zeros(1, dtype=torch.long),
                delta=0.1,
                uniforms=torch.zeros((1, 2)),
            )

    def test_c_margin_calibrates_and_matches_min_p_with_zero_dither(self):
        calibration_logits = torch.tensor(
            [[3.0, 2.0, 0.0], [2.0, 0.0, 1.0], [4.0, 1.0, 2.0], [3.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 2, 2, 1])
        fitted = calibrate_method(
            "c_margin",
            calibration_logits,
            targets,
            delta=0.25,
            uniforms=torch.zeros_like(calibration_logits),
        )
        test_logits = torch.tensor([[4.0, 2.0, 1.0]], dtype=torch.float64)
        keep = prediction_set_mask(
            fitted,
            test_logits,
            uniforms=torch.zeros_like(test_logits),
        )

        self.assertEqual(fitted.q_hat, 2.0)
        self.assertTrue(
            torch.equal(
                keep,
                get_keep_mask(test_logits, "min_p", p_min=math.exp(-2.0)),
            )
        )

    def test_c_nu_at_zero_matches_c_margin_end_to_end(self):
        torch.manual_seed(17)
        logits = torch.randn(19, 11, dtype=torch.float64)
        targets = torch.arange(19) % 11
        frequencies = torch.arange(11)
        uniforms = torch.zeros_like(logits)

        margin = calibrate_method(
            "c_margin", logits, targets, delta=0.1, uniforms=uniforms
        )
        nu = calibrate_method(
            "c_nu",
            logits,
            targets,
            delta=0.1,
            uniforms=uniforms,
            token_freq_table=frequencies,
            params={"kappa": 0.0, "alpha": 1.0},
        )

        self.assertEqual(margin.q_hat, nu.q_hat)
        self.assertTrue(
            torch.equal(
                prediction_set_mask(
                    margin, logits, uniforms=uniforms
                ),
                prediction_set_mask(
                    nu,
                    logits,
                    uniforms=uniforms,
                    token_freq_table=frequencies,
                ),
            )
        )

    def test_finite_sample_rank_overflow_stays_vacuous(self):
        logits = torch.stack(
            (torch.arange(18, dtype=torch.float64), torch.zeros(18)),
            dim=1,
        )
        targets = torch.ones(18, dtype=torch.long)
        fitted = calibrate_method(
            "c_margin",
            logits,
            targets,
            delta=0.05,
            uniforms=torch.zeros_like(logits),
        )

        self.assertTrue(math.isinf(fitted.q_hat))
        self.assertTrue(
            prediction_set_mask(
                fitted,
                logits[:2],
                uniforms=torch.zeros_like(logits[:2]),
            ).all()
        )

    def test_signed_kappa_is_explicit_and_does_not_mutate_margin_calibration(self):
        logits = torch.tensor(
            [[3.0, 1.0], [4.0, 2.0], [2.0, 0.0]], dtype=torch.float64
        )
        targets = torch.ones(3, dtype=torch.long)
        frequencies = torch.zeros(2)
        uniforms = torch.zeros_like(logits)
        margin_before = calibrate_method(
            "c_margin", logits, targets, delta=0.5, uniforms=uniforms
        )
        positive = calibrate_method(
            "c_nu",
            logits,
            targets,
            delta=0.5,
            uniforms=uniforms,
            token_freq_table=frequencies,
            params={"kappa": 1.0, "alpha": 1.0},
        )
        negative = calibrate_method(
            "c_nu",
            logits,
            targets,
            delta=0.5,
            uniforms=uniforms,
            token_freq_table=frequencies,
            params={"kappa": -1.0, "alpha": 1.0},
        )
        margin_after = calibrate_method(
            "c_margin", logits, targets, delta=0.5, uniforms=uniforms
        )

        self.assertEqual(margin_before, margin_after)
        self.assertLess(positive.q_hat, negative.q_hat)
        with self.assertRaisesRegex(ValueError, "exactly.*alpha.*kappa"):
            calibrate_method(
                "c_nu",
                logits,
                targets,
                delta=0.5,
                uniforms=uniforms,
                token_freq_table=frequencies,
                params={"kappa": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "token_freq_table"):
            calibrate_method(
                "c_nu",
                logits,
                targets,
                delta=0.5,
                uniforms=uniforms,
                params={"kappa": 1.0, "alpha": 1.0},
            )

    def test_aps_zero_boundary_uniforms_match_top_p_at_fitted_threshold(self):
        logits = torch.log(
            torch.tensor(
                [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1]],
                dtype=torch.float64,
            )
        )
        targets = torch.tensor([1, 1, 0])
        fitted = calibrate_method(
            "aps",
            logits,
            targets,
            delta=0.5,
            uniforms=torch.zeros_like(logits),
        )
        keep = prediction_set_mask(
            fitted,
            logits,
            uniforms=torch.zeros_like(logits),
        )

        self.assertTrue(
            torch.equal(
                keep,
                get_keep_mask(logits, "top_p", p=fitted.q_hat),
            )
        )

    def test_frequency_mondrian_uses_candidate_token_groups_and_vacuous_absent_group(self):
        logits = torch.tensor(
            [[4.0, 3.0, 1.0], [4.0, 3.0, 2.0], [3.0, 0.0, 2.0], [3.0, 2.0, 1.0]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 1, 0, 1])
        token_groups = torch.tensor([0, 1, 2])
        fitted = calibrate_method(
            "frequency_mondrian_margin",
            logits,
            targets,
            delta=0.5,
            uniforms=torch.zeros_like(logits),
            groups=token_groups[targets],
            expected_groups=(0, 1, 2),
            min_bucket=1,
        )
        test_logits = torch.tensor([[5.0, 3.0, -10.0]], dtype=torch.float64)
        keep = prediction_set_mask(
            fitted,
            test_logits,
            uniforms=torch.zeros_like(test_logits),
            groups=token_groups,
        )

        thresholds = {item.group: item for item in fitted.group_quantiles}
        self.assertEqual(thresholds[0].q_hat, 0.0)
        self.assertEqual(thresholds[1].q_hat, 1.0)
        self.assertTrue(math.isinf(thresholds[2].q_hat))
        self.assertEqual(keep.tolist(), [[True, False, True]])

    def test_entropy_mondrian_applies_one_context_threshold_per_row(self):
        logits = torch.tensor(
            [[3.0, 2.0], [3.0, 1.0], [4.0, 1.0], [4.0, 0.0]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 1, 0, 1])
        fitted = calibrate_method(
            "entropy_mondrian_margin",
            logits,
            targets,
            delta=0.5,
            uniforms=torch.zeros_like(logits),
            groups=torch.tensor([0, 0, 1, 1]),
            expected_groups=(0, 1),
            min_bucket=1,
        )
        test_logits = torch.tensor([[4.0, 2.0], [4.0, 2.0]], dtype=torch.float64)
        keep = prediction_set_mask(
            fitted,
            test_logits,
            uniforms=torch.zeros_like(test_logits),
            groups=torch.tensor([0, 1]),
        )

        self.assertEqual(keep.tolist(), [[True, True], [True, True]])
        with self.assertRaisesRegex(ValueError, "registered calibration group"):
            prediction_set_mask(
                fitted,
                test_logits[:1],
                uniforms=torch.zeros_like(test_logits[:1]),
                groups=torch.tensor([2]),
            )

    def test_uniforms_and_group_shapes_fail_closed(self):
        logits = torch.tensor([[2.0, 1.0]])
        targets = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, "uniforms"):
            calibrate_method("c_margin", logits, targets, delta=0.5)
        with self.assertRaisesRegex(ValueError, "uniforms.*\[0, 1\)"):
            calibrate_method(
                "c_margin",
                logits,
                targets,
                delta=0.5,
                uniforms=torch.tensor([[0.0, float("nan")]]),
            )
        with self.assertRaisesRegex(ValueError, "groups"):
            calibrate_method(
                "frequency_mondrian_margin",
                logits,
                targets,
                delta=0.5,
                uniforms=torch.zeros_like(logits),
                groups=torch.tensor([[0]]),
                expected_groups=(0,),
                min_bucket=1,
            )


if __name__ == "__main__":
    unittest.main()
