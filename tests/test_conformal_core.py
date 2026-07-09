import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    from conformal import (  # noqa: E402
        aps_nonconformity,
        aps_scores,
        conformal_nu_scores,
        conformal_quantile,
        dither_scores,
        descending_order,
        margin_scores,
        mondrian_quantiles,
    )
    from samplers import get_keep_mask  # noqa: E402
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ConformalCoreTests(unittest.TestCase):
    def test_impossible_finite_quantile_returns_infinity(self):
        scores = torch.arange(18, dtype=torch.float64)

        self.assertTrue(math.isinf(conformal_quantile(scores, delta=0.05)))
        self.assertEqual(
            conformal_quantile(torch.arange(19, dtype=torch.float64), delta=0.05),
            18.0,
        )

    def test_leave_one_out_rank_coverage_matches_finite_sample_rule(self):
        population = torch.arange(5, dtype=torch.float64)
        covered = 0
        for test_index in range(5):
            calibration = torch.cat(
                (population[:test_index], population[test_index + 1 :])
            )
            q_hat = conformal_quantile(calibration, delta=0.2)
            covered += int(float(population[test_index]) <= q_hat)
        self.assertEqual(covered, 4)

        for test_index in range(5):
            calibration = torch.cat(
                (population[:test_index], population[test_index + 1 :])
            )
            self.assertTrue(math.isinf(conformal_quantile(calibration, delta=0.1)))

    def test_quantile_rejects_nan_but_allows_infinite_thresholds(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            conformal_quantile(torch.tensor([0.0, float("nan")]), delta=0.1)
        self.assertTrue(
            math.isinf(
                conformal_quantile(
                    torch.tensor([0.0, float("inf")]),
                    delta=0.5,
                )
            )
        )

    def test_dither_is_explicit_reproducible_and_float64(self):
        scores = torch.full((4,), 1e8, dtype=torch.float32)
        uniforms = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)
        left = dither_scores(scores, uniforms)
        right = dither_scores(scores, uniforms)

        self.assertEqual(left.dtype, torch.float64)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.all(left >= 1e8))
        self.assertTrue(torch.all(left < 1e8 + 1e-6))
        self.assertGreater(torch.unique(left).numel(), 1)

    def test_dither_validates_uniforms_and_epsilon(self):
        scores = torch.ones(2)
        invalid_uniforms = (
            torch.zeros(1),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, -0.1]),
            torch.tensor([0.0, float("nan")]),
        )
        for uniforms in invalid_uniforms:
            with self.subTest(uniforms=uniforms):
                with self.assertRaisesRegex(ValueError, "uniforms"):
                    dither_scores(scores, uniforms)
        for epsilon in (0.0, -1.0, float("inf")):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "epsilon"):
                    dither_scores(
                        scores,
                        torch.zeros_like(scores),
                        epsilon=epsilon,
                    )

    def test_mondrian_reports_counts_and_vacuous_small_groups(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 11.0])
        groups = torch.tensor([0, 0, 0, 0, 1, 1])

        result = mondrian_quantiles(
            scores,
            groups,
            delta=0.25,
            min_bucket=3,
        )

        by_group = {item.group: item for item in result}
        self.assertEqual(by_group[0].count, 4)
        self.assertEqual(by_group[0].q_hat, 4.0)
        self.assertTrue(by_group[0].finite)
        self.assertEqual(by_group[0].reason, "finite")
        self.assertEqual(by_group[1].count, 2)
        self.assertTrue(math.isinf(by_group[1].q_hat))
        self.assertFalse(by_group[1].finite)
        self.assertEqual(by_group[1].reason, "below_min_bucket")

    def test_mondrian_default_floor_is_five_over_delta(self):
        result = mondrian_quantiles(
            torch.arange(49, dtype=torch.float64),
            torch.zeros(49, dtype=torch.int64),
            delta=0.1,
        )

        self.assertTrue(math.isinf(result[0].q_hat))
        self.assertEqual(result[0].reason, "below_min_bucket")

    def test_mondrian_reports_expected_but_absent_groups(self):
        result = mondrian_quantiles(
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 0]),
            delta=0.25,
            expected_groups=(0, 1),
            min_bucket=1,
        )

        by_group = {item.group: item for item in result}
        self.assertEqual(by_group[1].count, 0)
        self.assertTrue(math.isinf(by_group[1].q_hat))
        self.assertEqual(by_group[1].reason, "absent")

    def test_mondrian_reports_rank_exceeding_group_size(self):
        result = mondrian_quantiles(
            torch.arange(4, dtype=torch.float64),
            torch.zeros(4, dtype=torch.int64),
            delta=0.1,
            min_bucket=1,
        )

        self.assertTrue(math.isinf(result[0].q_hat))
        self.assertEqual(result[0].reason, "rank_exceeds_n")

    def test_mondrian_can_report_all_expected_groups_as_absent(self):
        result = mondrian_quantiles(
            torch.empty(0),
            torch.empty(0, dtype=torch.int64),
            delta=0.1,
            expected_groups=(0, 1),
        )

        self.assertEqual(tuple(item.group for item in result), (0, 1))
        self.assertTrue(all(item.reason == "absent" for item in result))

    def test_mondrian_requires_integer_one_dimensional_groups(self):
        scores = torch.tensor([1.0, 2.0])
        invalid = (
            torch.tensor([0.0, 1.0]),
            torch.tensor([[0, 1]]),
            torch.tensor([0]),
        )
        for groups in invalid:
            with self.subTest(shape=tuple(groups.shape), dtype=groups.dtype):
                with self.assertRaisesRegex(ValueError, "groups"):
                    mondrian_quantiles(scores, groups, delta=0.1, min_bucket=1)

    def test_c_nu_at_zero_is_exactly_c_margin(self):
        torch.manual_seed(7)
        logits = torch.randn(5, 17, dtype=torch.float64)
        frequencies = torch.arange(17, dtype=torch.int64)

        self.assertTrue(
            torch.equal(
                conformal_nu_scores(logits, frequencies, kappa=0.0),
                margin_scores(logits),
            )
        )

    def test_c_margin_is_calibrated_min_p(self):
        torch.manual_seed(11)
        logits = torch.randn(8, 31, dtype=torch.float64)
        q_hat = 1.7

        conformal_keep = margin_scores(logits) <= q_hat
        min_p_keep = get_keep_mask(
            logits,
            "min_p",
            p_min=math.exp(-q_hat),
        )

        self.assertTrue(torch.equal(conformal_keep, min_p_keep))

    def test_deterministic_aps_is_nucleus_with_crossing_token(self):
        torch.manual_seed(19)
        logits = torch.randn(8, 31, dtype=torch.float64)
        threshold = 0.73

        order = descending_order(logits)
        aps_keep = aps_scores(
            logits,
            order=order,
            uniforms=torch.zeros_like(logits),
        ) <= threshold
        nucleus_keep = get_keep_mask(logits, "top_p", p=threshold)

        self.assertTrue(torch.equal(aps_keep, nucleus_keep))

    def test_aps_target_scores_gather_full_scores(self):
        logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 4.0, 2.0]])
        target_ids = torch.tensor([1, 2])
        uniforms = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        order = descending_order(logits)
        full = aps_scores(logits, order=order, uniforms=uniforms)
        target = aps_nonconformity(
            logits,
            target_ids,
            order=order,
            uniforms=uniforms,
        )

        self.assertTrue(
            torch.equal(target, full.gather(1, target_ids.unsqueeze(1)).squeeze(1))
        )

    def test_aps_tie_order_is_explicit(self):
        logits = torch.tensor([[2.0, 2.0, 0.0]])
        uniforms = torch.zeros_like(logits)
        left_first = torch.tensor([[0, 1, 2]])
        right_first = torch.tensor([[1, 0, 2]])

        left_scores = aps_scores(logits, order=left_first, uniforms=uniforms)
        right_scores = aps_scores(logits, order=right_first, uniforms=uniforms)

        self.assertEqual(float(left_scores[0, 0]), 0.0)
        self.assertGreater(float(left_scores[0, 1]), 0.0)
        self.assertEqual(float(right_scores[0, 1]), 0.0)
        self.assertGreater(float(right_scores[0, 0]), 0.0)

    def test_aps_u_one_is_not_deterministic_nucleus(self):
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64))
        order = torch.tensor([[0, 1, 2]])
        threshold = 0.8

        u_zero = aps_scores(
            logits,
            order=order,
            uniforms=torch.zeros_like(logits),
        ) <= threshold
        u_one = aps_scores(
            logits,
            order=order,
            uniforms=torch.ones_like(logits),
        ) <= threshold

        self.assertEqual(u_zero.tolist(), [[True, True, False]])
        self.assertEqual(u_one.tolist(), [[True, False, False]])


if __name__ == "__main__":
    unittest.main()
