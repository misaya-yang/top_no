import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    import conformal  # noqa: E402
    from conformal import (  # noqa: E402
        aps_nonconformity,
        aps_scores,
        conformal_nu_scores,
        conformal_quantile,
        dither_scores,
        descending_order,
        margin_scores,
        margin_nonconformity,
        logprob_nonconformity,
        logprob_scores,
        mondrian_quantiles,
        nu_nonconformity,
        zmargin_nonconformity,
        zmargin_scores,
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

    @unittest.skipUnless(
        torch is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built(),
        "MPS is unavailable",
    )
    def test_dither_rejects_mps_with_controlled_cpu_guidance(self):
        scores = torch.ones(2, device="mps")

        with self.assertRaisesRegex(ValueError, "MPS.*move.*CPU"):
            dither_scores(scores, torch.zeros(2))

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

    def test_target_scores_do_not_materialize_full_candidate_scores(self):
        logits = torch.tensor([[4.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
        target_ids = torch.tensor([1, 2])
        frequencies = torch.tensor([10, 20, 30])

        with patch.object(
            conformal,
            "margin_scores",
            side_effect=AssertionError("full candidate scores were materialized"),
        ):
            margin = margin_nonconformity(logits, target_ids)
            nu = nu_nonconformity(
                logits,
                target_ids,
                frequencies,
                kappa=0.0,
            )

        self.assertEqual(margin.tolist(), [2.0, 1.0])
        self.assertTrue(torch.equal(nu, margin))

        with patch.object(
            conformal,
            "logprob_scores",
            side_effect=AssertionError("full candidate scores were materialized"),
        ), patch.object(
            conformal,
            "_logprob_working_logits",
            side_effect=AssertionError("full fp32 working logits were materialized"),
        ):
            logprob = logprob_nonconformity(logits, target_ids)
        expected = -torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        self.assertTrue(torch.allclose(logprob, expected))

    def test_c_logprob_scores_are_stable_negative_log_probabilities(self):
        logits = torch.tensor([[0.0, -100.0]], dtype=torch.float16)
        scores = logprob_scores(logits)

        self.assertEqual(scores.dtype, torch.float32)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.allclose(scores, torch.tensor([[0.0, 100.0]])))
        shifted = logprob_scores(logits + 50.0)
        self.assertTrue(torch.equal(scores, shifted))

        for dtype, offset in (
            (torch.float16, 60_000.0),
            (torch.float32, 1e8),
            (torch.float64, 1e16),
        ):
            with self.subTest(dtype=dtype):
                equal_logits = torch.full((2, 3), offset, dtype=dtype)
                expected = torch.full(
                    (2, 3),
                    math.log(3.0),
                    dtype=torch.float64 if dtype == torch.float64 else torch.float32,
                )
                full = logprob_scores(equal_logits)
                target = logprob_nonconformity(
                    equal_logits,
                    torch.tensor([0, 2]),
                )
                self.assertTrue(torch.allclose(full, expected, atol=1e-6))
                self.assertTrue(torch.allclose(target, expected[:, 0], atol=1e-6))

    def test_c_zmargin_is_shift_stable_and_handles_zero_variance(self):
        logits = torch.tensor([[0.0, 8.0], [3.0, 3.0]], dtype=torch.float32)
        shifted = torch.tensor([[1e8, 1e8 + 8.0], [1e8, 1e8]], dtype=torch.float32)

        scores = zmargin_scores(logits)
        shifted_scores = zmargin_scores(shifted)

        self.assertTrue(torch.allclose(scores, shifted_scores, atol=1e-6))
        self.assertEqual(scores[1].tolist(), [0.0, 0.0])
        targets = torch.tensor([0, 1])
        with patch.object(
            conformal,
            "_zmargin_working_logits",
            side_effect=AssertionError("full fp32 working logits were materialized"),
        ):
            target_scores = zmargin_nonconformity(shifted, targets)
        self.assertTrue(
            torch.allclose(
                target_scores,
                shifted_scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1),
            )
        )

    def test_c_zmargin_is_calibrated_top_nsigma(self):
        torch.manual_seed(31)
        logits = torch.randn(8, 31, dtype=torch.float64)
        q_hat = 1.7

        conformal_keep = zmargin_scores(logits) <= q_hat
        nsigma_keep = get_keep_mask(logits, "top_nsigma", n_sigma=q_hat)

        self.assertTrue(torch.equal(conformal_keep, nsigma_keep))

    def test_c_zmargin_target_matches_full_across_chunk_boundary(self):
        torch.manual_seed(41)
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits = torch.randn(2, 4097, dtype=dtype)
                targets = torch.tensor([0, 4096])
                full = zmargin_scores(logits).gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                target = zmargin_nonconformity(logits, targets)
                self.assertTrue(torch.equal(target, full))

        singleton = torch.tensor([[7.0]])
        self.assertEqual(zmargin_scores(singleton).tolist(), [[0.0]])
        self.assertEqual(
            zmargin_nonconformity(singleton, torch.tensor([0])).tolist(),
            [0.0],
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

    def test_c_margin_min_p_equivalence_survives_fp16_underflow(self):
        logits = torch.tensor([[0.0, -100.0]], dtype=torch.float16)
        q_hat = 20.0

        conformal_keep = margin_scores(logits) <= q_hat
        min_p_keep = get_keep_mask(
            logits,
            "min_p",
            p_min=math.exp(-q_hat),
        )

        self.assertEqual(conformal_keep.tolist(), [[True, False]])
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

    def test_aps_top_p_equivalence_survives_fp16_boundary_rounding(self):
        logits = torch.tensor(
            [[5.79296875, -7.75390625, 4.40625]],
            dtype=torch.float16,
        )
        threshold = 0.8
        order = descending_order(logits)

        aps_keep = aps_scores(
            logits,
            order=order,
            uniforms=torch.zeros_like(logits),
        ) <= threshold
        nucleus_keep = get_keep_mask(logits, "top_p", p=threshold)

        self.assertEqual(nucleus_keep.tolist(), [[True, False, False]])
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

    def test_aps_rejects_non_permutation_orders(self):
        logits = torch.tensor([[3.0, 2.0, 1.0]])
        uniforms = torch.zeros_like(logits)
        for order in (torch.tensor([[0, 0, 2]]), torch.tensor([[0, 1, 3]])):
            with self.subTest(order=order.tolist()):
                with self.assertRaisesRegex(ValueError, "permutation"):
                    aps_scores(logits, order=order, uniforms=uniforms)

    def test_aps_u_near_one_is_not_deterministic_nucleus(self):
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64))
        order = torch.tensor([[0, 1, 2]])
        threshold = 0.8

        u_zero = aps_scores(
            logits,
            order=order,
            uniforms=torch.zeros_like(logits),
        ) <= threshold
        near_one = torch.nextafter(torch.ones_like(logits), torch.zeros_like(logits))
        u_near_one = aps_scores(
            logits,
            order=order,
            uniforms=near_one,
        ) <= threshold

        self.assertEqual(u_zero.tolist(), [[True, True, False]])
        self.assertEqual(u_near_one.tolist(), [[True, False, False]])


if __name__ == "__main__":
    unittest.main()
