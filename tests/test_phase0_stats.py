import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phase0_stats import (  # noqa: E402
    DocumentGridStats,
    GridSpec,
    accumulate_document,
    analyze_grid,
    document_half,
    merge_document_stats,
    permuted_frequency_groups,
)


class Phase0StatsTests(unittest.TestCase):
    def test_default_margin_bins_match_frozen_boundaries(self):
        grid = GridSpec.default()
        margins = torch.tensor([0.0, 0.25, 0.25001, 10.0, 10.1, 20.0, 21.0])

        bins = grid.margin_bin_indices(margins)

        self.assertEqual(bins.tolist(), [0, 0, 1, 39, 40, 59, 60])
        self.assertEqual(grid.n_margin_bins, 61)

    def test_accumulate_counts_allowed_candidates_and_true_target(self):
        result = accumulate_document(
            "doc-a",
            torch.tensor([[4.0, 3.0, 1.0, 0.0]]),
            torch.tensor([1]),
            torch.tensor([0, 9, 10, 10_000_000], dtype=torch.int64),
            grid=GridSpec.default(),
            excluded_token_ids={3},
            permutation_seed=17,
        )

        self.assertEqual(int(result.den.sum()), 3)
        self.assertEqual(int(result.num.sum()), 1)
        self.assertEqual(int(result.perm_den.sum()), 3)
        self.assertEqual(int(result.perm_num.sum()), 1)
        self.assertEqual(result.n_positions, 1)
        self.assertEqual(result.num.dtype, torch.int64)
        self.assertEqual(result.den.dtype, torch.int64)

    def test_excluded_true_target_removes_entire_position(self):
        result = accumulate_document(
            "doc-a",
            torch.tensor([[3.0, 2.0, 1.0]]),
            torch.tensor([2]),
            torch.tensor([0, 1, 10], dtype=torch.int64),
            grid=GridSpec.default(),
            excluded_token_ids={2},
            permutation_seed=17,
        )

        self.assertEqual(result.n_positions, 0)
        self.assertEqual(int(result.den.sum()), 0)
        self.assertEqual(int(result.num.sum()), 0)

    def test_invalid_logits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulate_document(
                "doc-a",
                torch.tensor([[float("nan"), 0.0]]),
                torch.tensor([0]),
                torch.tensor([0, 1], dtype=torch.int64),
                grid=GridSpec.default(),
                excluded_token_ids=set(),
                permutation_seed=17,
            )

    def test_document_half_and_frequency_permutation_are_deterministic(self):
        groups = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

        self.assertEqual(document_half("doc-a", seed=17), document_half("doc-a", seed=17))
        self.assertTrue(
            torch.equal(
                permuted_frequency_groups(groups, seed=17),
                permuted_frequency_groups(groups, seed=17),
            )
        )
        self.assertFalse(
            torch.equal(
                permuted_frequency_groups(groups, seed=17),
                permuted_frequency_groups(groups, seed=18),
            )
        )

    def test_merge_is_additive_and_preserves_halves(self):
        shape = (GridSpec.default().n_margin_bins, 9)
        first = DocumentGridStats(
            "a", 0, torch.ones(shape, dtype=torch.int64),
            torch.full(shape, 2, dtype=torch.int64),
            torch.ones(shape, dtype=torch.int64),
            torch.full(shape, 2, dtype=torch.int64), 3,
        )
        second = DocumentGridStats(
            "b", 1, torch.full(shape, 3, dtype=torch.int64),
            torch.full(shape, 4, dtype=torch.int64),
            torch.full(shape, 3, dtype=torch.int64),
            torch.full(shape, 4, dtype=torch.int64), 5,
        )

        merged = merge_document_stats((first, second))

        self.assertTrue(torch.equal(merged["num"], torch.full(shape, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(merged["half_num"][0], first.num))
        self.assertTrue(torch.equal(merged["half_num"][1], second.num))
        self.assertEqual(merged["n_positions"], 8)
        self.assertEqual(merged["n_documents"], 2)

    def synthetic_grid(self, shift, *, crossing=False):
        grid = GridSpec.default(min_true_count=5)
        centers = np.asarray(grid.margin_bin_centers())
        den = torch.zeros((grid.n_margin_bins, 9), dtype=torch.int64)
        num = torch.zeros_like(den)
        active = (centers >= 2.0) & (centers <= 12.0)
        for group in (1, 8):
            den[active, group] = 1_000_000
            effective_shift = 0.0 if group == 8 else shift
            if crossing and group == 1:
                effective_shift = np.where(centers[active] < 7.0, 0.5, -0.5)
            log_rate = -2.0 - 0.5 * (centers[active] - effective_shift)
            values = np.maximum(5, np.rint(np.exp(log_rate) * 1_000_000)).astype(int)
            num[active, group] = torch.tensor(values, dtype=torch.int64)
        return grid, num, den

    def test_aligned_offset_recovers_positive_and_negative_half_nat(self):
        for shift in (-0.5, 0.0, 0.5):
            with self.subTest(shift=shift):
                grid, num, den = self.synthetic_grid(shift)
                analysis = analyze_grid(num, den, grid=grid, reference_group=8)
                self.assertAlmostEqual(analysis["shifts"]["1"], shift, delta=0.10)

    def test_crossing_curves_raise_non_additive_flag(self):
        grid, num, den = self.synthetic_grid(0.0, crossing=True)

        analysis = analyze_grid(num, den, grid=grid, reference_group=8)

        self.assertTrue(analysis["non_additive"])


if __name__ == "__main__":
    unittest.main()
