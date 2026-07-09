import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from frequency_buckets import (  # noqa: E402
        DIAGNOSTIC_BUCKET_KIND,
        DIAGNOSTIC_BUCKET_LABELS,
        METHOD_BUCKET_KIND,
        load_method_bucket_policy,
        method_bucket_policy_sha256,
        diagnostic_frequency_groups,
    )


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class DiagnosticFrequencyBucketTests(unittest.TestCase):
    def test_exact_log10_boundaries_match_fable5_b0_through_b8(self):
        counts = torch.tensor(
            [0, 1, 9, 10, 99, 100, 999, 1_000, 9_999, 10_000,
             99_999, 100_000, 999_999, 1_000_000, 9_999_999, 10_000_000],
            dtype=torch.int64,
        )

        groups = diagnostic_frequency_groups(counts)

        self.assertEqual(
            groups.tolist(),
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        )
        self.assertEqual(DIAGNOSTIC_BUCKET_KIND, "diagnostic-log10-v1")
        self.assertEqual(
            METHOD_BUCKET_KIND,
            "method-true-token-mass-quantile-v1",
        )
        self.assertEqual(len(DIAGNOSTIC_BUCKET_LABELS), 9)

    def test_large_counts_stay_in_open_ended_b8(self):
        counts = torch.tensor([10_000_000, 10**12, 2**62], dtype=torch.int64)
        self.assertEqual(diagnostic_frequency_groups(counts).tolist(), [8, 8, 8])

        int32_counts = torch.tensor([0, 99, 10_000, 10_000_000], dtype=torch.int32)
        self.assertEqual(
            diagnostic_frequency_groups(int32_counts).tolist(),
            [0, 2, 5, 8],
        )

    def test_invalid_frequency_tables_fail_closed(self):
        invalid = (
            torch.tensor([], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([0.0, 1.0]),
            torch.tensor([False, True]),
            torch.tensor([99, 100], dtype=torch.uint8),
            torch.tensor([9_999, 10_000], dtype=torch.int16),
            torch.tensor([0, -1], dtype=torch.int64),
        )
        for counts in invalid:
            with self.subTest(shape=tuple(counts.shape), dtype=counts.dtype):
                with self.assertRaisesRegex(ValueError, "token_counts"):
                    diagnostic_frequency_groups(counts)

    def test_committed_method_policy_freezes_all_pre_registration_choices(self):
        policy = load_method_bucket_policy(
            ROOT / "configs" / "frequency_bucket_policy_v1.json"
        )

        self.assertEqual(policy.initial_bucket_count, 8)
        self.assertEqual(policy.delta_grid[-1], 0.01)
        self.assertEqual(policy.minimum_tune_targets_per_bucket, 500)
        self.assertEqual(len(method_bucket_policy_sha256(policy)), 64)

    def test_method_policy_rejects_relaxed_floor_or_unknown_fields(self):
        source = json.loads(
            (ROOT / "configs" / "frequency_bucket_policy_v1.json").read_text()
        )
        invalid = (
            {**source, "minimum_tune_targets_per_bucket": 100},
            {**source, "initial_bucket_count": 9},
            {
                **source,
                "floor_multiplier": 4,
                "minimum_tune_targets_per_bucket": 400,
            },
            {**source, "min_final_bucket_count": 3},
            {**source, "unexpected": True},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, payload in enumerate(invalid):
                with self.subTest(index=index):
                    path = Path(tmp) / f"policy-{index}.json"
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(
                        ValueError,
                        "policy|minimum|initial|floor|min_final",
                    ):
                        load_method_bucket_policy(path)


if __name__ == "__main__":
    unittest.main()
