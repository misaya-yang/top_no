import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from splits import (  # noqa: E402
    ManifestDocument,
    pooled_positions,
    select_guarantee_position,
)


class DocumentSamplingTests(unittest.TestCase):
    def test_guarantee_position_is_deterministic_eligible_and_in_range(self):
        token_ids = tuple(range(40))
        document = ManifestDocument("doc-a", "0" * 64, "cluster-a")

        first = select_guarantee_position(
            document,
            token_ids,
            salt="cal-salt",
            excluded_target_ids={22, 23},
        )
        second = select_guarantee_position(
            document,
            token_ids,
            salt="cal-salt",
            excluded_target_ids={22, 23},
        )

        self.assertEqual(first, second)
        self.assertEqual(first.evidence_grade, "G")
        self.assertGreaterEqual(first.target_index, 16)
        self.assertLess(first.target_index, len(token_ids))
        self.assertNotIn(token_ids[first.target_index], {22, 23})

    def test_guarantee_position_is_approximately_uniform_across_document_ids(self):
        token_ids = tuple(range(20))
        counts = Counter(
            select_guarantee_position(
                ManifestDocument(f"doc-{index}", "0" * 64, f"cluster-{index}"),
                token_ids,
                salt="salt",
            ).target_index
            for index in range(4000)
        )

        self.assertEqual(set(counts), {16, 17, 18, 19})
        self.assertLess(max(counts.values()) - min(counts.values()), 180)

    def test_empty_eligible_position_set_fails(self):
        with self.assertRaisesRegex(ValueError, "no eligible target position"):
            select_guarantee_position(
                ManifestDocument("short", "0" * 64, "cluster-short"),
                tuple(range(16)),
                salt="salt",
            )
        with self.assertRaisesRegex(ValueError, "no eligible target position"):
            select_guarantee_position(
                ManifestDocument("excluded", "0" * 64, "cluster-excluded"),
                tuple(range(18)),
                salt="salt",
                excluded_target_ids={16, 17},
            )

    def test_pooled_positions_use_fixed_stride_and_exclusions(self):
        token_ids = tuple(range(35))
        document = ManifestDocument("doc", "0" * 64, "cluster")

        positions = pooled_positions(
            document,
            token_ids,
            stride=4,
            excluded_target_ids={20, 28},
        )

        self.assertEqual(tuple(item.target_index for item in positions), (16, 24, 32))
        self.assertEqual({item.evidence_grade for item in positions}, {"E"})
        self.assertEqual({item.doc_id for item in positions}, {"doc"})
        self.assertEqual({item.cluster_id for item in positions}, {"cluster"})

    def test_token_ids_must_be_integers(self):
        document = ManifestDocument("doc", "0" * 64, "cluster")

        with self.assertRaisesRegex(ValueError, "token_ids must contain integers"):
            select_guarantee_position(
                document,
                tuple(range(16)) + ("bad",),
                salt="salt",
            )


if __name__ == "__main__":
    unittest.main()
