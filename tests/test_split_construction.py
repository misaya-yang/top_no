import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from splits import (  # noqa: E402
    SourceDocument,
    build_split_artifacts,
    manifest_sha256,
    split_receipt_sha256,
    split_role_for_cluster,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def words(prefix: str, count: int = 48) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


class SplitConstructionTests(unittest.TestCase):
    def build(self, documents, salt="global-salt"):
        return build_split_artifacts(
            documents,
            source="fixture-corpus",
            source_snapshot_sha256=digest("fixture-snapshot"),
            global_salt=salt,
        )

    def test_input_order_does_not_change_clusters_manifests_or_receipt(self):
        documents = [
            SourceDocument("doc-a", words("a")),
            SourceDocument("doc-b", words("b")),
            SourceDocument("doc-c", words("c")),
        ]

        left = self.build(documents)
        right = self.build(list(reversed(documents)))

        self.assertEqual(left.clusters, right.clusters)
        self.assertEqual(
            {role: manifest_sha256(value) for role, value in left.manifests.items()},
            {role: manifest_sha256(value) for role, value in right.manifests.items()},
        )
        self.assertEqual(
            split_receipt_sha256(left.receipt),
            split_receipt_sha256(right.receipt),
        )

    def test_exact_and_near_duplicates_share_one_cluster_representative(self):
        base = words("token")
        documents = [
            SourceDocument("doc-exact-b", base),
            SourceDocument("doc-exact-a", base),
            SourceDocument("doc-near", base + " trailing1 trailing2"),
        ]

        result = self.build(documents)

        self.assertEqual(len(result.clusters), 1)
        cluster = result.clusters[0]
        self.assertEqual(cluster.member_doc_ids, ("doc-exact-a", "doc-exact-b", "doc-near"))
        expected_representative = min(
            documents,
            key=lambda item: (hashlib.sha256(item.text.encode()).hexdigest(), item.doc_id),
        ).doc_id
        self.assertEqual(cluster.representative_doc_id, expected_representative)
        self.assertEqual(sum(len(m.documents) for m in result.manifests.values()), 1)

    def test_unrelated_documents_remain_separate(self):
        result = self.build(
            [
                SourceDocument("doc-a", words("alpha")),
                SourceDocument("doc-b", words("omega")),
            ]
        )

        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(sum(len(m.documents) for m in result.manifests.values()), 2)

    def test_exact_jaccard_threshold_is_inclusive_and_lower_pair_is_rejected(self):
        base = " ".join(f"t{index}" for index in range(16))
        at_threshold = base + " t16"
        below_threshold = base + " t16 t17"

        merged = self.build(
            [SourceDocument("base", base), SourceDocument("threshold", at_threshold)]
        )
        separated = self.build(
            [SourceDocument("base", base), SourceDocument("below", below_threshold)]
        )

        self.assertEqual(len(merged.clusters), 1)
        self.assertEqual(len(separated.clusters), 2)

    def test_near_duplicate_edges_form_transitive_components(self):
        base = words("chain", 50)
        middle = base + " " + words("middle-extra", 5)
        end = middle + " " + words("end-extra", 5)

        result = self.build(
            [
                SourceDocument("base", base),
                SourceDocument("middle", middle),
                SourceDocument("end", end),
            ]
        )

        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].member_doc_ids, ("base", "end", "middle"))

    def test_split_role_uses_fixed_hash_bands(self):
        assignments = [
            split_role_for_cluster(f"cluster-{index}", "salt")
            for index in range(10_000)
        ]
        roles = set(assignments)

        self.assertEqual(roles, {"tune", "cal", "test"})
        counts = {role: assignments.count(role) for role in roles}
        self.assertLess(abs(counts["tune"] / 10_000 - 0.40), 0.02)
        self.assertLess(abs(counts["cal"] / 10_000 - 0.25), 0.02)
        self.assertLess(abs(counts["test"] / 10_000 - 0.35), 0.02)
        self.assertEqual(split_role_for_cluster("cluster-0", "salt"), "tune")
        self.assertEqual(split_role_for_cluster("cluster-19", "salt"), "cal")
        self.assertEqual(split_role_for_cluster("cluster-2", "salt"), "test")

    def test_salt_is_bound_even_if_assignments_happen_to_match(self):
        document = SourceDocument("doc", words("same"))
        first = self.build([document], salt="salt-a")
        second = self.build([document], salt="salt-b")

        self.assertNotEqual(
            split_receipt_sha256(first.receipt),
            split_receipt_sha256(second.receipt),
        )
        self.assertEqual(first.receipt.global_salt, "salt-a")
        self.assertEqual(second.receipt.global_salt, "salt-b")

    def test_minhash_parameters_are_bound_into_receipt(self):
        document = SourceDocument("doc", words("same"))
        baseline = self.build([document])
        changed_seed = build_split_artifacts(
            [document],
            source="fixture-corpus",
            source_snapshot_sha256=digest("fixture-snapshot"),
            global_salt="global-salt",
            minhash_seed=1730,
        )
        changed_layout = build_split_artifacts(
            [document],
            source="fixture-corpus",
            source_snapshot_sha256=digest("fixture-snapshot"),
            global_salt="global-salt",
            num_perm=120,
            lsh_bands=20,
        )

        identities = {
            split_receipt_sha256(result.receipt)
            for result in (baseline, changed_seed, changed_layout)
        }
        self.assertEqual(len(identities), 3)

    def test_duplicate_doc_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
            self.build(
                [
                    SourceDocument("same", words("a")),
                    SourceDocument("same", words("b")),
                ]
            )

    def test_documents_shorter_than_one_full_shingle_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at least shingle_size"):
            self.build([SourceDocument("short", "one two three")])


if __name__ == "__main__":
    unittest.main()
