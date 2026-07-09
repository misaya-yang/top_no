import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from splits import (  # noqa: E402
    DocumentManifest,
    ManifestDocument,
    assert_pairwise_disjoint,
    load_manifest,
    manifest_sha256,
    save_manifest,
)


def manifest(role: str, suffix: str) -> DocumentManifest:
    return DocumentManifest(
        protocol_version="icml2027-pr1a",
        role=role,
        source="fixture",
        documents=(
            ManifestDocument(
                doc_id=f"doc-{suffix}",
                content_sha256=hashlib.sha256(f"content-{suffix}".encode()).hexdigest(),
                cluster_id=f"cluster-{suffix}",
            ),
        ),
    )


class ProtocolManifestTests(unittest.TestCase):
    def test_hash_is_independent_of_document_order(self):
        first = manifest("cal", "a").documents[0]
        second = manifest("cal", "b").documents[0]
        left = DocumentManifest("icml2027-pr1a", "cal", "fixture", (first, second))
        right = DocumentManifest("icml2027-pr1a", "cal", "fixture", (second, first))

        self.assertEqual(manifest_sha256(left), manifest_sha256(right))

    def test_round_trip_preserves_hash(self):
        original = manifest("freq", "freq")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_manifest(original, Path(tmp) / "freq.json")
            loaded = load_manifest(path)

        self.assertEqual(loaded, original)
        self.assertEqual(manifest_sha256(loaded), manifest_sha256(original))

    def test_tampered_wrapper_hash_fails(self):
        original = manifest("freq", "freq")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_manifest(original, Path(tmp) / "freq.json")
            payload = json.loads(path.read_text())
            payload["manifest_sha256"] = "tampered"
            path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                load_manifest(path)

    def test_duplicate_identifier_inside_manifest_fails(self):
        first = manifest("cal", "same").documents[0]
        duplicate = ManifestDocument(first.doc_id, "other-content", "other-cluster")
        invalid = DocumentManifest("icml2027-pr1a", "cal", "fixture", (first, duplicate))

        with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
            manifest_sha256(invalid)

    def test_disjoint_manifests_pass(self):
        assert_pairwise_disjoint(
            {
                "freq": manifest("freq", "f"),
                "tune": manifest("tune", "u"),
                "cal": manifest("cal", "c"),
                "test": manifest("test", "t"),
            }
        )

    def test_doc_id_intersection_fails(self):
        left = manifest("freq", "same")
        right_doc = ManifestDocument(
            "doc-same",
            hashlib.sha256(b"other-content").hexdigest(),
            "other-cluster",
        )
        right = DocumentManifest("icml2027-pr1a", "test", "fixture", (right_doc,))

        with self.assertRaisesRegex(ValueError, "doc_id"):
            assert_pairwise_disjoint({"freq": left, "test": right})

    def test_content_intersection_fails(self):
        left = manifest("freq", "same")
        right_doc = ManifestDocument(
            "other-doc",
            left.documents[0].content_sha256,
            "other-cluster",
        )
        right = DocumentManifest("icml2027-pr1a", "test", "fixture", (right_doc,))

        with self.assertRaisesRegex(ValueError, "content_sha256"):
            assert_pairwise_disjoint({"freq": left, "test": right})

    def test_cluster_intersection_fails(self):
        left = manifest("freq", "same")
        right_doc = ManifestDocument(
            "other-doc",
            hashlib.sha256(b"other-content").hexdigest(),
            "cluster-same",
        )
        right = DocumentManifest("icml2027-pr1a", "test", "fixture", (right_doc,))

        with self.assertRaisesRegex(ValueError, "cluster_id"):
            assert_pairwise_disjoint({"freq": left, "test": right})

    def test_noncanonical_content_hash_is_rejected(self):
        digest = manifest("freq", "same").documents[0].content_sha256
        invalid = DocumentManifest(
            "icml2027-pr1a",
            "test",
            "fixture",
            (ManifestDocument("doc", digest.upper(), "cluster"),),
        )

        with self.assertRaisesRegex(ValueError, "content_sha256.*lowercase"):
            manifest_sha256(invalid)


if __name__ == "__main__":
    unittest.main()
