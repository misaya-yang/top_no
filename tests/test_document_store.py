import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from document_store import bind_split_documents  # noqa: E402
from splits import (  # noqa: E402
    DocumentManifest,
    SourceDocument,
    build_split_artifacts,
    save_manifest,
    save_split_artifacts,
)


class DocumentStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.documents = tuple(
            SourceDocument(
                f"doc-{index}",
                " ".join(f"topic{index}-token{position}" for position in range(32)),
            )
            for index in range(24)
        )
        self.result = build_split_artifacts(
            self.documents,
            source="fixture",
            source_snapshot_sha256=hashlib.sha256(b"snapshot").hexdigest(),
            global_salt="global-salt",
        )
        self.receipt_path = save_split_artifacts(self.result, self.root / "splits")
        self.document_path = self.root / "documents.jsonl"
        self.write_documents(self.documents)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_documents(self, documents):
        self.document_path.write_text(
            "\n".join(
                json.dumps({"doc_id": item.doc_id, "text": item.text})
                for item in documents
            )
            + "\n"
        )

    def configured_manifests(self):
        return {
            role: self.receipt_path.parent / f"{role}_manifest.json"
            for role in ("tune", "cal", "test")
        }

    def test_bind_returns_every_representative_with_role_and_exact_text(self):
        bound = bind_split_documents(
            self.receipt_path,
            self.document_path,
            configured_manifests=self.configured_manifests(),
        )

        rows = [item for _, documents in bound.documents_by_role for item in documents]
        expected_text = {item.doc_id: item.text for item in self.documents}
        self.assertEqual(len(rows), self.result.receipt.num_clusters)
        for row in rows:
            self.assertEqual(row.text, expected_text[row.doc_id])
            self.assertEqual(row.role, next(
                role
                for role, manifest in self.result.manifests.items()
                if row.doc_id in {item.doc_id for item in manifest.documents}
            ))

    def test_jsonl_row_order_does_not_change_binding(self):
        first = bind_split_documents(self.receipt_path, self.document_path)
        self.write_documents(tuple(reversed(self.documents)))
        second = bind_split_documents(self.receipt_path, self.document_path)

        self.assertEqual(first, second)

    def test_modified_text_fails_input_identity(self):
        modified = list(self.documents)
        modified[0] = SourceDocument(modified[0].doc_id, modified[0].text + " changed")
        self.write_documents(modified)

        with self.assertRaisesRegex(ValueError, "input_documents_sha256"):
            bind_split_documents(self.receipt_path, self.document_path)

    def test_missing_or_extra_document_fails_input_identity(self):
        with self.subTest("missing"):
            self.write_documents(self.documents[:-1])
            with self.assertRaisesRegex(ValueError, "input_documents_sha256"):
                bind_split_documents(self.receipt_path, self.document_path)

        with self.subTest("extra"):
            self.write_documents(
                self.documents
                + (SourceDocument("extra", " ".join(f"extra-{i}" for i in range(20))),)
            )
            with self.assertRaisesRegex(ValueError, "input_documents_sha256"):
                bind_split_documents(self.receipt_path, self.document_path)

    def test_duplicate_document_id_fails_without_printing_text(self):
        duplicate = self.documents + (self.documents[0],)
        self.write_documents(duplicate)

        with self.assertRaisesRegex(ValueError, "duplicate doc_id") as raised:
            bind_split_documents(self.receipt_path, self.document_path)
        self.assertNotIn(self.documents[0].text, str(raised.exception))

    def test_configured_manifest_hash_must_match_receipt(self):
        configured = self.configured_manifests()
        tune = self.result.manifests["tune"]
        wrong = DocumentManifest(
            protocol_version=tune.protocol_version,
            role=tune.role,
            source=tune.source,
            documents=tune.documents[:-1],
        )
        wrong_path = self.root / "wrong-tune.json"
        save_manifest(wrong, wrong_path)
        configured["tune"] = wrong_path

        with self.assertRaisesRegex(ValueError, "configured manifest hash mismatch"):
            bind_split_documents(
                self.receipt_path,
                self.document_path,
                configured_manifests=configured,
            )

    def test_blank_malformed_and_extra_field_rows_fail_closed(self):
        invalid_rows = {
            "blank": "\n",
            "malformed": "{not-json}\n",
            "extra": json.dumps({"doc_id": "doc", "text": "text", "extra": 1}) + "\n",
        }
        for label, payload in invalid_rows.items():
            with self.subTest(label):
                self.document_path.write_text(payload)
                with self.assertRaises(ValueError):
                    bind_split_documents(self.receipt_path, self.document_path)


if __name__ == "__main__":
    unittest.main()
