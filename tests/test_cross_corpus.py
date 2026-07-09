import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import document_store  # noqa: E402
from cross_corpus import (  # noqa: E402
    CrossCorpusAudit,
    audit_cross_corpus,
    load_cross_corpus_audit,
    save_cross_corpus_audit,
    validate_cross_corpus_audit,
)
from splits import (  # noqa: E402
    DocumentManifest,
    ManifestDocument,
    SourceDocument,
    build_split_artifacts,
    content_sha256,
    save_manifest,
    save_split_artifacts,
)


class CrossCorpusTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.frequency_documents = tuple(
            SourceDocument(
                f"freq-{index}",
                " ".join(f"frequency{index}-token{j}" for j in range(32)),
            )
            for index in range(4)
        )
        self.evaluation_documents = tuple(
            SourceDocument(
                f"eval-{index}",
                " ".join(f"evaluation{index}-token{j}" for j in range(32)),
            )
            for index in range(20)
        )
        self.frequency_manifest_path = self.root / "frequency-manifest.json"
        self.frequency_jsonl = self.root / "frequency.jsonl"
        self.evaluation_jsonl = self.root / "evaluation.jsonl"
        self.write_frequency(self.frequency_documents)
        self.write_jsonl(self.evaluation_jsonl, self.evaluation_documents)
        result = build_split_artifacts(
            self.evaluation_documents,
            source="eval-fixture",
            source_snapshot_sha256=hashlib.sha256(b"eval-snapshot").hexdigest(),
            global_salt="global-salt",
        )
        self.eval_receipt_path = save_split_artifacts(result, self.root / "eval-split")

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def write_jsonl(path, documents):
        path.write_text(
            "\n".join(
                json.dumps({"doc_id": item.doc_id, "text": item.text})
                for item in documents
            )
            + "\n"
        )

    def write_frequency(self, documents):
        manifest = DocumentManifest(
            protocol_version="icml2027-pr1a",
            role="freq",
            source="frequency-fixture",
            documents=tuple(
                ManifestDocument(
                    item.doc_id,
                    content_sha256(item.text),
                    hashlib.sha256(f"cluster:{item.doc_id}".encode()).hexdigest(),
                )
                for item in documents
            ),
        )
        save_manifest(manifest, self.frequency_manifest_path)
        self.write_jsonl(self.frequency_jsonl, documents)

    def audit(self):
        return audit_cross_corpus(
            frequency_manifest_path=self.frequency_manifest_path,
            frequency_document_jsonl=self.frequency_jsonl,
            evaluation_split_receipt_path=self.eval_receipt_path,
            evaluation_document_jsonl=self.evaluation_jsonl,
        )

    def test_disjoint_corpora_emit_round_trip_pass_receipt(self):
        audit = self.audit()
        self.assertEqual(audit.receipt.verdict, "pass")
        self.assertEqual(audit.receipt.match_count, 0)
        self.assertEqual(audit.matches, ())

        path = save_cross_corpus_audit(audit, self.root / "cross-corpus.json")
        loaded = load_cross_corpus_audit(path)
        self.assertEqual(loaded, audit)
        self.assertEqual(
            validate_cross_corpus_audit(
                path,
                frequency_manifest_path=self.frequency_manifest_path,
                frequency_document_jsonl=self.frequency_jsonl,
                evaluation_split_receipt_path=self.eval_receipt_path,
                evaluation_document_jsonl=self.evaluation_jsonl,
            ),
            audit.receipt,
        )

    def test_jsonl_row_order_does_not_change_audit(self):
        expected = self.audit()
        self.write_jsonl(
            self.frequency_jsonl, tuple(reversed(self.frequency_documents))
        )
        self.write_jsonl(
            self.evaluation_jsonl, tuple(reversed(self.evaluation_documents))
        )

        self.assertEqual(self.audit(), expected)

    def test_audit_loads_the_split_snapshot_once(self):
        with patch.object(
            document_store,
            "load_split_receipt",
            wraps=document_store.load_split_receipt,
        ) as mocked_load:
            self.audit()

        mocked_load.assert_called_once()

    def test_exact_cross_corpus_duplicate_is_recorded_as_failure(self):
        duplicated = list(self.frequency_documents)
        duplicated[0] = SourceDocument("freq-overlap", self.evaluation_documents[0].text)
        self.write_frequency(duplicated)

        audit = self.audit()

        self.assertEqual(audit.receipt.verdict, "fail")
        self.assertEqual(audit.receipt.match_count, 1)
        self.assertEqual(audit.matches[0].frequency_doc_id, "freq-overlap")
        self.assertEqual(audit.matches[0].evaluation_doc_id, "eval-0")

    def test_exact_threshold_pair_cannot_be_missed_by_lsh(self):
        base = " ".join(f"x1078_{index}" for index in range(16))
        self.write_frequency((SourceDocument("freq-threshold", base),))
        evaluation = list(self.evaluation_documents)
        evaluation[0] = SourceDocument("eval-threshold", base + " x1078_16")
        self.write_jsonl(self.evaluation_jsonl, evaluation)
        result = build_split_artifacts(
            evaluation,
            source="eval-fixture",
            source_snapshot_sha256=hashlib.sha256(b"eval-snapshot-2").hexdigest(),
            global_salt="global-salt",
        )
        self.eval_receipt_path = save_split_artifacts(result, self.root / "threshold-split")

        audit = self.audit()

        self.assertEqual(audit.receipt.match_count, 1)
        self.assertEqual(audit.matches[0].intersection_size, 4)
        self.assertEqual(audit.matches[0].union_size, 5)

    def test_pair_below_exact_threshold_passes(self):
        base = " ".join(f"below-threshold-{index}" for index in range(16))
        self.write_frequency((SourceDocument("freq-below", base),))
        evaluation = list(self.evaluation_documents)
        evaluation[0] = SourceDocument(
            "eval-below", base + " below-threshold-16 below-threshold-17"
        )
        self.write_jsonl(self.evaluation_jsonl, evaluation)
        result = build_split_artifacts(
            evaluation,
            source="eval-fixture",
            source_snapshot_sha256=hashlib.sha256(b"below-snapshot").hexdigest(),
            global_salt="global-salt",
        )
        self.eval_receipt_path = save_split_artifacts(result, self.root / "below-split")

        self.assertEqual(self.audit().receipt.verdict, "pass")

    def test_normalization_equivalent_cross_corpus_text_is_rejected(self):
        normalized = " ".join(f"alpha{index}" for index in range(13))
        width_variant = normalized.replace("a", "Ａ").upper()
        self.write_frequency((SourceDocument("freq-normalized", width_variant),))
        evaluation = list(self.evaluation_documents)
        evaluation[0] = SourceDocument("eval-normalized", normalized)
        self.write_jsonl(self.evaluation_jsonl, evaluation)
        result = build_split_artifacts(
            evaluation,
            source="eval-fixture",
            source_snapshot_sha256=hashlib.sha256(b"normalized-snapshot").hexdigest(),
            global_salt="global-salt",
        )
        self.eval_receipt_path = save_split_artifacts(
            result, self.root / "normalized-split"
        )

        audit = self.audit()

        self.assertEqual(audit.receipt.match_count, 1)
        self.assertEqual(audit.matches[0].intersection_size, 1)
        self.assertEqual(audit.matches[0].union_size, 1)

    def test_scope_includes_discarded_members_of_evaluation_clusters(self):
        base_tokens = [f"chain-{index}" for index in range(32)]
        left_text = " ".join(base_tokens)
        middle_core = " ".join(base_tokens + [f"chain-{index}" for index in range(32, 36)])
        right_text = " ".join(base_tokens + [f"chain-{index}" for index in range(32, 40)])
        endpoint_min_hash = min(content_sha256(left_text), content_sha256(right_text))
        middle_text = None
        for spaces in range(1, 1000):
            candidate = " " * spaces + middle_core
            if content_sha256(candidate) > endpoint_min_hash:
                middle_text = candidate
                break
        self.assertIsNotNone(middle_text)
        chain = (
            SourceDocument("chain-left", left_text),
            SourceDocument("chain-middle", middle_text),
            SourceDocument("chain-right", right_text),
        )
        evaluation = chain + self.evaluation_documents
        self.write_jsonl(self.evaluation_jsonl, evaluation)
        result = build_split_artifacts(
            evaluation,
            source="eval-fixture",
            source_snapshot_sha256=hashlib.sha256(b"scope-snapshot").hexdigest(),
            global_salt="global-salt",
        )
        cluster = next(
            item
            for item in result.clusters
            if {member.doc_id for member in item.members}
            == {"chain-left", "chain-middle", "chain-right"}
        )
        self.assertNotEqual(cluster.representative.doc_id, "chain-middle")
        discarded_endpoint = (
            chain[2]
            if cluster.representative.doc_id == "chain-left"
            else chain[0]
        )
        self.write_frequency(
            (SourceDocument("freq-discarded-member", discarded_endpoint.text),)
        )
        self.eval_receipt_path = save_split_artifacts(result, self.root / "scope-split")

        audit = self.audit()

        self.assertEqual(
            audit.receipt.comparison_scope,
            "frequency-manifest-docs-vs-evaluation-input-documents-v1",
        )
        self.assertEqual(audit.receipt.verdict, "fail")
        self.assertGreaterEqual(audit.receipt.match_count, 1)
        self.assertIn(
            discarded_endpoint.doc_id,
            {match.evaluation_doc_id for match in audit.matches},
        )

    def test_frequency_jsonl_must_exactly_match_frequency_manifest(self):
        changed = list(self.frequency_documents)
        changed[0] = SourceDocument(changed[0].doc_id, changed[0].text + " changed")
        self.write_jsonl(self.frequency_jsonl, changed)

        with self.assertRaisesRegex(ValueError, "frequency manifest.*content_sha256"):
            self.audit()

    def test_frequency_jsonl_rejects_missing_and_extra_rows(self):
        for documents in (
            self.frequency_documents[:-1],
            self.frequency_documents
            + (SourceDocument("freq-extra", "extra row content"),),
        ):
            with self.subTest(count=len(documents)):
                self.write_jsonl(self.frequency_jsonl, documents)
                with self.assertRaisesRegex(ValueError, "IDs mismatch"):
                    self.audit()
        self.write_jsonl(self.frequency_jsonl, self.frequency_documents)

    def test_frequency_content_error_does_not_expose_text(self):
        secret = "DO-NOT-PRINT-THIS-CONTENT"
        changed = list(self.frequency_documents)
        changed[0] = SourceDocument(changed[0].doc_id, secret)
        self.write_jsonl(self.frequency_jsonl, changed)

        with self.assertRaises(ValueError) as raised:
            self.audit()

        self.assertNotIn(secret, str(raised.exception))

    def test_tampered_saved_audit_is_rejected(self):
        path = save_cross_corpus_audit(self.audit(), self.root / "cross-corpus.json")
        payload = json.loads(path.read_text())
        payload["receipt"]["verdict"] = "fail"
        path.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "verdict|artifact_id"):
            load_cross_corpus_audit(path)

    def test_forged_zero_match_receipt_fails_recomputed_scan(self):
        duplicated = list(self.frequency_documents)
        duplicated[0] = SourceDocument("freq-overlap", self.evaluation_documents[0].text)
        self.write_frequency(duplicated)
        failed = self.audit()
        empty_matches_hash = hashlib.sha256(b"[]").hexdigest()
        forged = CrossCorpusAudit(
            receipt=replace(
                failed.receipt,
                matches_sha256=empty_matches_hash,
                match_count=0,
                verdict="pass",
            ),
            matches=(),
        )
        path = save_cross_corpus_audit(forged, self.root / "forged.json")

        with self.assertRaisesRegex(ValueError, "recomputed cross-corpus audit"):
            validate_cross_corpus_audit(
                path,
                frequency_manifest_path=self.frequency_manifest_path,
                frequency_document_jsonl=self.frequency_jsonl,
                evaluation_split_receipt_path=self.eval_receipt_path,
                evaluation_document_jsonl=self.evaluation_jsonl,
            )

    def test_weakened_split_threshold_is_rejected(self):
        weakened = build_split_artifacts(
            self.evaluation_documents,
            source="eval-fixture",
            source_snapshot_sha256=hashlib.sha256(b"weak-snapshot").hexdigest(),
            global_salt="global-salt",
            jaccard_threshold=0.9,
        )
        self.eval_receipt_path = save_split_artifacts(weakened, self.root / "weak-split")

        with self.assertRaisesRegex(ValueError, "jaccard_threshold=0.8"):
            self.audit()

    def test_cli_writes_only_a_passing_receipt(self):
        output = self.root / "cli-receipt.json"
        command = [
            sys.executable,
            str(ROOT / "experiments" / "prepare_cross_corpus_receipt.py"),
            "--frequency-manifest",
            str(self.frequency_manifest_path),
            "--frequency-document-jsonl",
            str(self.frequency_jsonl),
            "--evaluation-split-receipt",
            str(self.eval_receipt_path),
            "--evaluation-document-jsonl",
            str(self.evaluation_jsonl),
            "--output",
            str(output),
        ]

        completed = subprocess.run(command, capture_output=True, text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(load_cross_corpus_audit(output).receipt.verdict, "pass")

        duplicated = list(self.frequency_documents)
        duplicated[0] = SourceDocument(
            "freq-overlap", self.evaluation_documents[0].text
        )
        self.write_frequency(duplicated)
        output.unlink()
        completed = subprocess.run(command, capture_output=True, text=True)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('"match_count": 1', completed.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
