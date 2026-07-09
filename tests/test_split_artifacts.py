import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from splits import (  # noqa: E402
    DocumentManifest,
    SourceDocument,
    SplitBuildArtifacts,
    build_split_artifacts,
    load_split_receipt,
    manifest_sha256,
    save_manifest,
    save_split_artifacts,
    split_receipt_sha256,
)


class SplitArtifactTests(unittest.TestCase):
    def build(self):
        documents = [
            SourceDocument(
                f"doc-{index}",
                " ".join(f"token{index}-{j}" for j in range(48)),
            )
            for index in range(12)
        ]
        return build_split_artifacts(
            documents,
            source="fixture",
            source_snapshot_sha256=hashlib.sha256(b"snapshot").hexdigest(),
            global_salt="global-salt",
        )

    def test_receipt_round_trip_binds_all_manifest_hashes(self):
        result = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = save_split_artifacts(result, Path(tmp))
            receipt, manifests = load_split_receipt(receipt_path)

        self.assertEqual(receipt, result.receipt)
        self.assertEqual(split_receipt_sha256(receipt), split_receipt_sha256(result.receipt))
        self.assertEqual(
            dict(receipt.manifest_sha256s),
            {role: manifest_sha256(manifest) for role, manifest in manifests.items()},
        )

    def test_tampered_receipt_fails(self):
        result = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = save_split_artifacts(result, Path(tmp))
            payload = json.loads(receipt_path.read_text())
            payload["receipt"]["global_salt"] = "tampered"
            receipt_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "receipt_sha256"):
                load_split_receipt(receipt_path)

    def test_tampered_manifest_fails_receipt_binding(self):
        result = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = save_split_artifacts(result, Path(tmp))
            payload = json.loads(receipt_path.read_text())
            manifest_path = receipt_path.parent / payload["manifest_files"]["tune"]
            manifest_payload = json.loads(manifest_path.read_text())
            manifest_payload["manifest"]["source"] = "tampered"
            manifest_path.write_text(json.dumps(manifest_payload))

            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                load_split_receipt(receipt_path)

    def test_unsafe_manifest_path_fails(self):
        result = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = save_split_artifacts(result, Path(tmp))
            payload = json.loads(receipt_path.read_text())
            payload["manifest_files"]["tune"] = "../tune.json"
            receipt_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
                load_split_receipt(receipt_path)

    def test_receipt_source_must_match_role_manifests(self):
        result = self.build()
        mismatched = SplitBuildArtifacts(
            clusters=result.clusters,
            manifests=result.manifests,
            receipt=replace(result.receipt, source="other-source"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                save_split_artifacts(mismatched, Path(tmp))

    def test_receipt_manifest_hash_order_is_canonical(self):
        result = self.build()
        reordered = replace(
            result.receipt,
            manifest_sha256s=tuple(reversed(result.receipt.manifest_sha256s)),
        )

        with self.assertRaisesRegex(ValueError, "canonical role order"):
            split_receipt_sha256(reordered)

    def moved_to_wrong_role(self, result):
        source_role = next(
            role for role, manifest in result.manifests.items() if manifest.documents
        )
        target_role = next(role for role in ("tune", "cal", "test") if role != source_role)
        moved = result.manifests[source_role].documents[0]
        manifests = dict(result.manifests)
        manifests[source_role] = replace(
            manifests[source_role],
            documents=manifests[source_role].documents[1:],
        )
        manifests[target_role] = replace(
            manifests[target_role],
            documents=manifests[target_role].documents + (moved,),
        )
        receipt = replace(
            result.receipt,
            manifest_sha256s=tuple(
                (role, manifest_sha256(manifests[role]))
                for role in ("tune", "cal", "test")
            ),
        )
        return SplitBuildArtifacts(result.clusters, manifests, receipt)

    def test_save_recomputes_cluster_role_from_salt(self):
        tampered = self.moved_to_wrong_role(self.build())

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "split role mismatch"):
                save_split_artifacts(tampered, Path(tmp))

    def test_load_recomputes_cluster_role_from_salt(self):
        result = self.build()
        tampered = self.moved_to_wrong_role(result)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_path = save_split_artifacts(result, root)
            for role, manifest in tampered.manifests.items():
                save_manifest(manifest, root / f"{role}_manifest.json")
            wrapper = json.loads(receipt_path.read_text())
            wrapper["receipt"] = asdict(tampered.receipt)
            wrapper["receipt_sha256"] = split_receipt_sha256(tampered.receipt)
            receipt_path.write_text(json.dumps(wrapper))

            with self.assertRaisesRegex(ValueError, "split role mismatch"):
                load_split_receipt(receipt_path)

    def test_manifest_symlink_cannot_escape_artifact_directory(self):
        result = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            receipt_path = save_split_artifacts(result, artifact_dir)
            manifest_path = artifact_dir / "tune_manifest.json"
            outside_path = root / "outside.json"
            outside_path.write_bytes(manifest_path.read_bytes())
            manifest_path.unlink()
            manifest_path.symlink_to(outside_path)

            with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
                load_split_receipt(receipt_path)

    def test_jsonl_cli_builds_loadable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "documents.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "doc_id": f"doc-{index}",
                            "text": " ".join(
                                f"token{index}-{position}" for position in range(48)
                            ),
                        }
                    )
                    for index in range(8)
                )
                + "\n"
            )
            output_dir = root / "artifacts"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "prepare_document_splits.py"),
                    "--input-jsonl",
                    str(input_path),
                    "--source",
                    "fixture",
                    "--source-snapshot-sha256",
                    hashlib.sha256(b"snapshot").hexdigest(),
                    "--global-salt",
                    "global-salt",
                    "--output-dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt, manifests = load_split_receipt(output_dir / "split_receipt.json")

        self.assertEqual(receipt.num_input_documents, 8)
        self.assertEqual(sum(len(value.documents) for value in manifests.values()), 8)


if __name__ == "__main__":
    unittest.main()
