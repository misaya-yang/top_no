import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from eval_prediction_sets import (  # noqa: E402
        assert_protocol_is_allowed,
        load_config,
        load_texts,
        main as prediction_main,
        resolve_token_counts,
    )
    from eval_openended_quality import (  # noqa: E402
        build_counts_for_strategies as build_openended_counts,
    )
    from eval_reasoning_self_consistency import (  # noqa: E402
        build_counts_for_strategies as build_reasoning_counts,
    )
    from freq_table import (  # noqa: E402
        counts_sha256,
        load_frequency_table_from_metrics,
        make_frequency_table_metadata,
        save_frequency_table,
    )
    from protocol import effective_config_sha256, validate_protocol_inputs  # noqa: E402
    from splits import (  # noqa: E402
        DocumentManifest,
        ManifestDocument,
        manifest_sha256,
        save_manifest,
    )


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class PredictionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_manifest(self, role, suffix):
        manifest = DocumentManifest(
            protocol_version="icml2027-pr1a",
            role=role,
            source="fixture",
            documents=(
                ManifestDocument(
                    doc_id=f"doc-{suffix}",
                    content_sha256=hashlib.sha256(
                        f"content-{suffix}".encode()
                    ).hexdigest(),
                    cluster_id=f"cluster-{suffix}",
                ),
            ),
        )
        return save_manifest(manifest, self.root / f"{role}.json"), manifest

    def write_frequency_table(self, frequency_manifest, source_hash=None):
        counts = torch.tensor([4, 0, 7], dtype=torch.int64)
        metadata = make_frequency_table_metadata(
            counts,
            model_id="fixture/model",
            tokenizer_id="fixture/model",
            tokenizer_revision="model-commit",
            source_manifest_sha256=source_hash or manifest_sha256(frequency_manifest),
            exclusion_token_ids=(1,),
            num_documents=1,
        )
        return save_frequency_table(counts, metadata, self.root / "frequency-table")

    def complete_config(self):
        frequency_path, frequency = self.write_manifest("freq", "freq")
        tune_path, _ = self.write_manifest("tune", "tune")
        calibration_path, _ = self.write_manifest("cal", "cal")
        test_path, _ = self.write_manifest("test", "test")
        table_path = self.write_frequency_table(frequency)
        return {
            "allow_legacy_protocol": False,
            "model_revision": "model-commit",
            "frequency_table": str(table_path),
            "frequency_manifest": str(frequency_path),
            "tune_manifest": str(tune_path),
            "calibration_manifest": str(calibration_path),
            "test_manifest": str(test_path),
        }

    def test_legacy_protocol_is_explicitly_non_citable(self):
        protocol = validate_protocol_inputs({"allow_legacy_protocol": True})

        self.assertEqual(protocol["protocol_version"], "legacy-pre-pr1")
        self.assertEqual(protocol["evidence_grade"], "legacy-smoke")
        self.assertFalse(protocol["paper_grade"])
        self.assertIsNone(protocol["frequency_table"])

    def test_evaluator_delegates_to_fail_closed_protocol_validation(self):
        protocol = assert_protocol_is_allowed({"allow_legacy_protocol": True})

        self.assertEqual(protocol["evidence_grade"], "legacy-smoke")
        with self.assertRaisesRegex(RuntimeError, "frequency_table"):
            assert_protocol_is_allowed({"allow_legacy_protocol": False})

    def test_evaluator_blocks_before_model_allocation(self):
        with patch("sys.argv", ["eval_prediction_sets.py"]), patch(
            "eval_prediction_sets.load_model_and_tokenizer"
        ) as mocked_load:
            with self.assertRaisesRegex(RuntimeError, "frequency_table"):
                prediction_main()

        mocked_load.assert_not_called()

    def test_evaluator_config_declares_protocol_artifact_paths(self):
        config = load_config(None)

        self.assertIsNone(config["model_revision"])
        self.assertIsNone(config["frequency_table"])
        self.assertIsNone(config["frequency_manifest"])
        self.assertIsNone(config["tune_manifest"])
        self.assertIsNone(config["calibration_manifest"])
        self.assertIsNone(config["test_manifest"])

    def test_wikitext_uses_current_dataset_repository_id(self):
        rows = [
            {"text": "This is a sufficiently long fixture line for loading."},
            {"text": "This is another sufficiently long fixture line."},
        ]
        config = {
            "seed": 42,
            "n_texts": 1,
            "dataset": "wikitext",
            "max_length": 32,
            "split": "validation",
        }

        with patch("eval_prediction_sets.load_dataset", return_value=rows) as mocked:
            loaded = load_texts(config)

        self.assertEqual(len(loaded), 1)
        mocked.assert_called_once_with(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="validation",
        )

    def test_evaluator_loads_external_counts_instead_of_rebuilding(self):
        _, frequency = self.write_manifest("freq", "freq")
        table_path = self.write_frequency_table(frequency)

        class Tokenizer:
            all_special_ids = [1]
            name_or_path = "fixture/model"
            init_kwargs = {"_commit_hash": "model-commit"}

        class ModelConfig:
            vocab_size = 3
            _commit_hash = "model-commit"

        class Model:
            config = ModelConfig()

        counts, metadata = resolve_token_counts(
            Tokenizer(),
            Model(),
            {
                "model": "fixture/model",
                "model_revision": "model-commit",
                "frequency_table": str(table_path),
                "allow_legacy_protocol": True,
                "max_length": 8,
                "batch_size": 1,
            },
            texts=[],
        )

        self.assertEqual(counts.tolist(), [4, 0, 7])
        self.assertEqual(metadata.counts_sha256, counts_sha256(counts))

    def test_effective_config_hash_is_order_independent(self):
        left = {"model": "fixture", "delta": 0.05, "nested": {"b": 2, "a": 1}}
        right = {"nested": {"a": 1, "b": 2}, "delta": 0.05, "model": "fixture"}

        self.assertEqual(effective_config_sha256(left), effective_config_sha256(right))
        self.assertNotEqual(
            effective_config_sha256(left),
            effective_config_sha256({**left, "delta": 0.1}),
        )

    def test_nonlegacy_protocol_lists_missing_inputs(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "frequency_table.*frequency_manifest.*tune_manifest.*calibration_manifest.*test_manifest",
        ):
            validate_protocol_inputs({"allow_legacy_protocol": False})

    def test_complete_inputs_remain_blocked_pending_pr1b(self):
        with self.assertRaisesRegex(RuntimeError, "blocked_pending_pr1b"):
            validate_protocol_inputs(self.complete_config())

    def test_manifest_role_mismatch_fails(self):
        config = self.complete_config()
        wrong_path, _ = self.write_manifest("test", "replacement")
        config["calibration_manifest"] = str(wrong_path)

        with self.assertRaisesRegex(ValueError, "role mismatch.*cal"):
            validate_protocol_inputs(config)

    def test_manifest_intersection_fails_before_pr1b_block(self):
        config = self.complete_config()
        frequency = json.loads(Path(config["frequency_manifest"]).read_text())
        test = json.loads(Path(config["test_manifest"]).read_text())
        test["manifest"]["documents"][0]["cluster_id"] = frequency["manifest"]["documents"][0]["cluster_id"]
        raw = test["manifest"]
        manifest = DocumentManifest(
            protocol_version=raw["protocol_version"],
            role=raw["role"],
            source=raw["source"],
            documents=tuple(ManifestDocument(**row) for row in raw["documents"]),
        )
        save_manifest(manifest, Path(config["test_manifest"]))

        with self.assertRaisesRegex(ValueError, "cluster_id intersection"):
            validate_protocol_inputs(config)

    def test_frequency_source_manifest_hash_mismatch_fails(self):
        config = self.complete_config()
        frequency_path = Path(config["frequency_manifest"])
        frequency_payload = json.loads(frequency_path.read_text())
        raw = frequency_payload["manifest"]
        frequency = DocumentManifest(
            protocol_version=raw["protocol_version"],
            role=raw["role"],
            source=raw["source"],
            documents=tuple(ManifestDocument(**row) for row in raw["documents"]),
        )
        config["frequency_table"] = str(
            self.write_frequency_table(
                frequency,
                source_hash=hashlib.sha256(b"wrong-source").hexdigest(),
            )
        )

        with self.assertRaisesRegex(ValueError, "source_manifest_sha256"):
            validate_protocol_inputs(config)

    def test_legacy_external_table_records_exact_reference(self):
        frequency_path, frequency = self.write_manifest("freq", "freq")
        table_path = self.write_frequency_table(frequency)
        protocol = validate_protocol_inputs(
            {
                "allow_legacy_protocol": True,
                "frequency_table": str(table_path),
                "frequency_manifest": str(frequency_path),
            }
        )

        reference = protocol["frequency_table"]
        self.assertEqual(reference["metadata_path"], str(table_path.resolve()))
        self.assertEqual(reference["source_manifest_sha256"], manifest_sha256(frequency))
        self.assertEqual(len(reference["artifact_id"]), 64)
        self.assertEqual(len(reference["counts_sha256"]), 64)

    def write_metrics_reference(self):
        frequency_path, frequency = self.write_manifest("freq", "freq")
        table_path = self.write_frequency_table(frequency)
        protocol = validate_protocol_inputs(
            {
                "allow_legacy_protocol": True,
                "frequency_table": str(table_path),
                "frequency_manifest": str(frequency_path),
            }
        )
        metrics_path = self.root / "prediction_set_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "model": "fixture/model",
                    "model_revision": "model-commit",
                    "protocol": protocol,
                    "q_hat": 2.0,
                    "kappa": 1.0,
                    "alpha": 1.0,
                }
            )
        )
        return metrics_path

    def load_metrics_counts(self, metrics_path):
        return load_frequency_table_from_metrics(
            metrics_path,
            expected_model_id="fixture/model",
            expected_model_revision="model-commit",
            expected_tokenizer_id="fixture/model",
            expected_tokenizer_revision="model-commit",
            expected_vocab_size=3,
            expected_exclusion_token_ids=(1,),
        )

    def test_downstream_loads_exact_artifact_referenced_by_metrics(self):
        metrics_path = self.write_metrics_reference()

        counts, metadata = self.load_metrics_counts(metrics_path)

        self.assertEqual(counts.tolist(), [4, 0, 7])
        self.assertEqual(metadata.counts_sha256, counts_sha256(counts))

    def test_downstream_entrypoints_use_the_upstream_artifact(self):
        metrics_path = self.write_metrics_reference()

        class Tokenizer:
            all_special_ids = [1]
            name_or_path = "fixture/model"
            init_kwargs = {"_commit_hash": "model-commit"}

        class ModelConfig:
            vocab_size = 3
            _commit_hash = "model-commit"

        class Model:
            config = ModelConfig()

        config = {
            "model": "fixture/model",
            "model_revision": "model-commit",
            "prediction_set_metrics": str(metrics_path),
        }
        for builder in (build_reasoning_counts, build_openended_counts):
            with self.subTest(builder=builder.__module__):
                counts = builder(Model(), Tokenizer(), config)
                self.assertEqual(counts.tolist(), [4, 0, 7])

    def test_downstream_rejects_missing_frequency_reference(self):
        metrics_path = self.root / "prediction_set_metrics.json"
        metrics_path.write_text(json.dumps({"protocol": {"frequency_table": None}}))

        with self.assertRaisesRegex(RuntimeError, "frequency_table"):
            self.load_metrics_counts(metrics_path)

    def test_downstream_rejects_recorded_hash_drift(self):
        metrics_path = self.write_metrics_reference()
        payload = json.loads(metrics_path.read_text())
        payload["protocol"]["frequency_table"]["counts_sha256"] = "tampered"
        metrics_path.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "counts_sha256"):
            self.load_metrics_counts(metrics_path)

    def test_downstream_resolves_relative_artifact_path_from_metrics(self):
        metrics_path = self.write_metrics_reference()
        payload = json.loads(metrics_path.read_text())
        absolute_path = Path(payload["protocol"]["frequency_table"]["metadata_path"])
        payload["protocol"]["frequency_table"]["metadata_path"] = os.path.relpath(
            absolute_path.resolve(),
            metrics_path.parent.resolve(),
        )
        metrics_path.write_text(json.dumps(payload))

        counts, _ = self.load_metrics_counts(metrics_path)

        self.assertEqual(counts.tolist(), [4, 0, 7])

    def test_legacy_flag_must_be_an_explicit_boolean(self):
        for value in ("false", 1, {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "allow_legacy_protocol.*boolean"):
                    validate_protocol_inputs({"allow_legacy_protocol": value})

    def test_nonlegacy_validates_counts_payload_before_pr1b_block(self):
        config = self.complete_config()
        sidecar = Path(config["frequency_table"])
        payload = json.loads(sidecar.read_text())
        (sidecar.parent / payload["counts_file"]).unlink()

        with self.assertRaisesRegex(ValueError, "counts file is missing"):
            validate_protocol_inputs(config)

    def test_runtime_tokenizer_identity_cannot_be_spoofed_by_config(self):
        _, frequency = self.write_manifest("freq", "freq")
        table_path = self.write_frequency_table(frequency)

        class Tokenizer:
            all_special_ids = [1]
            name_or_path = "other/model"
            init_kwargs = {"_commit_hash": "model-commit"}

        class ModelConfig:
            vocab_size = 3
            _commit_hash = "model-commit"

        class Model:
            config = ModelConfig()

        with self.assertRaisesRegex(ValueError, "tokenizer_id"):
            resolve_token_counts(
                Tokenizer(),
                Model(),
                {
                    "model": "fixture/model",
                    "model_revision": "model-commit",
                    "tokenizer_id": "fixture/model",
                    "frequency_table": str(table_path),
                    "allow_legacy_protocol": True,
                },
                texts=[],
            )

    def test_runtime_tokenizer_revision_must_match_artifact(self):
        _, frequency = self.write_manifest("freq", "freq")
        table_path = self.write_frequency_table(frequency)

        class Tokenizer:
            all_special_ids = [1]
            name_or_path = "fixture/model"
            init_kwargs = {"_commit_hash": "other-commit"}

        class ModelConfig:
            vocab_size = 3
            _commit_hash = "model-commit"

        class Model:
            config = ModelConfig()

        with self.assertRaisesRegex(ValueError, "tokenizer_revision"):
            resolve_token_counts(
                Tokenizer(),
                Model(),
                {
                    "model": "fixture/model",
                    "model_revision": "model-commit",
                    "frequency_table": str(table_path),
                    "allow_legacy_protocol": True,
                },
                texts=[],
            )


if __name__ == "__main__":
    unittest.main()
