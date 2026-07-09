import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from gate_evidence import (  # noqa: E402
        EVIDENCE_SCHEMA_VERSION,
        EvidenceCell,
        EvidenceProvenance,
        GateEvidence,
        PositionEvidence,
        gate_evidence_artifact_id,
        gate_evidence_test_rows_sha256,
        load_gate_evidence,
        partition_gate_evidence,
        save_gate_evidence,
        summarize_gate_evidence,
        validate_gate_evidence,
    )
    from methods import calibrate_method  # noqa: E402


def digest(character):
    return character * 64


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class GateEvidenceTests(unittest.TestCase):
    def calibration(self, method_key="c_margin"):
        logits = torch.tensor(
            [[4.0, 3.0, 1.0], [4.0, 2.0, 3.0], [3.0, 2.0, 1.0], [3.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 2, 1, 2])
        kwargs = {}
        if method_key == "c_nu":
            kwargs = {
                "token_freq_table": torch.tensor([0, 10, 100]),
                "params": {"kappa": -1.0, "alpha": 1.0},
            }
        elif method_key == "frequency_mondrian_margin":
            token_groups = torch.tensor([0, 1, 1])
            kwargs = {
                "groups": token_groups[targets],
                "expected_groups": (0, 1),
                "min_bucket": 1,
            }
        elif method_key == "entropy_mondrian_margin":
            kwargs = {
                "groups": torch.tensor([0, 0, 1, 1]),
                "expected_groups": (0, 1),
                "min_bucket": 1,
            }
        return calibrate_method(
            method_key,
            logits,
            targets,
            delta=0.5,
            uniforms=torch.zeros_like(logits),
            **kwargs,
        )

    def provenance(self):
        return EvidenceProvenance(
            created_by_commit="a" * 40,
            effective_config_sha256=digest("1"),
            primary_config_sha256=digest("2"),
            preregistration_artifact_id=digest("3"),
            gate_thresholds_sha256=digest("4"),
            frequency_table_artifact_id=digest("5"),
            frequency_counts_sha256=digest("6"),
            frequency_source_manifest_sha256=digest("7"),
            frequency_bucket_artifact_id=digest("8"),
            split_receipt_id=digest("9"),
            input_documents_sha256=digest("a"),
            cluster_manifest_sha256=digest("b"),
            tune_manifest_sha256=digest("c"),
            calibration_manifest_sha256=digest("d"),
            test_manifest_sha256=digest("e"),
            cross_corpus_artifact_id=digest("f"),
            cross_corpus_transcript_sha256=digest("0"),
            calibration_rows_sha256=digest("1"),
            randomization_artifact_sha256=digest("2"),
            calibration_position_salt_sha256=digest("3"),
            test_position_salt_sha256=digest("4"),
            tuning_artifact_id=None,
        )

    def evidence(self, method_key="c_margin", grade="G"):
        calibration = self.calibration(method_key)
        provenance = self.provenance()
        if method_key in {
            "c_nu",
            "frequency_mondrian_margin",
            "entropy_mondrian_margin",
        }:
            provenance = replace(provenance, tuning_artifact_id=digest("5"))
        rows = (
            PositionEvidence("doc-a", "cluster-a", 11, 1, True, 2, 0),
            PositionEvidence("doc-b", "cluster-b", 17, 2, False, 4, 1),
        )
        return GateEvidence(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            cell=EvidenceCell(
                model_id="Qwen/Qwen2.5-7B",
                model_revision="f" * 40,
                model_family="qwen2.5",
                domain_id="fixture-domain",
                domain_snapshot_sha256=digest("6"),
                vocab_size=8,
            ),
            method_key=method_key,
            delta=0.5,
            evidence_grade=grade,
            position_policy_id=(
                "one-position-per-document-v1" if grade == "G" else "stride-4-v1"
            ),
            frequency_group_kind="method-mass-quantile-v1",
            test_manifest_doc_count=2,
            calibration=calibration,
            provenance=provenance,
            records=rows,
        )

    def test_content_addressed_round_trip_and_tamper_rejection(self):
        evidence = self.evidence()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_gate_evidence(evidence, Path(tmp))
            loaded, artifact_id = load_gate_evidence(path)

            self.assertEqual(loaded, evidence)
            self.assertEqual(artifact_id, gate_evidence_artifact_id(evidence))
            self.assertEqual(path.name, f"{artifact_id}.json")

            payload = json.loads(path.read_text())
            payload["artifact_id"] = digest("0")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "artifact_id"):
                load_gate_evidence(path)

    def test_old_aggregate_metrics_cannot_masquerade_as_gate_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction_set_metrics.json"
            path.write_text(json.dumps({"methods": {}, "protocol": {}}))
            with self.assertRaisesRegex(ValueError, "wrapper"):
                load_gate_evidence(path)

    def test_g_requires_one_unique_document_and_cluster_and_exact_manifest_count(self):
        evidence = self.evidence()
        validate_gate_evidence(evidence)

        duplicate_doc = replace(
            evidence,
            records=(evidence.records[0], replace(evidence.records[0], target_index=12)),
        )
        with self.assertRaisesRegex(ValueError, "\[G\].*doc_id"):
            validate_gate_evidence(duplicate_doc)

        duplicate_cluster = replace(
            evidence,
            records=(evidence.records[0], replace(evidence.records[1], cluster_id="cluster-a")),
        )
        with self.assertRaisesRegex(ValueError, "\[G\].*cluster_id"):
            validate_gate_evidence(duplicate_cluster)

        with self.assertRaisesRegex(ValueError, "test manifest document count"):
            validate_gate_evidence(replace(evidence, test_manifest_doc_count=3))

    def test_e_allows_multiple_positions_but_preserves_document_cluster_mapping(self):
        evidence = self.evidence(grade="E")
        rows = (
            evidence.records[0],
            replace(evidence.records[0], target_index=15, covered=False, set_size=3),
        )
        pooled = replace(evidence, test_manifest_doc_count=2, records=rows)
        validate_gate_evidence(pooled)

        with self.assertRaisesRegex(ValueError, "doc_id.*cluster"):
            validate_gate_evidence(
                replace(
                    pooled,
                    records=(rows[0], replace(rows[1], cluster_id="other-cluster")),
                )
            )
        with self.assertRaisesRegex(ValueError, "doc_id.*target_index"):
            validate_gate_evidence(replace(pooled, records=(rows[0], rows[0])))

    def test_summary_is_losslessly_recomputed_from_position_records(self):
        summary = summarize_gate_evidence(self.evidence())

        self.assertEqual(summary["evidence_label"], "[G]")
        self.assertEqual(summary["n_positions"], 2)
        self.assertEqual(summary["n_documents"], 2)
        self.assertEqual(summary["n_clusters"], 2)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["mean_set_size"], 3.0)
        self.assertEqual(
            summary["test_rows_sha256"],
            gate_evidence_test_rows_sha256(self.evidence()),
        )
        self.assertEqual(summary["frequency_groups"]["0"]["coverage"], 1.0)
        self.assertEqual(summary["frequency_groups"]["1"]["coverage"], 0.0)

    def test_provenance_cell_and_position_fields_fail_closed(self):
        evidence = self.evidence()
        cases = (
            (replace(evidence.cell, model_revision="main"), "model_revision"),
            (replace(evidence.provenance, split_receipt_id="bad"), "split_receipt_id"),
            (replace(evidence.records[0], set_size=9), "set_size"),
            (replace(evidence.records[0], target_token_id=8), "target_token_id"),
        )
        for replacement, message in cases:
            with self.subTest(message=message):
                if isinstance(replacement, EvidenceCell):
                    candidate = replace(evidence, cell=replacement)
                elif isinstance(replacement, EvidenceProvenance):
                    candidate = replace(evidence, provenance=replacement)
                else:
                    candidate = replace(
                        evidence, records=(replacement, evidence.records[1])
                    )
                with self.assertRaisesRegex(ValueError, message):
                    validate_gate_evidence(candidate)

    def test_tuned_methods_require_tune_artifact_and_calibration_identity(self):
        evidence = self.evidence("c_nu")
        with self.assertRaisesRegex(ValueError, "tuning_artifact_id"):
            validate_gate_evidence(
                replace(
                    evidence,
                    provenance=replace(evidence.provenance, tuning_artifact_id=None),
                )
            )
        with self.assertRaisesRegex(ValueError, "calibration method_key"):
            validate_gate_evidence(replace(evidence, method_key="c_margin"))

    def test_registry_roles_partition_comparators_without_name_substrings(self):
        evidence = (
            self.evidence("c_margin"),
            self.evidence("entropy_mondrian_margin"),
            self.evidence("frequency_mondrian_margin"),
            self.evidence("c_nu"),
        )
        partition = partition_gate_evidence(evidence)

        self.assertEqual(
            {item.method_key for item in partition["non_frequency"]},
            {"c_margin", "entropy_mondrian_margin"},
        )
        self.assertEqual(
            {item.method_key for item in partition["frequency_primary"]},
            {"frequency_mondrian_margin"},
        )
        self.assertEqual(
            {item.method_key for item in partition["ablation"]},
            {"c_nu"},
        )

        mismatched_rows = replace(
            evidence[2],
            records=(replace(evidence[2].records[0], target_index=12), evidence[2].records[1]),
        )
        with self.assertRaisesRegex(ValueError, "frozen cell"):
            partition_gate_evidence((evidence[0], mismatched_rows))

    def test_records_require_canonical_order(self):
        evidence = self.evidence()
        with self.assertRaisesRegex(ValueError, "canonical.*order"):
            validate_gate_evidence(
                replace(evidence, records=tuple(reversed(evidence.records)))
            )

    def test_positive_infinity_threshold_has_canonical_json_round_trip(self):
        logits = torch.zeros((18, 2), dtype=torch.float64)
        calibration = calibrate_method(
            "c_margin",
            logits,
            torch.zeros(18, dtype=torch.long),
            delta=0.05,
            uniforms=torch.zeros_like(logits),
        )
        evidence = replace(
            self.evidence(), delta=0.05, calibration=calibration
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_gate_evidence(evidence, Path(tmp))
            self.assertIn('"+inf"', path.read_text())
            loaded, _ = load_gate_evidence(path)
        self.assertEqual(loaded, evidence)


if __name__ == "__main__":
    unittest.main()
