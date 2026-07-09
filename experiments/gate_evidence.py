"""Content-addressed per-position evidence for the future calibrated gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from conformal import GroupQuantile
from frequency_buckets import METHOD_BUCKET_KIND
from methods import (
    METHOD_REGISTRY_VERSION,
    MethodCalibration,
    get_method_definition,
    method_registry,
)


EVIDENCE_SCHEMA_VERSION = "icml2027-gate-evidence-v2"
FREQUENCY_GROUP_KIND = METHOD_BUCKET_KIND
TUNING_REQUIRED_METHOD_KEYS = frozenset(
    {
        "raps",
        "ts_aps",
        "cns",
        "entropy_mondrian_margin",
        "frequency_mondrian_margin",
        "learned_h",
        "learned_g",
        "c_nu",
    }
)


@dataclass(frozen=True)
class EvidenceCell:
    model_id: str
    model_revision: str
    model_family: str
    domain_id: str
    domain_snapshot_sha256: str
    vocab_size: int


@dataclass(frozen=True)
class EvidenceProvenance:
    created_by_commit: str
    method_registry_sha256: str
    effective_config_sha256: str
    primary_config_sha256: str
    preregistration_artifact_id: str
    gate_thresholds_sha256: str
    frequency_table_artifact_id: str
    frequency_counts_sha256: str
    frequency_source_manifest_sha256: str
    frequency_bucket_artifact_id: str
    split_receipt_id: str
    input_documents_sha256: str
    cluster_manifest_sha256: str
    tune_manifest_sha256: str
    calibration_manifest_sha256: str
    test_manifest_sha256: str
    cross_corpus_artifact_id: str
    cross_corpus_transcript_sha256: str
    calibration_rows_sha256: str
    randomization_artifact_sha256: str
    calibration_position_salt_sha256: str
    test_position_salt_sha256: str
    tuning_artifact_id: str | None


@dataclass(frozen=True)
class PositionEvidence:
    doc_id: str
    cluster_id: str
    target_index: int
    target_token_id: int
    covered: bool
    set_size: int
    target_frequency_group: int


@dataclass(frozen=True)
class GateEvidence:
    schema_version: str
    cell: EvidenceCell
    method_key: str
    delta: float
    evidence_grade: str
    position_policy_id: str
    frequency_group_kind: str
    test_manifest_doc_count: int
    calibration: MethodCalibration
    provenance: EvidenceProvenance
    records: tuple[PositionEvidence, ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def current_method_registry_sha256() -> str:
    payload = [asdict(definition) for definition in method_registry()]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    value = _require_nonempty(value, field_name)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a canonical SHA-256 string")
    return value


def _require_commit(value: object, field_name: str) -> str:
    value = _require_nonempty(value, field_name)
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field_name} must be a pinned 40-hex commit")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_threshold(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    value = float(value)
    if math.isnan(value) or value == float("-inf"):
        raise ValueError(f"{field_name} must be finite or +inf")
    return value


def _validate_calibration(
    calibration: MethodCalibration,
    *,
    method_key: str,
    delta: float,
) -> None:
    if not isinstance(calibration, MethodCalibration):
        raise ValueError("calibration must be a MethodCalibration")
    if calibration.registry_version != METHOD_REGISTRY_VERSION:
        raise ValueError("calibration registry_version mismatch")
    if calibration.method_key != method_key:
        raise ValueError("calibration method_key mismatch")
    definition = get_method_definition(method_key)
    if not definition.implemented:
        raise ValueError(f"method_key {method_key!r} is not executable")
    _require_positive_int(calibration.n_calibration, "calibration n_calibration")
    if (
        isinstance(calibration.delta, bool)
        or not isinstance(calibration.delta, (int, float))
        or not math.isfinite(float(calibration.delta))
        or not 0.0 < float(calibration.delta) < 1.0
    ):
        raise ValueError("calibration delta must be in (0, 1)")
    if float(calibration.delta) != delta:
        raise ValueError("calibration delta mismatch")

    params = dict(calibration.params)
    if len(params) != len(calibration.params):
        raise ValueError("calibration params contain duplicate keys")
    if tuple(sorted(calibration.params)) != calibration.params:
        raise ValueError("calibration params must use canonical key order")
    expected_params = {"alpha", "kappa"} if method_key == "c_nu" else set()
    if set(params) != expected_params:
        raise ValueError("calibration params do not match method_key")
    for key, value in params.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"calibration param {key} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"calibration param {key} must be finite")
    if "alpha" in params and params["alpha"] <= 0:
        raise ValueError("calibration alpha must be positive")

    is_mondrian = method_key in {
        "frequency_mondrian_margin",
        "entropy_mondrian_margin",
    }
    if is_mondrian:
        if calibration.q_hat is not None:
            raise ValueError("Mondrian calibration must not have global q_hat")
        if calibration.group_axis != definition.conditioning_axis:
            raise ValueError("Mondrian calibration group_axis mismatch")
        if not calibration.group_quantiles:
            raise ValueError("Mondrian calibration needs group_quantiles")
        _require_positive_int(calibration.min_bucket, "calibration min_bucket")
    else:
        if calibration.q_hat is None:
            raise ValueError("global calibration is missing q_hat")
        _validate_threshold(calibration.q_hat, "calibration q_hat")
        if calibration.group_axis is not None or calibration.group_quantiles:
            raise ValueError("global calibration cannot contain Mondrian groups")
        if calibration.min_bucket is not None:
            raise ValueError("global calibration cannot contain min_bucket")
        rank = math.ceil((calibration.n_calibration + 1) * (1.0 - delta))
        rank_exceeds = rank > calibration.n_calibration
        if rank_exceeds != math.isinf(float(calibration.q_hat)):
            raise ValueError("global q_hat is inconsistent with conformal rank")

    if method_key == "aps":
        if calibration.dither_epsilon is not None:
            raise ValueError("APS calibration cannot contain score dither")
    elif (
        isinstance(calibration.dither_epsilon, bool)
        or not isinstance(calibration.dither_epsilon, (int, float))
        or not math.isfinite(float(calibration.dither_epsilon))
        or calibration.dither_epsilon <= 0
    ):
        raise ValueError("calibration dither_epsilon must be positive")

    seen_groups = set()
    total = 0
    previous = -1
    valid_reasons = {"finite", "absent", "below_min_bucket", "rank_exceeds_n"}
    for item in calibration.group_quantiles:
        if not isinstance(item, GroupQuantile):
            raise ValueError("calibration group_quantiles are malformed")
        if isinstance(item.group, bool) or not isinstance(item.group, int) or item.group < 0:
            raise ValueError("calibration group must be a non-negative integer")
        if item.group in seen_groups or item.group <= previous:
            raise ValueError("calibration groups must be unique and sorted")
        if isinstance(item.count, bool) or not isinstance(item.count, int) or item.count < 0:
            raise ValueError("calibration group count must be non-negative")
        threshold = _validate_threshold(item.q_hat, "calibration group q_hat")
        if not isinstance(item.finite, bool) or item.finite != math.isfinite(threshold):
            raise ValueError("calibration group finite flag mismatch")
        if item.reason not in valid_reasons:
            raise ValueError("calibration group reason is invalid")
        if item.reason == "finite" and not item.finite:
            raise ValueError("finite calibration group must have finite q_hat")
        if item.reason != "finite" and item.finite:
            raise ValueError("vacuous calibration group must have +inf q_hat")
        if item.reason == "absent" and item.count != 0:
            raise ValueError("absent calibration group must have count zero")
        if item.reason != "absent" and item.count == 0:
            raise ValueError("observed calibration group must have positive count")
        if is_mondrian and item.count > 0:
            if item.count < calibration.min_bucket:
                expected_reason = "below_min_bucket"
            else:
                rank = math.ceil((item.count + 1) * (1.0 - delta))
                expected_reason = "rank_exceeds_n" if rank > item.count else "finite"
            if item.reason != expected_reason:
                raise ValueError(
                    "calibration group reason is inconsistent with conformal rank/min_bucket"
                )
        seen_groups.add(item.group)
        previous = item.group
        total += item.count
    if is_mondrian and total != calibration.n_calibration:
        raise ValueError("calibration group counts do not sum to n_calibration")


def validate_gate_evidence(evidence: GateEvidence) -> None:
    if not isinstance(evidence, GateEvidence):
        raise ValueError("evidence must be a GateEvidence")
    if evidence.schema_version != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported gate evidence schema_version")
    if not isinstance(evidence.cell, EvidenceCell):
        raise ValueError("cell must be EvidenceCell")
    _require_nonempty(evidence.cell.model_id, "model_id")
    _require_commit(evidence.cell.model_revision, "model_revision")
    _require_nonempty(evidence.cell.model_family, "model_family")
    _require_nonempty(evidence.cell.domain_id, "domain_id")
    _require_sha256(evidence.cell.domain_snapshot_sha256, "domain_snapshot_sha256")
    _require_positive_int(evidence.cell.vocab_size, "vocab_size")

    definition = get_method_definition(evidence.method_key)
    if not definition.implemented:
        raise ValueError(f"method_key {evidence.method_key!r} is not executable")
    if (
        isinstance(evidence.delta, bool)
        or not isinstance(evidence.delta, (int, float))
        or not math.isfinite(float(evidence.delta))
        or not 0.0 < float(evidence.delta) < 1.0
    ):
        raise ValueError("delta must be in (0, 1)")
    delta = float(evidence.delta)
    _validate_calibration(
        evidence.calibration,
        method_key=evidence.method_key,
        delta=delta,
    )

    policies = {"G": "one-position-per-document-v1", "E": "stride-4-v1"}
    if evidence.evidence_grade not in policies:
        raise ValueError("evidence_grade must be 'G' or 'E'")
    if evidence.position_policy_id != policies[evidence.evidence_grade]:
        raise ValueError("position_policy_id does not match evidence_grade")
    if evidence.frequency_group_kind != FREQUENCY_GROUP_KIND:
        raise ValueError("frequency_group_kind must use frozen method buckets")
    _require_positive_int(evidence.test_manifest_doc_count, "test_manifest_doc_count")

    if not isinstance(evidence.provenance, EvidenceProvenance):
        raise ValueError("provenance must be EvidenceProvenance")
    _require_commit(evidence.provenance.created_by_commit, "created_by_commit")
    for item in fields(EvidenceProvenance):
        if item.name in {"created_by_commit", "tuning_artifact_id"}:
            continue
        _require_sha256(getattr(evidence.provenance, item.name), item.name)
    if evidence.provenance.method_registry_sha256 != current_method_registry_sha256():
        raise ValueError("method_registry_sha256 does not match the runtime registry")
    if evidence.provenance.tuning_artifact_id is not None:
        _require_sha256(
            evidence.provenance.tuning_artifact_id,
            "tuning_artifact_id",
        )
    if (
        evidence.method_key in TUNING_REQUIRED_METHOD_KEYS
        and evidence.provenance.tuning_artifact_id is None
    ):
        raise ValueError(f"{evidence.method_key} requires tuning_artifact_id")
    if (
        evidence.provenance.calibration_position_salt_sha256
        == evidence.provenance.test_position_salt_sha256
    ):
        raise ValueError("calibration/test position salt hashes must differ")

    if not isinstance(evidence.records, tuple) or not evidence.records:
        raise ValueError("records must be a non-empty tuple")
    if any(not isinstance(record, PositionEvidence) for record in evidence.records):
        raise ValueError("records must contain PositionEvidence values")
    canonical_records = tuple(
        sorted(
            evidence.records,
            key=lambda item: (item.cluster_id, item.doc_id, item.target_index),
        )
    )
    if evidence.records != canonical_records:
        raise ValueError("records must use canonical cluster/doc/position order")
    doc_to_cluster = {}
    cluster_to_doc = {}
    positions = set()
    for record in evidence.records:
        _require_nonempty(record.doc_id, "doc_id")
        _require_nonempty(record.cluster_id, "cluster_id")
        _require_positive_int(record.target_index, "target_index")
        if (
            isinstance(record.target_token_id, bool)
            or not isinstance(record.target_token_id, int)
            or not 0 <= record.target_token_id < evidence.cell.vocab_size
        ):
            raise ValueError("target_token_id must be inside vocab_size")
        if not isinstance(record.covered, bool):
            raise ValueError("covered must be boolean")
        if (
            isinstance(record.set_size, bool)
            or not isinstance(record.set_size, int)
            or not 1 <= record.set_size <= evidence.cell.vocab_size
        ):
            raise ValueError("set_size must be inside [1, vocab_size]")
        if (
            isinstance(record.target_frequency_group, bool)
            or not isinstance(record.target_frequency_group, int)
            or record.target_frequency_group < 0
        ):
            raise ValueError("target_frequency_group must be non-negative")
        prior_cluster = doc_to_cluster.setdefault(record.doc_id, record.cluster_id)
        if prior_cluster != record.cluster_id:
            raise ValueError("doc_id must map to exactly one cluster")
        prior_doc = cluster_to_doc.setdefault(record.cluster_id, record.doc_id)
        if prior_doc != record.doc_id:
            prefix = "[G] " if evidence.evidence_grade == "G" else ""
            raise ValueError(f"{prefix}cluster_id must map to exactly one doc_id")
        position = (record.doc_id, record.target_index)
        if position in positions:
            raise ValueError("duplicate doc_id and target_index record")
        positions.add(position)

    if len(doc_to_cluster) > evidence.test_manifest_doc_count:
        raise ValueError("records exceed test manifest document count")
    if evidence.evidence_grade == "G":
        if len(doc_to_cluster) != len(evidence.records):
            raise ValueError("[G] evidence requires unique doc_id values")
        if len(cluster_to_doc) != len(evidence.records):
            raise ValueError("[G] evidence requires unique cluster_id values")
        if len(evidence.records) != evidence.test_manifest_doc_count:
            raise ValueError("[G] evidence must match test manifest document count")


def _encode_threshold(value: float | None) -> float | str | None:
    if value is None:
        return None
    value = _validate_threshold(value, "q_hat")
    return "+inf" if value == float("inf") else value


def _decode_threshold(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if value == "+inf":
        return float("inf")
    return _validate_threshold(value, field_name)


def _calibration_payload(calibration: MethodCalibration) -> dict[str, object]:
    return {
        "registry_version": calibration.registry_version,
        "method_key": calibration.method_key,
        "delta": calibration.delta,
        "n_calibration": calibration.n_calibration,
        "q_hat": _encode_threshold(calibration.q_hat),
        "group_quantiles": [
            {
                "group": item.group,
                "count": item.count,
                "q_hat": _encode_threshold(item.q_hat),
                "finite": item.finite,
                "reason": item.reason,
            }
            for item in calibration.group_quantiles
        ],
        "params": [[key, value] for key, value in calibration.params],
        "group_axis": calibration.group_axis,
        "dither_epsilon": calibration.dither_epsilon,
        "min_bucket": calibration.min_bucket,
    }


def _evidence_payload(evidence: GateEvidence) -> dict[str, object]:
    return {
        "schema_version": evidence.schema_version,
        "cell": asdict(evidence.cell),
        "method_key": evidence.method_key,
        "delta": evidence.delta,
        "evidence_grade": evidence.evidence_grade,
        "position_policy_id": evidence.position_policy_id,
        "frequency_group_kind": evidence.frequency_group_kind,
        "test_manifest_doc_count": evidence.test_manifest_doc_count,
        "calibration": _calibration_payload(evidence.calibration),
        "provenance": asdict(evidence.provenance),
        "records": [asdict(record) for record in evidence.records],
    }


def gate_evidence_artifact_id(evidence: GateEvidence) -> str:
    validate_gate_evidence(evidence)
    return hashlib.sha256(_canonical_json(_evidence_payload(evidence))).hexdigest()


def gate_evidence_test_rows_sha256(evidence: GateEvidence) -> str:
    """Hash the method-independent ordered test sampling frame."""
    validate_gate_evidence(evidence)
    rows = [
        {
            "doc_id": item.doc_id,
            "cluster_id": item.cluster_id,
            "target_index": item.target_index,
            "target_token_id": item.target_token_id,
            "target_frequency_group": item.target_frequency_group,
        }
        for item in evidence.records
    ]
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def save_gate_evidence(evidence: GateEvidence, output_dir: Path) -> Path:
    artifact_id = gate_evidence_artifact_id(evidence)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{artifact_id}.json"
    wrapper = {"artifact_id": artifact_id, "evidence": _evidence_payload(evidence)}
    path.write_bytes(_canonical_json(wrapper) + b"\n")
    return path


def _require_keys(payload: object, expected: set[str], field_name: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{field_name} has invalid fields")
    return payload


def _parse_calibration(payload: object) -> MethodCalibration:
    expected = {
        "registry_version",
        "method_key",
        "delta",
        "n_calibration",
        "q_hat",
        "group_quantiles",
        "params",
        "group_axis",
        "dither_epsilon",
        "min_bucket",
    }
    payload = _require_keys(payload, expected, "calibration")
    raw_groups = payload["group_quantiles"]
    if not isinstance(raw_groups, list):
        raise ValueError("calibration group_quantiles must be a list")
    groups = []
    group_fields = {"group", "count", "q_hat", "finite", "reason"}
    for raw in raw_groups:
        raw = _require_keys(raw, group_fields, "calibration group")
        groups.append(
            GroupQuantile(
                group=raw["group"],
                count=raw["count"],
                q_hat=_decode_threshold(raw["q_hat"], "calibration group q_hat"),
                finite=raw["finite"],
                reason=raw["reason"],
            )
        )
    raw_params = payload["params"]
    if not isinstance(raw_params, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in raw_params
    ):
        raise ValueError("calibration params must be key/value pairs")
    return MethodCalibration(
        registry_version=payload["registry_version"],
        method_key=payload["method_key"],
        delta=payload["delta"],
        n_calibration=payload["n_calibration"],
        q_hat=_decode_threshold(payload["q_hat"], "calibration q_hat"),
        group_quantiles=tuple(groups),
        params=tuple((item[0], item[1]) for item in raw_params),
        group_axis=payload["group_axis"],
        dither_epsilon=payload["dither_epsilon"],
        min_bucket=payload["min_bucket"],
    )


def load_gate_evidence(path: Path) -> tuple[GateEvidence, str]:
    path = Path(path)
    try:
        wrapper = json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("failed to read gate evidence") from exc
    wrapper = _require_keys(wrapper, {"artifact_id", "evidence"}, "gate evidence wrapper")
    raw = _require_keys(
        wrapper["evidence"],
        {
            "schema_version",
            "cell",
            "method_key",
            "delta",
            "evidence_grade",
            "position_policy_id",
            "frequency_group_kind",
            "test_manifest_doc_count",
            "calibration",
            "provenance",
            "records",
        },
        "gate evidence",
    )
    cell_payload = _require_keys(
        raw["cell"], {item.name for item in fields(EvidenceCell)}, "cell"
    )
    provenance_payload = _require_keys(
        raw["provenance"],
        {item.name for item in fields(EvidenceProvenance)},
        "provenance",
    )
    raw_records = raw["records"]
    if not isinstance(raw_records, list):
        raise ValueError("records must be a list")
    record_fields = {item.name for item in fields(PositionEvidence)}
    records = tuple(
        PositionEvidence(**_require_keys(item, record_fields, "position record"))
        for item in raw_records
    )
    try:
        evidence = GateEvidence(
            schema_version=raw["schema_version"],
            cell=EvidenceCell(**cell_payload),
            method_key=raw["method_key"],
            delta=raw["delta"],
            evidence_grade=raw["evidence_grade"],
            position_policy_id=raw["position_policy_id"],
            frequency_group_kind=raw["frequency_group_kind"],
            test_manifest_doc_count=raw["test_manifest_doc_count"],
            calibration=_parse_calibration(raw["calibration"]),
            provenance=EvidenceProvenance(**provenance_payload),
            records=records,
        )
    except TypeError as exc:
        raise ValueError("gate evidence contains invalid field types") from exc
    artifact_id = gate_evidence_artifact_id(evidence)
    if wrapper["artifact_id"] != artifact_id:
        raise ValueError("artifact_id mismatch for gate evidence")
    if path.name != f"{artifact_id}.json":
        raise ValueError("artifact_id filename mismatch for gate evidence")
    return evidence, artifact_id


def summarize_gate_evidence(evidence: GateEvidence) -> dict[str, object]:
    validate_gate_evidence(evidence)
    records = evidence.records
    groups = {}
    for group in sorted({item.target_frequency_group for item in records}):
        selected = [item for item in records if item.target_frequency_group == group]
        groups[str(group)] = {
            "n": len(selected),
            "coverage": sum(item.covered for item in selected) / len(selected),
            "mean_set_size": sum(item.set_size for item in selected) / len(selected),
        }
    return {
        "method_key": evidence.method_key,
        "delta": float(evidence.delta),
        "evidence_label": f"[{evidence.evidence_grade}]",
        "n_positions": len(records),
        "n_documents": len({item.doc_id for item in records}),
        "n_clusters": len({item.cluster_id for item in records}),
        "test_rows_sha256": gate_evidence_test_rows_sha256(evidence),
        "coverage": sum(item.covered for item in records) / len(records),
        "mean_set_size": sum(item.set_size for item in records) / len(records),
        "frequency_groups": groups,
    }


def partition_gate_evidence(
    evidence_items: Iterable[GateEvidence],
) -> dict[str, tuple[GateEvidence, ...]]:
    items = tuple(evidence_items)
    if not items:
        raise ValueError("gate evidence partition requires at least one artifact")
    for item in items:
        validate_gate_evidence(item)
    reference = items[0]
    for item in items[1:]:
        if (
            item.cell != reference.cell
            or float(item.delta) != float(reference.delta)
            or item.evidence_grade != reference.evidence_grade
            or item.provenance.test_manifest_sha256
            != reference.provenance.test_manifest_sha256
            or item.provenance.calibration_manifest_sha256
            != reference.provenance.calibration_manifest_sha256
            or item.provenance.calibration_rows_sha256
            != reference.provenance.calibration_rows_sha256
            or item.provenance.created_by_commit
            != reference.provenance.created_by_commit
            or item.provenance.method_registry_sha256
            != reference.provenance.method_registry_sha256
            or item.provenance.primary_config_sha256
            != reference.provenance.primary_config_sha256
            or item.provenance.preregistration_artifact_id
            != reference.provenance.preregistration_artifact_id
            or item.provenance.gate_thresholds_sha256
            != reference.provenance.gate_thresholds_sha256
            or item.provenance.randomization_artifact_sha256
            != reference.provenance.randomization_artifact_sha256
            or item.provenance.calibration_position_salt_sha256
            != reference.provenance.calibration_position_salt_sha256
            or item.provenance.test_position_salt_sha256
            != reference.provenance.test_position_salt_sha256
            or item.provenance.frequency_table_artifact_id
            != reference.provenance.frequency_table_artifact_id
            or item.provenance.split_receipt_id
            != reference.provenance.split_receipt_id
            or item.provenance.cross_corpus_artifact_id
            != reference.provenance.cross_corpus_artifact_id
            or item.provenance.frequency_bucket_artifact_id
            != reference.provenance.frequency_bucket_artifact_id
            or gate_evidence_test_rows_sha256(item)
            != gate_evidence_test_rows_sha256(reference)
        ):
            raise ValueError("gate evidence comparators do not share one frozen cell")
    partitions = {"non_frequency": [], "frequency_primary": [], "ablation": []}
    for item in items:
        role = get_method_definition(item.method_key).paper_role
        if role in {"null", "baseline", "conditioning_control"}:
            partitions["non_frequency"].append(item)
        elif role == "frequency_family":
            partitions["frequency_primary"].append(item)
        else:
            partitions["ablation"].append(item)
    return {key: tuple(value) for key, value in partitions.items()}
