"""Fail-closed protocol validation shared by experiment entrypoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cross_corpus import validate_cross_corpus_audit
from freq_table import (
    PROTOCOL_VERSION,
    load_frequency_table,
    load_frequency_table_metadata,
)
from splits import assert_pairwise_disjoint, load_manifest, manifest_sha256


_MANIFEST_CONFIG_ROLES = {
    "frequency_manifest": "freq",
    "tune_manifest": "tune",
    "calibration_manifest": "cal",
    "test_manifest": "test",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Config value is not JSON-serializable: {type(value).__name__}")


def effective_config_sha256(config: dict[str, Any]) -> str:
    """Hash the complete effective configuration with canonical JSON."""
    payload = json.dumps(
        _json_value(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frequency_table_reference(metadata_path: str | Path) -> dict[str, str]:
    resolved_path = Path(metadata_path).expanduser().resolve()
    metadata, artifact_id, _ = load_frequency_table_metadata(resolved_path)
    return {
        "metadata_path": str(resolved_path),
        "artifact_id": artifact_id,
        "counts_sha256": metadata.counts_sha256,
        "source_manifest_sha256": metadata.source_manifest_sha256,
    }


def _legacy_protocol(config: dict[str, Any]) -> dict[str, Any]:
    reference = None
    if config.get("frequency_table"):
        reference = _frequency_table_reference(config["frequency_table"])
        if config.get("frequency_manifest"):
            frequency_manifest = load_manifest(Path(config["frequency_manifest"]))
            if frequency_manifest.role != "freq":
                raise ValueError(
                    "role mismatch for frequency_manifest: "
                    f"expected='freq' actual={frequency_manifest.role!r}"
                )
            actual_source_hash = manifest_sha256(frequency_manifest)
            if reference["source_manifest_sha256"] != actual_source_hash:
                raise ValueError(
                    "source_manifest_sha256 mismatch: "
                    f"artifact={reference['source_manifest_sha256']!r} "
                    f"manifest={actual_source_hash!r}"
                )
    return {
        "protocol_version": "legacy-pre-pr1",
        "evidence_grade": "legacy-smoke",
        "paper_grade": False,
        "frequency_table": reference,
    }


def validate_protocol_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Validate provenance before any model or dataset allocation.

    PR-1a validates frequency artifacts, PR-1b provides split receipts, PR-1c
    binds those receipts to exact source text, and PR-1d recomputes a fixed
    threshold-complete D_freq/evaluation near-duplicate proof. Non-legacy
    requests remain blocked until the PR-2 conformal core and PR-3 gate land.
    """
    legacy_flag = config.get("allow_legacy_protocol", False)
    if not isinstance(legacy_flag, bool):
        raise ValueError("allow_legacy_protocol must be a boolean")
    if legacy_flag is True:
        return _legacy_protocol(config)

    required = [
        "frequency_table",
        *_MANIFEST_CONFIG_ROLES,
        "model_revision",
        "split_receipt",
        "document_jsonl",
        "frequency_document_jsonl",
        "cross_corpus_receipt",
        "calibration_position_salt",
        "test_position_salt",
    ]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise RuntimeError(
            "Paper-grade protocol inputs are missing: " + ", ".join(missing)
        )
    position_salts = (
        config["calibration_position_salt"],
        config["test_position_salt"],
    )
    if any(not isinstance(value, str) or not value.strip() for value in position_salts):
        raise ValueError("calibration and test position salts must be non-empty strings")
    if config["calibration_position_salt"] == config["test_position_salt"]:
        raise ValueError("calibration and test position salts must differ")

    manifests = {}
    for config_key, expected_role in _MANIFEST_CONFIG_ROLES.items():
        manifest = load_manifest(Path(config[config_key]))
        if manifest.role != expected_role:
            raise ValueError(
                f"role mismatch for {config_key}: expected={expected_role!r} "
                f"actual={manifest.role!r}"
            )
        if manifest.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"protocol_version mismatch for {config_key}: "
                f"expected={PROTOCOL_VERSION!r} actual={manifest.protocol_version!r}"
            )
        manifests[expected_role] = manifest
    assert_pairwise_disjoint(manifests)

    metadata_path = Path(config["frequency_table"]).expanduser().resolve()
    metadata, _, _ = load_frequency_table_metadata(metadata_path)
    if metadata.tokenizer_revision is None:
        raise ValueError(
            "tokenizer_revision must be pinned for nonlegacy protocol inputs"
        )
    if metadata.tokenizer_revision != config["model_revision"]:
        raise ValueError(
            "tokenizer_revision mismatch between frequency artifact and "
            f"model_revision: artifact={metadata.tokenizer_revision!r} "
            f"model={config['model_revision']!r}"
        )
    load_frequency_table(
        metadata_path,
        expected_model_id=metadata.model_id,
        expected_tokenizer_id=metadata.tokenizer_id,
        expected_tokenizer_revision=metadata.tokenizer_revision,
        expected_vocab_size=metadata.vocab_size,
        expected_exclusion_token_ids=metadata.exclusion_token_ids,
    )
    reference = _frequency_table_reference(metadata_path)
    frequency_manifest_hash = manifest_sha256(manifests["freq"])
    if reference["source_manifest_sha256"] != frequency_manifest_hash:
        raise ValueError(
            "source_manifest_sha256 mismatch: "
            f"artifact={reference['source_manifest_sha256']!r} "
            f"manifest={frequency_manifest_hash!r}"
        )
    if metadata.num_documents != len(manifests["freq"].documents):
        raise ValueError(
            "frequency table num_documents does not match frequency manifest: "
            f"artifact={metadata.num_documents} "
            f"manifest={len(manifests['freq'].documents)}"
        )

    cross_receipt = validate_cross_corpus_audit(
        Path(config["cross_corpus_receipt"]),
        frequency_manifest_path=Path(config["frequency_manifest"]),
        frequency_document_jsonl=Path(config["frequency_document_jsonl"]),
        evaluation_split_receipt_path=Path(config["split_receipt"]),
        evaluation_document_jsonl=Path(config["document_jsonl"]),
        configured_eval_manifests={
            "tune": Path(config["tune_manifest"]),
            "cal": Path(config["calibration_manifest"]),
            "test": Path(config["test_manifest"]),
        },
    )
    if cross_receipt.frequency_manifest_sha256 != frequency_manifest_hash:
        raise ValueError("cross-corpus frequency manifest changed during validation")
    if cross_receipt.frequency_document_count != metadata.num_documents:
        raise ValueError(
            "cross-corpus frequency document count does not match frequency table"
        )
    cross_role_hashes = {
        "tune": cross_receipt.evaluation_tune_manifest_sha256,
        "cal": cross_receipt.evaluation_cal_manifest_sha256,
        "test": cross_receipt.evaluation_test_manifest_sha256,
    }
    for role in ("tune", "cal", "test"):
        if cross_role_hashes[role] != manifest_sha256(manifests[role]):
            raise ValueError(
                f"cross-corpus evaluation manifest changed during validation: {role}"
            )

    raise RuntimeError(
        "blocked_pending_pr2_pr3: PR-1 provenance, deterministic document splits, "
        "exact text binding, and recomputed cross-corpus disjointness all passed, "
        "but the conformal core/method registry and calibrated-vs-calibrated gate "
        "are not yet complete. Paper-grade execution remains blocked."
    )
