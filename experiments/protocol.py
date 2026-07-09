"""Fail-closed protocol validation shared by experiment entrypoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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

    PR-1a validates artifact identity and manifest disjointness, while PR-1b
    provides trusted split construction receipts and position selectors. The
    evaluator does not yet bind those artifacts to the exact forwarded text,
    so every non-legacy request remains blocked pending PR-1c.
    """
    legacy_flag = config.get("allow_legacy_protocol", False)
    if not isinstance(legacy_flag, bool):
        raise ValueError("allow_legacy_protocol must be a boolean")
    if legacy_flag is True:
        return _legacy_protocol(config)

    required = ["frequency_table", *_MANIFEST_CONFIG_ROLES, "model_revision"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise RuntimeError(
            "Paper-grade protocol inputs are missing: " + ", ".join(missing)
        )

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

    raise RuntimeError(
        "blocked_pending_pr1c: provenance checks passed and deterministic split "
        "construction is available, but evaluator text is not yet bound to a "
        "split receipt and manifest-selected positions. Paper-grade execution "
        "remains blocked."
    )
