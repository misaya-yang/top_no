"""Immutable token-frequency artifacts with verifiable provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Collection

import torch


PROTOCOL_VERSION = "icml2027-pr1a"


@dataclass(frozen=True)
class FrequencyTableMetadata:
    protocol_version: str
    model_id: str
    tokenizer_id: str
    tokenizer_revision: str | None
    vocab_size: int
    counts_dtype: str
    counts_sha256: str
    source_manifest_sha256: str
    exclusion_token_ids: tuple[int, ...]
    num_documents: int
    num_tokens: int


def special_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the tokenizer's valid integer special IDs in canonical order."""
    values = getattr(tokenizer, "all_special_ids", ()) or ()
    return tuple(
        sorted(
            {
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
    )


def runtime_tokenizer_identity(
    tokenizer: Any,
    resolved_model_revision: str | None = None,
) -> tuple[str, str | None]:
    """Return the tokenizer ID and commit resolved at the shared load boundary."""
    tokenizer_id = getattr(tokenizer, "name_or_path", None)
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise ValueError("runtime tokenizer_id is unavailable")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    tokenizer_revision = (
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
    )
    for revision in (tokenizer_revision, resolved_model_revision):
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("runtime tokenizer_revision must be a non-empty string")
    if (
        tokenizer_revision is not None
        and resolved_model_revision is not None
        and tokenizer_revision != resolved_model_revision
    ):
        raise ValueError(
            "tokenizer_revision mismatch between tokenizer and resolved model: "
            f"tokenizer={tokenizer_revision!r} model={resolved_model_revision!r}"
        )
    revision = tokenizer_revision or resolved_model_revision
    return tokenizer_id, revision


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _normalize_exclusion_ids(
    exclusion_token_ids: Collection[int],
    vocab_size: int,
) -> tuple[int, ...]:
    normalized: set[int] = set()
    for value in exclusion_token_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("exclusion_token_ids must contain integers")
        if value < 0 or value >= vocab_size:
            raise ValueError(
                f"exclusion_token_ids contains out-of-range id {value} "
                f"for vocab_size={vocab_size}"
            )
        normalized.add(value)
    return tuple(sorted(normalized))


def _validated_int64_counts(counts: torch.Tensor) -> torch.Tensor:
    if not isinstance(counts, torch.Tensor):
        raise ValueError("counts must be a torch.Tensor")
    if counts.dim() != 1:
        raise ValueError("counts must be one-dimensional")
    if counts.is_floating_point():
        if not torch.isfinite(counts).all():
            raise ValueError("counts must be finite and integer-valued")
        if not torch.equal(counts, torch.round(counts)):
            raise ValueError("counts must be integer-valued")
    elif counts.is_complex():
        raise ValueError("counts must be integer-valued")
    canonical = counts.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if (canonical < 0).any():
        raise ValueError("counts must be non-negative")
    return canonical


def _canonicalize_for_artifact(
    counts: torch.Tensor,
    exclusion_token_ids: Collection[int],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    canonical = _validated_int64_counts(counts)
    exclusions = _normalize_exclusion_ids(exclusion_token_ids, canonical.numel())
    if exclusions:
        canonical = canonical.clone()
        canonical[list(exclusions)] = 0
    return canonical, exclusions


def counts_sha256(counts: torch.Tensor) -> str:
    """Hash canonical one-dimensional CPU int64 counts."""
    canonical = _validated_int64_counts(counts)
    digest = hashlib.sha256()
    digest.update(f"torch.int64:{canonical.numel()}\n".encode("utf-8"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _metadata_dict(metadata: FrequencyTableMetadata) -> dict[str, Any]:
    payload = asdict(metadata)
    payload["exclusion_token_ids"] = list(metadata.exclusion_token_ids)
    return payload


def _artifact_id(metadata: FrequencyTableMetadata) -> str:
    return hashlib.sha256(_canonical_json(_metadata_dict(metadata))).hexdigest()


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    value = _require_string(value, field_name)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"{field_name} must be a canonical 64-character lowercase SHA-256 digest"
        )
    return value


def _validate_metadata(metadata: FrequencyTableMetadata) -> None:
    if metadata.protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"protocol_version mismatch: artifact={metadata.protocol_version!r} "
            f"expected={PROTOCOL_VERSION!r}"
        )
    _require_string(metadata.model_id, "model_id")
    _require_string(metadata.tokenizer_id, "tokenizer_id")
    if metadata.tokenizer_revision is not None:
        _require_string(metadata.tokenizer_revision, "tokenizer_revision")
    if _require_integer(metadata.vocab_size, "vocab_size") <= 0:
        raise ValueError("vocab_size must be positive")
    if metadata.counts_dtype != "torch.int64":
        raise ValueError("counts_dtype must equal 'torch.int64'")
    _require_sha256(metadata.counts_sha256, "counts_sha256")
    _require_sha256(metadata.source_manifest_sha256, "source_manifest_sha256")
    if _require_integer(metadata.num_documents, "num_documents") < 0:
        raise ValueError("num_documents must be non-negative")
    if _require_integer(metadata.num_tokens, "num_tokens") < 0:
        raise ValueError("num_tokens must be non-negative")
    normalized = _normalize_exclusion_ids(
        metadata.exclusion_token_ids,
        metadata.vocab_size,
    )
    if normalized != metadata.exclusion_token_ids:
        raise ValueError("exclusion_token_ids must be sorted and unique")


def make_frequency_table_metadata(
    counts: torch.Tensor,
    *,
    model_id: str,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    source_manifest_sha256: str,
    exclusion_token_ids: Collection[int],
    num_documents: int,
) -> FrequencyTableMetadata:
    """Create metadata from validated counts and frozen source provenance."""
    canonical, exclusions = _canonicalize_for_artifact(counts, exclusion_token_ids)
    _require_string(model_id, "model_id")
    _require_string(tokenizer_id, "tokenizer_id")
    if tokenizer_revision is not None:
        _require_string(tokenizer_revision, "tokenizer_revision")
    _require_sha256(source_manifest_sha256, "source_manifest_sha256")
    if num_documents < 0:
        raise ValueError("num_documents must be non-negative")
    metadata = FrequencyTableMetadata(
        protocol_version=PROTOCOL_VERSION,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        vocab_size=canonical.numel(),
        counts_dtype="torch.int64",
        counts_sha256=counts_sha256(canonical),
        source_manifest_sha256=source_manifest_sha256,
        exclusion_token_ids=exclusions,
        num_documents=num_documents,
        num_tokens=int(canonical.sum().item()),
    )
    _validate_metadata(metadata)
    return metadata


def _validate_counts_against_metadata(
    counts: torch.Tensor,
    metadata: FrequencyTableMetadata,
) -> torch.Tensor:
    canonical = _validated_int64_counts(counts)
    if canonical.numel() != metadata.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: metadata={metadata.vocab_size} "
            f"counts={canonical.numel()}"
        )
    exclusions = _normalize_exclusion_ids(
        metadata.exclusion_token_ids,
        metadata.vocab_size,
    )
    if exclusions and (canonical[list(exclusions)] != 0).any():
        raise ValueError("excluded token counts must be zero")
    actual_hash = counts_sha256(canonical)
    if actual_hash != metadata.counts_sha256:
        raise ValueError(
            f"counts_sha256 mismatch: metadata={metadata.counts_sha256!r} "
            f"actual={actual_hash!r}"
        )
    if metadata.counts_dtype != "torch.int64":
        raise ValueError(f"Unsupported counts_dtype: {metadata.counts_dtype!r}")
    if int(canonical.sum().item()) != metadata.num_tokens:
        raise ValueError(
            f"num_tokens mismatch: metadata={metadata.num_tokens} "
            f"actual={int(canonical.sum().item())}"
        )
    return canonical


def save_frequency_table(
    counts: torch.Tensor,
    metadata: FrequencyTableMetadata,
    output_dir: Path,
) -> Path:
    """Write deterministic count and metadata filenames for an artifact."""
    canonical, exclusions = _canonicalize_for_artifact(
        counts,
        metadata.exclusion_token_ids,
    )
    if exclusions != metadata.exclusion_token_ids:
        raise ValueError("exclusion_token_ids are not canonical")
    _validate_metadata(metadata)
    canonical = _validate_counts_against_metadata(canonical, metadata)
    artifact_id = _artifact_id(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_name = f"{artifact_id}.counts.pt"
    counts_path = output_dir / counts_name
    sidecar_path = output_dir / f"{artifact_id}.json"
    torch.save(canonical, counts_path)
    payload = {
        "artifact_id": artifact_id,
        "counts_file": counts_name,
        "metadata": _metadata_dict(metadata),
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sidecar_path


def _metadata_from_dict(payload: object) -> FrequencyTableMetadata:
    if not isinstance(payload, dict):
        raise ValueError("frequency-table metadata must be a JSON object")
    required = {
        "protocol_version",
        "model_id",
        "tokenizer_id",
        "tokenizer_revision",
        "vocab_size",
        "counts_dtype",
        "counts_sha256",
        "source_manifest_sha256",
        "exclusion_token_ids",
        "num_documents",
        "num_tokens",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            f"frequency-table metadata fields mismatch: missing={missing} extra={extra}"
        )
    revision = payload["tokenizer_revision"]
    if revision is not None and not isinstance(revision, str):
        raise ValueError("tokenizer_revision must be a string or null")
    exclusions = payload["exclusion_token_ids"]
    if not isinstance(exclusions, list):
        raise ValueError("exclusion_token_ids must be a JSON array")
    metadata = FrequencyTableMetadata(
        protocol_version=_require_string(payload["protocol_version"], "protocol_version"),
        model_id=_require_string(payload["model_id"], "model_id"),
        tokenizer_id=_require_string(payload["tokenizer_id"], "tokenizer_id"),
        tokenizer_revision=revision,
        vocab_size=_require_integer(payload["vocab_size"], "vocab_size"),
        counts_dtype=_require_string(payload["counts_dtype"], "counts_dtype"),
        counts_sha256=_require_string(payload["counts_sha256"], "counts_sha256"),
        source_manifest_sha256=_require_string(
            payload["source_manifest_sha256"],
            "source_manifest_sha256",
        ),
        exclusion_token_ids=tuple(exclusions),
        num_documents=_require_integer(payload["num_documents"], "num_documents"),
        num_tokens=_require_integer(payload["num_tokens"], "num_tokens"),
    )
    _validate_metadata(metadata)
    return metadata


def load_frequency_table_metadata(
    metadata_path: Path,
) -> tuple[FrequencyTableMetadata, str, Path]:
    """Load a sidecar and verify its metadata-derived artifact identity."""
    metadata_path = Path(metadata_path)
    with metadata_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("frequency-table sidecar must be a JSON object")
    if set(payload) != {"artifact_id", "counts_file", "metadata"}:
        raise ValueError(
            "frequency-table sidecar must contain artifact_id, counts_file, and metadata"
        )
    metadata = _metadata_from_dict(payload["metadata"])
    actual_artifact_id = _artifact_id(metadata)
    recorded_artifact_id = payload["artifact_id"]
    if recorded_artifact_id != actual_artifact_id:
        raise ValueError(
            f"artifact_id mismatch: recorded={recorded_artifact_id!r} "
            f"actual={actual_artifact_id!r}"
        )
    if metadata_path.name != f"{actual_artifact_id}.json":
        raise ValueError(
            f"artifact_id filename mismatch: path={metadata_path.name!r} "
            f"expected={actual_artifact_id + '.json'!r}"
        )
    counts_file = payload["counts_file"]
    if not isinstance(counts_file, str) or Path(counts_file).name != counts_file:
        raise ValueError("counts_file must be a relative filename")
    expected_counts_file = f"{actual_artifact_id}.counts.pt"
    if counts_file != expected_counts_file:
        raise ValueError(
            f"counts_file mismatch: recorded={counts_file!r} "
            f"expected={expected_counts_file!r}"
        )
    return metadata, actual_artifact_id, metadata_path.parent / counts_file


def _safe_torch_load(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_frequency_table(
    metadata_path: Path,
    *,
    expected_model_id: str,
    expected_tokenizer_id: str,
    expected_tokenizer_revision: str | None,
    expected_vocab_size: int,
    expected_exclusion_token_ids: Collection[int],
) -> tuple[torch.Tensor, FrequencyTableMetadata]:
    """Load counts only after integrity and compatibility checks pass."""
    metadata, _, counts_path = load_frequency_table_metadata(metadata_path)
    expected_exclusions = _normalize_exclusion_ids(
        expected_exclusion_token_ids,
        expected_vocab_size,
    )
    checks = {
        "model_id": (metadata.model_id, expected_model_id),
        "tokenizer_id": (metadata.tokenizer_id, expected_tokenizer_id),
        "tokenizer_revision": (
            metadata.tokenizer_revision,
            expected_tokenizer_revision,
        ),
        "vocab_size": (metadata.vocab_size, expected_vocab_size),
        "exclusion_token_ids": (metadata.exclusion_token_ids, expected_exclusions),
    }
    for field_name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{field_name} mismatch: artifact={actual!r} expected={expected!r}"
            )
    if not counts_path.is_file():
        raise ValueError(f"counts file is missing: {counts_path}")
    loaded = _safe_torch_load(counts_path)
    if not isinstance(loaded, torch.Tensor):
        raise ValueError("counts file must contain a torch.Tensor")
    if loaded.dtype != torch.int64:
        raise ValueError(
            f"serialized counts dtype must be torch.int64, got {loaded.dtype}"
        )
    return _validate_counts_against_metadata(loaded, metadata), metadata


def load_frequency_table_from_metrics(
    metrics_path: Path,
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_tokenizer_id: str,
    expected_tokenizer_revision: str | None,
    expected_vocab_size: int,
    expected_exclusion_token_ids: Collection[int],
) -> tuple[torch.Tensor, FrequencyTableMetadata]:
    """Load the exact frequency artifact recorded by upstream calibration."""
    metrics_path = Path(metrics_path)
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError("prediction-set metrics must be a JSON object")
    protocol = metrics.get("protocol")
    reference = protocol.get("frequency_table") if isinstance(protocol, dict) else None
    if not isinstance(reference, dict):
        raise RuntimeError(
            "prediction_set_metrics does not reference a frequency_table artifact; "
            "downstream runs may not rebuild counts"
        )
    metrics_model = metrics.get("model")
    metrics_revision = metrics.get("model_revision")
    if metrics_model != expected_model_id:
        raise ValueError(
            f"model_id mismatch between metrics and runtime: "
            f"metrics={metrics_model!r} runtime={expected_model_id!r}"
        )
    if metrics_revision != expected_model_revision:
        raise ValueError(
            f"model_revision mismatch between metrics and runtime: "
            f"metrics={metrics_revision!r} runtime={expected_model_revision!r}"
        )
    required = {
        "metadata_path",
        "artifact_id",
        "counts_sha256",
        "source_manifest_sha256",
    }
    missing = sorted(required - set(reference))
    if missing:
        raise RuntimeError(
            "frequency_table reference is missing fields: " + ", ".join(missing)
        )
    metadata_path = Path(reference["metadata_path"])
    if not metadata_path.is_absolute():
        metadata_path = metrics_path.parent / metadata_path
    metadata_path = metadata_path.resolve()
    metadata, artifact_id, _ = load_frequency_table_metadata(metadata_path)
    recorded_checks = {
        "artifact_id": (reference["artifact_id"], artifact_id),
        "counts_sha256": (reference["counts_sha256"], metadata.counts_sha256),
        "source_manifest_sha256": (
            reference["source_manifest_sha256"],
            metadata.source_manifest_sha256,
        ),
    }
    for field_name, (recorded, actual) in recorded_checks.items():
        if recorded != actual:
            raise ValueError(
                f"{field_name} mismatch between metrics and artifact: "
                f"metrics={recorded!r} artifact={actual!r}"
            )
    return load_frequency_table(
        metadata_path,
        expected_model_id=expected_model_id,
        expected_tokenizer_id=expected_tokenizer_id,
        expected_tokenizer_revision=expected_tokenizer_revision,
        expected_vocab_size=expected_vocab_size,
        expected_exclusion_token_ids=expected_exclusion_token_ids,
    )
