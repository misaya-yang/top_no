"""Document-manifest contracts for paper-grade experiment splits.

PR-1a validates provenance manifests and their disjointness. Construction of
deduplicated, cluster-level tune/calibration/test splits lands in PR-1b.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ManifestDocument:
    doc_id: str
    content_sha256: str
    cluster_id: str


@dataclass(frozen=True)
class DocumentManifest:
    protocol_version: str
    role: str
    source: str
    documents: tuple[ManifestDocument, ...]


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_manifest(manifest: DocumentManifest) -> None:
    _require_nonempty(manifest.protocol_version, "protocol_version")
    _require_nonempty(manifest.role, "role")
    _require_nonempty(manifest.source, "source")

    for field_name in ("doc_id", "content_sha256", "cluster_id"):
        seen: set[str] = set()
        for document in manifest.documents:
            value = getattr(document, field_name)
            _require_nonempty(value, field_name)
            if value in seen:
                raise ValueError(
                    f"manifest role={manifest.role!r} contains duplicate "
                    f"{field_name}: {value!r}"
                )
            seen.add(value)
    for document in manifest.documents:
        if re.fullmatch(r"[0-9a-f]{64}", document.content_sha256) is None:
            raise ValueError(
                "content_sha256 must be a canonical 64-character lowercase "
                "hexadecimal digest"
            )


def _canonical_manifest_dict(manifest: DocumentManifest) -> dict[str, object]:
    _validate_manifest(manifest)
    documents = sorted(
        (asdict(document) for document in manifest.documents),
        key=lambda row: (row["doc_id"], row["content_sha256"], row["cluster_id"]),
    )
    return {
        "protocol_version": manifest.protocol_version,
        "role": manifest.role,
        "source": manifest.source,
        "documents": documents,
    }


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def manifest_sha256(manifest: DocumentManifest) -> str:
    """Return an order-independent SHA-256 identity for a manifest."""
    return hashlib.sha256(_canonical_json(_canonical_manifest_dict(manifest))).hexdigest()


def save_manifest(manifest: DocumentManifest, path: Path) -> Path:
    """Write a canonical manifest wrapper and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_sha256": manifest_sha256(manifest),
        "manifest": _canonical_manifest_dict(manifest),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(path: Path) -> DocumentManifest:
    """Load a manifest and verify its recorded content hash."""
    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest wrapper at {path} must be a JSON object")
    if set(payload) != {"manifest_sha256", "manifest"}:
        raise ValueError(
            f"Manifest wrapper at {path} must contain only manifest_sha256 and manifest"
        )
    raw_manifest = payload["manifest"]
    if not isinstance(raw_manifest, dict):
        raise ValueError(f"manifest at {path} must be a JSON object")
    try:
        documents = tuple(
            ManifestDocument(
                doc_id=row["doc_id"],
                content_sha256=row["content_sha256"],
                cluster_id=row["cluster_id"],
            )
            for row in raw_manifest["documents"]
        )
        manifest = DocumentManifest(
            protocol_version=raw_manifest["protocol_version"],
            role=raw_manifest["role"],
            source=raw_manifest["source"],
            documents=documents,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed manifest at {path}: {exc}") from exc

    actual_hash = manifest_sha256(manifest)
    recorded_hash = payload["manifest_sha256"]
    if recorded_hash != actual_hash:
        raise ValueError(
            f"manifest_sha256 mismatch for {path}: recorded={recorded_hash!r} "
            f"actual={actual_hash!r}"
        )
    return manifest


def assert_pairwise_disjoint(manifests: Mapping[str, DocumentManifest]) -> None:
    """Raise if any document, content, or cluster identity crosses manifests."""
    indexed: dict[str, dict[str, str]] = {
        "doc_id": {},
        "content_sha256": {},
        "cluster_id": {},
    }
    for label, manifest in manifests.items():
        _validate_manifest(manifest)
        for document in manifest.documents:
            for field_name, owners in indexed.items():
                value = getattr(document, field_name)
                previous = owners.get(value)
                if previous is not None and previous != label:
                    raise ValueError(
                        f"{field_name} intersection between manifests "
                        f"{previous!r} and {label!r}: {value!r}"
                    )
                owners[value] = label
