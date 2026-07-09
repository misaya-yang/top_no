"""Bind PR-1b split artifacts to exact frozen source text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from splits import (
    content_sha256,
    load_manifest,
    load_source_documents_jsonl,
    load_split_receipt,
    manifest_sha256,
    source_documents_sha256,
    split_receipt_sha256,
)


@dataclass(frozen=True)
class BoundDocument:
    role: str
    doc_id: str
    content_sha256: str
    cluster_id: str
    text: str


@dataclass(frozen=True)
class BoundSplitDocuments:
    receipt_sha256: str
    input_documents_sha256: str
    source: str
    source_snapshot_sha256: str
    cluster_namespace_sha256: str
    documents_by_role: tuple[tuple[str, tuple[BoundDocument, ...]], ...]

    def for_role(self, role: str) -> tuple[BoundDocument, ...]:
        for stored_role, documents in self.documents_by_role:
            if stored_role == role:
                return documents
        raise KeyError(role)


def bind_split_documents(
    receipt_path: Path,
    document_jsonl: Path,
    *,
    configured_manifests: Mapping[str, Path] | None = None,
) -> BoundSplitDocuments:
    """Verify and bind every retained representative to one exact text row."""
    receipt, receipt_manifests = load_split_receipt(Path(receipt_path))
    documents = load_source_documents_jsonl(Path(document_jsonl))
    actual_input_hash = source_documents_sha256(documents)
    if actual_input_hash != receipt.input_documents_sha256:
        raise ValueError(
            "input_documents_sha256 mismatch: "
            f"receipt={receipt.input_documents_sha256!r} actual={actual_input_hash!r}"
        )

    if configured_manifests is not None:
        if set(configured_manifests) != {"tune", "cal", "test"}:
            raise ValueError(
                "configured_manifests must contain tune, cal, and test"
            )
        expected_hashes = dict(receipt.manifest_sha256s)
        for role in ("tune", "cal", "test"):
            configured = load_manifest(Path(configured_manifests[role]))
            actual_hash = manifest_sha256(configured)
            if actual_hash != expected_hashes[role]:
                raise ValueError(
                    "configured manifest hash mismatch: "
                    f"role={role!r} receipt={expected_hashes[role]!r} "
                    f"actual={actual_hash!r}"
                )

    source_by_id = {item.doc_id: item.text for item in documents}
    bound_roles = []
    for role in ("tune", "cal", "test"):
        bound_documents = []
        for manifest_document in receipt_manifests[role].documents:
            text = source_by_id.get(manifest_document.doc_id)
            if text is None:
                raise ValueError(
                    f"manifest representative is missing: {manifest_document.doc_id!r}"
                )
            actual_content_hash = content_sha256(text)
            if actual_content_hash != manifest_document.content_sha256:
                raise ValueError(
                    "manifest representative content_sha256 mismatch: "
                    f"doc_id={manifest_document.doc_id!r}"
                )
            bound_documents.append(
                BoundDocument(
                    role=role,
                    doc_id=manifest_document.doc_id,
                    content_sha256=manifest_document.content_sha256,
                    cluster_id=manifest_document.cluster_id,
                    text=text,
                )
            )
        bound_roles.append((role, tuple(bound_documents)))

    return BoundSplitDocuments(
        receipt_sha256=split_receipt_sha256(receipt),
        input_documents_sha256=actual_input_hash,
        source=receipt.source,
        source_snapshot_sha256=receipt.source_snapshot_sha256,
        cluster_namespace_sha256=receipt.cluster_namespace_sha256,
        documents_by_role=tuple(bound_roles),
    )
