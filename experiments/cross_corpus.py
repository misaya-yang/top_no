"""Threshold-complete D_freq/evaluation near-duplicate audits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from document_store import bind_split_documents
from splits import (
    MINHASH_IMPLEMENTATION,
    NORMALIZATION_POLICY,
    SPLIT_CONSTRUCTION_VERSION,
    SourceDocument,
    content_sha256,
    load_manifest,
    load_source_documents_jsonl,
    manifest_sha256,
    minhash_signature,
    shingle_hashes,
    source_documents_sha256,
    split_receipt_sha256,
)


CROSS_CORPUS_VERSION = "icml2027-pr1d-v1"
COMPARISON_SCOPE = "frequency-manifest-docs-vs-evaluation-input-documents-v1"
REQUIRED_SHINGLE_SIZE = 13
THRESHOLD_NUMERATOR = 4
THRESHOLD_DENOMINATOR = 5
MANIFEST_PROTOCOL_VERSION = "icml2027-pr1a"


@dataclass(frozen=True)
class CrossCorpusMatch:
    frequency_doc_id: str
    evaluation_doc_id: str
    intersection_size: int
    union_size: int


@dataclass(frozen=True)
class CrossCorpusReceipt:
    protocol_version: str
    comparison_scope: str
    frequency_manifest_sha256: str
    frequency_documents_sha256: str
    frequency_document_count: int
    evaluation_split_receipt_sha256: str
    evaluation_documents_sha256: str
    evaluation_document_count: int
    evaluation_tune_manifest_sha256: str
    evaluation_cal_manifest_sha256: str
    evaluation_test_manifest_sha256: str
    split_construction_schema_version: str
    normalization_policy: str
    shingle_size: int
    jaccard_threshold: float
    threshold_numerator: int
    threshold_denominator: int
    minhash_implementation: str
    minhash_seed: int
    num_perm: int
    lsh_bands: int
    lsh_rows: int
    comparison_transcript_sha256: str
    candidate_pair_count: int
    exact_comparison_count: int
    matches_sha256: str
    match_count: int
    verdict: str


@dataclass(frozen=True)
class CrossCorpusAudit:
    receipt: CrossCorpusReceipt
    matches: tuple[CrossCorpusMatch, ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _matches_sha256(matches: Sequence[CrossCorpusMatch]) -> str:
    return hashlib.sha256(
        _canonical_json([asdict(match) for match in matches])
    ).hexdigest()


def cross_corpus_receipt_sha256(receipt: CrossCorpusReceipt) -> str:
    _validate_receipt(receipt)
    return hashlib.sha256(_canonical_json(asdict(receipt))).hexdigest()


def _bind_frequency_documents(
    manifest_path: Path,
    document_jsonl: Path,
) -> tuple[object, tuple[SourceDocument, ...]]:
    manifest = load_manifest(Path(manifest_path))
    if manifest.role != "freq":
        raise ValueError(f"frequency manifest role must be 'freq', got {manifest.role!r}")
    if manifest.protocol_version != MANIFEST_PROTOCOL_VERSION:
        raise ValueError("frequency manifest protocol_version mismatch")
    documents = load_source_documents_jsonl(Path(document_jsonl))
    by_id = {document.doc_id: document for document in documents}
    manifest_ids = {document.doc_id for document in manifest.documents}
    document_ids = set(by_id)
    if manifest_ids != document_ids:
        missing = sorted(manifest_ids - document_ids)
        extra = sorted(document_ids - manifest_ids)
        raise ValueError(
            "frequency manifest/document IDs mismatch: "
            f"missing={missing!r} extra={extra!r}"
        )
    for recorded in manifest.documents:
        actual_hash = content_sha256(by_id[recorded.doc_id].text)
        if actual_hash != recorded.content_sha256:
            raise ValueError(
                "frequency manifest content_sha256 mismatch: "
                f"doc_id={recorded.doc_id!r}"
            )
    return manifest, documents


def _cross_candidate_rows(
    frequency_sets: Sequence[frozenset[int]],
    evaluation_sets: Sequence[frozenset[int]],
    *,
    threshold: float,
    minhash_seed: int,
    num_perm: int,
    lsh_bands: int,
) -> Iterator[tuple[int, tuple[int, ...]]]:
    if threshold != THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR:
        raise ValueError("cross-corpus candidate threshold must equal 4/5")
    rows = num_perm // lsh_bands
    evaluation_prefix_buckets: dict[int, list[int]] = {}
    evaluation_lsh_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, shingles in enumerate(evaluation_sets):
        ceil_threshold_size = (
            THRESHOLD_NUMERATOR * len(shingles)
            + THRESHOLD_DENOMINATOR
            - 1
        ) // THRESHOLD_DENOMINATOR
        prefix_length = len(shingles) - ceil_threshold_size + 1
        for value in sorted(shingles)[:prefix_length]:
            evaluation_prefix_buckets.setdefault(value, []).append(index)
        signature = minhash_signature(
            shingles,
            minhash_seed=minhash_seed,
            num_perm=num_perm,
        )
        for band in range(lsh_bands):
            key = (band, signature[band * rows : (band + 1) * rows])
            evaluation_lsh_buckets.setdefault(key, []).append(index)

    for frequency_index, shingles in enumerate(frequency_sets):
        candidates: set[int] = set()
        for value in shingles:
            candidates.update(evaluation_prefix_buckets.get(value, ()))
        signature = minhash_signature(
            shingles,
            minhash_seed=minhash_seed,
            num_perm=num_perm,
        )
        for band in range(lsh_bands):
            key = (band, signature[band * rows : (band + 1) * rows])
            candidates.update(evaluation_lsh_buckets.get(key, ()))
        yield frequency_index, tuple(sorted(candidates))


def _find_matches(
    frequency_documents: Sequence[SourceDocument],
    evaluation_documents: Sequence[SourceDocument],
    *,
    shingle_size: int,
    threshold: float,
    minhash_seed: int,
    num_perm: int,
    lsh_bands: int,
) -> tuple[tuple[CrossCorpusMatch, ...], str, int, int]:
    frequency_sets = tuple(
        shingle_hashes(document.text, shingle_size)
        for document in frequency_documents
    )
    evaluation_sets = tuple(
        shingle_hashes(document.text, shingle_size)
        for document in evaluation_documents
    )
    matches = []
    transcript = []
    candidate_pair_count = 0
    for frequency_index, evaluation_indices in _cross_candidate_rows(
        frequency_sets,
        evaluation_sets,
        threshold=threshold,
        minhash_seed=minhash_seed,
        num_perm=num_perm,
        lsh_bands=lsh_bands,
    ):
        candidate_pair_count += len(evaluation_indices)
        for evaluation_index in evaluation_indices:
            left = frequency_sets[frequency_index]
            right = evaluation_sets[evaluation_index]
            if (
                THRESHOLD_DENOMINATOR * min(len(left), len(right))
                < THRESHOLD_NUMERATOR * max(len(left), len(right))
            ):
                continue
            intersection_size = len(left & right)
            union_size = len(left | right)
            transcript.append(
                {
                    "frequency_doc_id": frequency_documents[frequency_index].doc_id,
                    "evaluation_doc_id": evaluation_documents[evaluation_index].doc_id,
                    "frequency_content_sha256": content_sha256(
                        frequency_documents[frequency_index].text
                    ),
                    "evaluation_content_sha256": content_sha256(
                        evaluation_documents[evaluation_index].text
                    ),
                    "intersection_size": intersection_size,
                    "union_size": union_size,
                }
            )
            if (
                THRESHOLD_DENOMINATOR * intersection_size
                < THRESHOLD_NUMERATOR * union_size
            ):
                continue
            matches.append(
                CrossCorpusMatch(
                    frequency_doc_id=frequency_documents[frequency_index].doc_id,
                    evaluation_doc_id=evaluation_documents[evaluation_index].doc_id,
                    intersection_size=intersection_size,
                    union_size=union_size,
                )
            )
    transcript_hash = hashlib.sha256(_canonical_json(transcript)).hexdigest()
    return tuple(matches), transcript_hash, candidate_pair_count, len(transcript)


def audit_cross_corpus(
    *,
    frequency_manifest_path: Path,
    frequency_document_jsonl: Path,
    evaluation_split_receipt_path: Path,
    evaluation_document_jsonl: Path,
    configured_eval_manifests: Mapping[str, Path] | None = None,
) -> CrossCorpusAudit:
    """Recompute a cross-corpus audit from frozen source artifacts."""
    frequency_manifest, frequency_documents = _bind_frequency_documents(
        frequency_manifest_path,
        frequency_document_jsonl,
    )
    bound_evaluation = bind_split_documents(
        evaluation_split_receipt_path,
        evaluation_document_jsonl,
        configured_manifests=configured_eval_manifests,
    )
    if not bound_evaluation.for_role("cal") or not bound_evaluation.for_role("test"):
        raise ValueError("calibration and test manifests must both be non-empty")
    evaluation_receipt = bound_evaluation.split_receipt
    evaluation_documents = bound_evaluation.source_documents
    if evaluation_receipt.shingle_size != REQUIRED_SHINGLE_SIZE:
        raise ValueError(
            f"paper protocol requires shingle_size={REQUIRED_SHINGLE_SIZE}"
        )
    if evaluation_receipt.jaccard_threshold != (
        THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR
    ):
        raise ValueError("paper protocol requires jaccard_threshold=0.8")
    if evaluation_receipt.normalization_policy != NORMALIZATION_POLICY:
        raise ValueError("paper protocol normalization_policy mismatch")
    if evaluation_receipt.minhash_implementation != MINHASH_IMPLEMENTATION:
        raise ValueError("paper protocol requires threshold-complete implementation")
    (
        matches,
        transcript_hash,
        candidate_pair_count,
        exact_comparison_count,
    ) = _find_matches(
        frequency_documents,
        evaluation_documents,
        shingle_size=evaluation_receipt.shingle_size,
        threshold=evaluation_receipt.jaccard_threshold,
        minhash_seed=evaluation_receipt.minhash_seed,
        num_perm=evaluation_receipt.num_perm,
        lsh_bands=evaluation_receipt.lsh_bands,
    )
    eval_manifest_hashes = dict(evaluation_receipt.manifest_sha256s)
    receipt = CrossCorpusReceipt(
        protocol_version=CROSS_CORPUS_VERSION,
        comparison_scope=COMPARISON_SCOPE,
        frequency_manifest_sha256=manifest_sha256(frequency_manifest),
        frequency_documents_sha256=source_documents_sha256(frequency_documents),
        frequency_document_count=len(frequency_documents),
        evaluation_split_receipt_sha256=split_receipt_sha256(evaluation_receipt),
        evaluation_documents_sha256=evaluation_receipt.input_documents_sha256,
        evaluation_document_count=len(evaluation_documents),
        evaluation_tune_manifest_sha256=eval_manifest_hashes["tune"],
        evaluation_cal_manifest_sha256=eval_manifest_hashes["cal"],
        evaluation_test_manifest_sha256=eval_manifest_hashes["test"],
        split_construction_schema_version=evaluation_receipt.construction_schema_version,
        normalization_policy=evaluation_receipt.normalization_policy,
        shingle_size=evaluation_receipt.shingle_size,
        jaccard_threshold=evaluation_receipt.jaccard_threshold,
        threshold_numerator=THRESHOLD_NUMERATOR,
        threshold_denominator=THRESHOLD_DENOMINATOR,
        minhash_implementation=evaluation_receipt.minhash_implementation,
        minhash_seed=evaluation_receipt.minhash_seed,
        num_perm=evaluation_receipt.num_perm,
        lsh_bands=evaluation_receipt.lsh_bands,
        lsh_rows=evaluation_receipt.lsh_rows,
        comparison_transcript_sha256=transcript_hash,
        candidate_pair_count=candidate_pair_count,
        exact_comparison_count=exact_comparison_count,
        matches_sha256=_matches_sha256(matches),
        match_count=len(matches),
        verdict="pass" if not matches else "fail",
    )
    return CrossCorpusAudit(receipt=receipt, matches=matches)


def _validate_receipt(receipt: CrossCorpusReceipt) -> None:
    if receipt.protocol_version != CROSS_CORPUS_VERSION:
        raise ValueError("unsupported cross-corpus protocol_version")
    if receipt.comparison_scope != COMPARISON_SCOPE:
        raise ValueError("unsupported comparison_scope")
    for field_name in (
        "frequency_manifest_sha256",
        "frequency_documents_sha256",
        "evaluation_split_receipt_sha256",
        "evaluation_documents_sha256",
        "evaluation_tune_manifest_sha256",
        "evaluation_cal_manifest_sha256",
        "evaluation_test_manifest_sha256",
        "comparison_transcript_sha256",
        "matches_sha256",
    ):
        value = getattr(receipt, field_name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{field_name} must be a canonical SHA-256 string")
    if receipt.split_construction_schema_version != SPLIT_CONSTRUCTION_VERSION:
        raise ValueError("split construction schema mismatch")
    for field_name in (
        "frequency_document_count",
        "evaluation_document_count",
        "shingle_size",
        "threshold_numerator",
        "threshold_denominator",
        "minhash_seed",
        "num_perm",
        "lsh_bands",
        "lsh_rows",
        "candidate_pair_count",
        "exact_comparison_count",
        "match_count",
    ):
        value = getattr(receipt, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
    if receipt.num_perm != receipt.lsh_bands * receipt.lsh_rows:
        raise ValueError("invalid MinHash-LSH layout")
    if (
        receipt.frequency_document_count <= 0
        or receipt.evaluation_document_count <= 0
        or receipt.candidate_pair_count < 0
        or receipt.exact_comparison_count < 0
        or receipt.match_count < 0
    ):
        raise ValueError("cross-corpus counts are invalid")
    if not (
        receipt.match_count
        <= receipt.exact_comparison_count
        <= receipt.candidate_pair_count
    ):
        raise ValueError(
            "match_count cannot exceed exact comparisons or candidate pairs"
        )
    if (
        isinstance(receipt.jaccard_threshold, bool)
        or not isinstance(receipt.jaccard_threshold, (int, float))
        or not 0.0 < receipt.jaccard_threshold <= 1.0
    ):
        raise ValueError("jaccard_threshold must be in (0, 1]")
    if (
        receipt.shingle_size != REQUIRED_SHINGLE_SIZE
        or receipt.jaccard_threshold != THRESHOLD_NUMERATOR / THRESHOLD_DENOMINATOR
        or receipt.threshold_numerator != THRESHOLD_NUMERATOR
        or receipt.threshold_denominator != THRESHOLD_DENOMINATOR
        or receipt.normalization_policy != NORMALIZATION_POLICY
        or receipt.minhash_implementation != MINHASH_IMPLEMENTATION
    ):
        raise ValueError("cross-corpus receipt weakens the fixed paper protocol")
    expected_verdict = "pass" if receipt.match_count == 0 else "fail"
    if receipt.verdict != expected_verdict:
        raise ValueError("cross-corpus verdict does not match match_count")


def save_cross_corpus_audit(audit: CrossCorpusAudit, path: Path) -> Path:
    _validate_matches(audit.matches)
    if audit.receipt.match_count != len(audit.matches):
        raise ValueError("match_count does not match serialized matches")
    if audit.receipt.matches_sha256 != _matches_sha256(audit.matches):
        raise ValueError("matches_sha256 mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_id": cross_corpus_receipt_sha256(audit.receipt),
        "receipt": asdict(audit.receipt),
        "matches": [asdict(match) for match in audit.matches],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_cross_corpus_audit(path: Path) -> CrossCorpusAudit:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != {"artifact_id", "receipt", "matches"}:
        raise ValueError("cross-corpus wrapper has unexpected fields")
    try:
        receipt = CrossCorpusReceipt(**payload["receipt"])
        matches = tuple(CrossCorpusMatch(**item) for item in payload["matches"])
    except (TypeError, KeyError) as exc:
        raise ValueError(f"malformed cross-corpus audit: {exc}") from exc
    artifact_id = cross_corpus_receipt_sha256(receipt)
    if payload["artifact_id"] != artifact_id:
        raise ValueError("artifact_id mismatch for cross-corpus audit")
    audit = CrossCorpusAudit(receipt=receipt, matches=matches)
    _validate_matches(matches)
    if receipt.match_count != len(matches):
        raise ValueError("match_count does not match serialized matches")
    if receipt.matches_sha256 != _matches_sha256(matches):
        raise ValueError("matches_sha256 mismatch")
    return audit


def _validate_matches(matches: Sequence[CrossCorpusMatch]) -> None:
    seen: set[tuple[str, str]] = set()
    for match in matches:
        if (
            not isinstance(match.frequency_doc_id, str)
            or not match.frequency_doc_id
            or not isinstance(match.evaluation_doc_id, str)
            or not match.evaluation_doc_id
        ):
            raise ValueError("cross-corpus match doc IDs must be non-empty strings")
        for field_name in ("intersection_size", "union_size"):
            value = getattr(match, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"cross-corpus match {field_name} must be positive")
        if match.intersection_size > match.union_size:
            raise ValueError("cross-corpus match intersection exceeds union")
        if (
            THRESHOLD_DENOMINATOR * match.intersection_size
            < THRESHOLD_NUMERATOR * match.union_size
        ):
            raise ValueError("serialized cross-corpus match is below threshold")
        pair = (match.frequency_doc_id, match.evaluation_doc_id)
        if pair in seen:
            raise ValueError("duplicate serialized cross-corpus match")
        seen.add(pair)


def validate_cross_corpus_audit(
    path: Path,
    *,
    frequency_manifest_path: Path,
    frequency_document_jsonl: Path,
    evaluation_split_receipt_path: Path,
    evaluation_document_jsonl: Path,
    configured_eval_manifests: Mapping[str, Path] | None = None,
) -> CrossCorpusReceipt:
    """Load a receipt, then independently recompute every bound comparison."""
    loaded = load_cross_corpus_audit(path)
    recomputed = audit_cross_corpus(
        frequency_manifest_path=frequency_manifest_path,
        frequency_document_jsonl=frequency_document_jsonl,
        evaluation_split_receipt_path=evaluation_split_receipt_path,
        evaluation_document_jsonl=evaluation_document_jsonl,
        configured_eval_manifests=configured_eval_manifests,
    )
    if loaded != recomputed:
        raise ValueError("recomputed cross-corpus audit does not match saved artifact")
    if recomputed.receipt.verdict != "pass":
        raise ValueError(
            "cross-corpus near-duplicate audit failed: "
            f"match_count={recomputed.receipt.match_count}"
        )
    return recomputed.receipt
