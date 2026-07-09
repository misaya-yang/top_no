"""Document-manifest contracts for paper-grade experiment splits.

PR-1a validates provenance manifests and their disjointness. PR-1b adds
deterministic near-duplicate clustering, cluster-level tune/calibration/test
assignment, construction receipts, and document-position selection. Binding
those artifacts to evaluator text is deliberately deferred to PR-1c.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence


SPLIT_CONSTRUCTION_VERSION = "icml2027-pr1b-v1"
MINHASH_IMPLEMENTATION = "stdlib-splitmix64-lsh-plus-exact-prefix-v1"
NORMALIZATION_POLICY = "unicode-nfkc-casefold-whitespace-short-doc-fallback-v2"
REPRESENTATIVE_POLICY = "minimum-content-sha256-then-doc-id-v1"
CLUSTER_ID_POLICY = "sha256-namespace-plus-sorted-unique-content-hashes-v1"
SPLIT_HASH_POLICY = "sha256-domain-separated-representative-doc-id-mod-100-v1"
SPLIT_BANDS = (("tune", 0, 40), ("cal", 40, 65), ("test", 65, 100))


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


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str


@dataclass(frozen=True)
class DocumentCluster:
    cluster_id: str
    representative: ManifestDocument
    members: tuple[ManifestDocument, ...]

    @property
    def representative_doc_id(self) -> str:
        return self.representative.doc_id

    @property
    def member_doc_ids(self) -> tuple[str, ...]:
        return tuple(member.doc_id for member in self.members)


@dataclass(frozen=True)
class SelectedPosition:
    doc_id: str
    cluster_id: str
    target_index: int
    evidence_grade: str


@dataclass(frozen=True)
class SplitBuildReceipt:
    construction_schema_version: str
    source: str
    source_snapshot_sha256: str
    cluster_namespace_sha256: str
    input_documents_sha256: str
    normalization_policy: str
    shingle_size: int
    jaccard_threshold: float
    minhash_implementation: str
    minhash_seed: int
    num_perm: int
    lsh_bands: int
    lsh_rows: int
    representative_policy: str
    cluster_id_policy: str
    split_hash_policy: str
    global_salt: str
    split_bands: tuple[tuple[str, int, int], ...]
    cluster_manifest_sha256: str
    manifest_sha256s: tuple[tuple[str, str], ...]
    num_input_documents: int
    num_clusters: int


@dataclass(frozen=True)
class SplitBuildArtifacts:
    clusters: tuple[DocumentCluster, ...]
    manifests: Mapping[str, DocumentManifest]
    receipt: SplitBuildReceipt


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


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"{field_name} must be a canonical 64-character lowercase SHA-256"
        )


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(normalized.split())


def _shingle_hashes(text: str, shingle_size: int) -> frozenset[int]:
    tokens = _normalize_text(text)
    if not tokens:
        raise ValueError("source document text must contain a non-whitespace token")
    width = min(shingle_size, len(tokens))
    shingles = []
    for index in range(len(tokens) - width + 1):
        payload = "\x1f".join(tokens[index : index + width]).encode("utf-8")
        shingles.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    return frozenset(shingles)


def _mix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _minhash_signature(
    shingles: frozenset[int],
    *,
    minhash_seed: int,
    num_perm: int,
) -> tuple[int, ...]:
    seeds = tuple(_mix64(minhash_seed + index) for index in range(num_perm))
    return tuple(min(_mix64(value ^ seed) for value in shingles) for seed in seeds)


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    return len(left & right) / len(left | right)


def _exact_prefix_candidates(
    shingle_sets: Sequence[frozenset[int]],
    threshold: float,
) -> set[tuple[int, int]]:
    """Return a no-false-negative candidate superset for Jaccard thresholding.

    For an earlier set A, its suffix after the indexed prefix contains fewer
    than ceil(threshold * |A|) elements. Any later B with J(A, B) >= threshold
    must therefore contain at least one indexed prefix element from A.
    """
    inverted: dict[int, list[int]] = {}
    candidates: set[tuple[int, int]] = set()
    for index, shingles in enumerate(shingle_sets):
        for value in shingles:
            candidates.update((previous, index) for previous in inverted.get(value, ()))
        prefix_length = len(shingles) - math.ceil(threshold * len(shingles)) + 1
        for value in sorted(shingles)[:prefix_length]:
            inverted.setdefault(value, []).append(index)
    return candidates


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _validated_source_documents(
    documents: Sequence[SourceDocument],
) -> tuple[SourceDocument, ...]:
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise ValueError("documents must be a sequence of SourceDocument values")
    ordered = []
    seen_doc_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, SourceDocument):
            raise ValueError("documents must contain SourceDocument values")
        _require_nonempty(document.doc_id, "doc_id")
        if not isinstance(document.text, str) or not document.text.strip():
            raise ValueError("source document text must be a non-empty string")
        if document.doc_id in seen_doc_ids:
            raise ValueError(f"duplicate doc_id: {document.doc_id!r}")
        seen_doc_ids.add(document.doc_id)
        ordered.append(document)
    if not ordered:
        raise ValueError("documents must not be empty")
    return tuple(sorted(ordered, key=lambda item: (item.doc_id, _content_sha256(item.text))))


def _cluster_id(
    content_hashes: Collection[str],
    cluster_namespace_sha256: str,
) -> str:
    payload = {
        "cluster_id_policy": CLUSTER_ID_POLICY,
        "cluster_namespace_sha256": cluster_namespace_sha256,
        "content_sha256s": sorted(set(content_hashes)),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def cluster_documents_minhash_lsh(
    documents: Sequence[SourceDocument],
    *,
    cluster_namespace_sha256: str,
    shingle_size: int = 13,
    jaccard_threshold: float = 0.8,
    minhash_seed: int = 1729,
    num_perm: int = 100,
    lsh_bands: int = 20,
) -> tuple[DocumentCluster, ...]:
    """Cluster exact/near duplicates using deterministic MinHash-LSH candidates."""
    ordered = _validated_source_documents(documents)
    _require_sha256(cluster_namespace_sha256, "cluster_namespace_sha256")
    if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size <= 0:
        raise ValueError("shingle_size must be a positive integer")
    if not isinstance(jaccard_threshold, (int, float)) or isinstance(jaccard_threshold, bool):
        raise ValueError("jaccard_threshold must be numeric")
    threshold = float(jaccard_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    for name, value in (
        ("minhash_seed", minhash_seed),
        ("num_perm", num_perm),
        ("lsh_bands", lsh_bands),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if num_perm <= 0 or lsh_bands <= 0 or num_perm % lsh_bands:
        raise ValueError("num_perm must be positive and divisible by lsh_bands")

    content_hashes = tuple(_content_sha256(item.text) for item in ordered)
    shingle_sets = tuple(_shingle_hashes(item.text, shingle_size) for item in ordered)
    signatures = tuple(
        _minhash_signature(
            shingles,
            minhash_seed=minhash_seed,
            num_perm=num_perm,
        )
        for shingles in shingle_sets
    )
    union_find = _UnionFind(len(ordered))

    exact_owners: dict[str, int] = {}
    for index, content_hash in enumerate(content_hashes):
        previous = exact_owners.get(content_hash)
        if previous is None:
            exact_owners[content_hash] = index
        else:
            union_find.union(previous, index)

    rows = num_perm // lsh_bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    candidates: set[tuple[int, int]] = set()
    for index, signature in enumerate(signatures):
        for band in range(lsh_bands):
            key = (band, signature[band * rows : (band + 1) * rows])
            owners = buckets.setdefault(key, [])
            candidates.update((previous, index) for previous in owners)
            owners.append(index)
    candidates.update(_exact_prefix_candidates(shingle_sets, threshold))
    for left, right in sorted(candidates):
        if content_hashes[left] == content_hashes[right]:
            continue
        if (
            min(len(shingle_sets[left]), len(shingle_sets[right]))
            / max(len(shingle_sets[left]), len(shingle_sets[right]))
            < threshold
        ):
            continue
        if _jaccard(shingle_sets[left], shingle_sets[right]) >= threshold:
            union_find.union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        components.setdefault(union_find.find(index), []).append(index)

    clusters = []
    for member_indices in components.values():
        component_hashes = [content_hashes[index] for index in member_indices]
        cluster_id = _cluster_id(component_hashes, cluster_namespace_sha256)
        member_rows = tuple(
            sorted(
                (
                    ManifestDocument(
                        doc_id=ordered[index].doc_id,
                        content_sha256=content_hashes[index],
                        cluster_id=cluster_id,
                    )
                    for index in member_indices
                ),
                key=lambda item: item.doc_id,
            )
        )
        representative = min(
            member_rows,
            key=lambda item: (item.content_sha256, item.doc_id),
        )
        clusters.append(
            DocumentCluster(
                cluster_id=cluster_id,
                representative=representative,
                members=member_rows,
            )
        )
    return tuple(sorted(clusters, key=lambda item: item.cluster_id))


def split_role_for_cluster(representative_doc_id: str, global_salt: str) -> str:
    """Assign a cluster via its canonical representative document ID."""
    _require_nonempty(representative_doc_id, "representative_doc_id")
    _require_nonempty(global_salt, "global_salt")
    payload = (
        b"icml2027-split-v1\x00"
        + global_salt.encode()
        + b"\x00"
        + representative_doc_id.encode()
    )
    band = int.from_bytes(hashlib.sha256(payload).digest(), "big") % 100
    for role, lower, upper in SPLIT_BANDS:
        if lower <= band < upper:
            return role
    raise AssertionError(f"unreachable split band: {band}")


def _input_documents_sha256(documents: Sequence[SourceDocument]) -> str:
    rows = [
        {"doc_id": item.doc_id, "content_sha256": _content_sha256(item.text)}
        for item in _validated_source_documents(documents)
    ]
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _canonical_cluster_manifest(clusters: Sequence[DocumentCluster]) -> dict[str, object]:
    rows = []
    for cluster in sorted(clusters, key=lambda item: item.cluster_id):
        members = sorted(
            (
                {"doc_id": item.doc_id, "content_sha256": item.content_sha256}
                for item in cluster.members
            ),
            key=lambda item: (item["content_sha256"], item["doc_id"]),
        )
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "representative_doc_id": cluster.representative.doc_id,
                "members": members,
            }
        )
    return {
        "construction_schema_version": SPLIT_CONSTRUCTION_VERSION,
        "clusters": rows,
    }


def _cluster_manifest_sha256(clusters: Sequence[DocumentCluster]) -> str:
    return hashlib.sha256(_canonical_json(_canonical_cluster_manifest(clusters))).hexdigest()


def build_split_artifacts(
    documents: Sequence[SourceDocument],
    *,
    source: str,
    source_snapshot_sha256: str,
    global_salt: str,
    cluster_namespace_sha256: str | None = None,
    shingle_size: int = 13,
    jaccard_threshold: float = 0.8,
    minhash_seed: int = 1729,
    num_perm: int = 100,
    lsh_bands: int = 20,
) -> SplitBuildArtifacts:
    """Build representative role manifests and their construction receipt."""
    _require_nonempty(source, "source")
    _require_nonempty(global_salt, "global_salt")
    _require_sha256(source_snapshot_sha256, "source_snapshot_sha256")
    namespace = cluster_namespace_sha256 or source_snapshot_sha256
    _require_sha256(namespace, "cluster_namespace_sha256")
    ordered = _validated_source_documents(documents)
    clusters = cluster_documents_minhash_lsh(
        ordered,
        cluster_namespace_sha256=namespace,
        shingle_size=shingle_size,
        jaccard_threshold=jaccard_threshold,
        minhash_seed=minhash_seed,
        num_perm=num_perm,
        lsh_bands=lsh_bands,
    )
    documents_by_role: dict[str, list[ManifestDocument]] = {
        "tune": [],
        "cal": [],
        "test": [],
    }
    for cluster in clusters:
        role = split_role_for_cluster(cluster.representative.doc_id, global_salt)
        documents_by_role[role].append(cluster.representative)
    manifests = {
        role: DocumentManifest(
            protocol_version="icml2027-pr1a",
            role=role,
            source=source,
            documents=tuple(sorted(rows, key=lambda item: item.doc_id)),
        )
        for role, rows in documents_by_role.items()
    }
    assert_pairwise_disjoint(manifests)
    rows_per_band = num_perm // lsh_bands
    receipt = SplitBuildReceipt(
        construction_schema_version=SPLIT_CONSTRUCTION_VERSION,
        source=source,
        source_snapshot_sha256=source_snapshot_sha256,
        cluster_namespace_sha256=namespace,
        input_documents_sha256=_input_documents_sha256(ordered),
        normalization_policy=NORMALIZATION_POLICY,
        shingle_size=shingle_size,
        jaccard_threshold=float(jaccard_threshold),
        minhash_implementation=MINHASH_IMPLEMENTATION,
        minhash_seed=minhash_seed,
        num_perm=num_perm,
        lsh_bands=lsh_bands,
        lsh_rows=rows_per_band,
        representative_policy=REPRESENTATIVE_POLICY,
        cluster_id_policy=CLUSTER_ID_POLICY,
        split_hash_policy=SPLIT_HASH_POLICY,
        global_salt=global_salt,
        split_bands=SPLIT_BANDS,
        cluster_manifest_sha256=_cluster_manifest_sha256(clusters),
        manifest_sha256s=tuple(
            (role, manifest_sha256(manifests[role]))
            for role in ("tune", "cal", "test")
        ),
        num_input_documents=len(ordered),
        num_clusters=len(clusters),
    )
    return SplitBuildArtifacts(clusters=clusters, manifests=manifests, receipt=receipt)


def _validate_receipt(receipt: SplitBuildReceipt) -> None:
    if receipt.construction_schema_version != SPLIT_CONSTRUCTION_VERSION:
        raise ValueError("unsupported construction_schema_version")
    _require_nonempty(receipt.source, "source")
    _require_sha256(receipt.source_snapshot_sha256, "source_snapshot_sha256")
    _require_sha256(receipt.cluster_namespace_sha256, "cluster_namespace_sha256")
    _require_sha256(receipt.input_documents_sha256, "input_documents_sha256")
    _require_sha256(receipt.cluster_manifest_sha256, "cluster_manifest_sha256")
    if receipt.normalization_policy != NORMALIZATION_POLICY:
        raise ValueError("unsupported normalization_policy")
    if receipt.minhash_implementation != MINHASH_IMPLEMENTATION:
        raise ValueError("unsupported minhash_implementation")
    if receipt.representative_policy != REPRESENTATIVE_POLICY:
        raise ValueError("unsupported representative_policy")
    if receipt.cluster_id_policy != CLUSTER_ID_POLICY:
        raise ValueError("unsupported cluster_id_policy")
    if receipt.split_hash_policy != SPLIT_HASH_POLICY:
        raise ValueError("unsupported split_hash_policy")
    if receipt.split_bands != SPLIT_BANDS:
        raise ValueError("split_bands mismatch")
    _require_nonempty(receipt.global_salt, "global_salt")
    for field_name in (
        "shingle_size",
        "minhash_seed",
        "num_perm",
        "lsh_bands",
        "lsh_rows",
        "num_input_documents",
        "num_clusters",
    ):
        value = getattr(receipt, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
    if receipt.shingle_size <= 0 or receipt.num_perm <= 0 or receipt.lsh_bands <= 0:
        raise ValueError("invalid MinHash-LSH dimensions")
    if receipt.num_perm != receipt.lsh_bands * receipt.lsh_rows:
        raise ValueError("num_perm must equal lsh_bands * lsh_rows")
    if (
        isinstance(receipt.jaccard_threshold, bool)
        or not isinstance(receipt.jaccard_threshold, (int, float))
        or not 0.0 < receipt.jaccard_threshold <= 1.0
    ):
        raise ValueError("jaccard_threshold must be in (0, 1]")
    if receipt.num_input_documents <= 0 or receipt.num_clusters <= 0:
        raise ValueError("receipt counts must be positive")
    if receipt.num_clusters > receipt.num_input_documents:
        raise ValueError("num_clusters cannot exceed num_input_documents")
    if len(receipt.manifest_sha256s) != 3:
        raise ValueError("manifest_sha256s must contain exactly three roles")
    if tuple(role for role, _ in receipt.manifest_sha256s) != (
        "tune",
        "cal",
        "test",
    ):
        raise ValueError("manifest_sha256s must use canonical role order")
    manifest_hashes = dict(receipt.manifest_sha256s)
    if set(manifest_hashes) != {"tune", "cal", "test"}:
        raise ValueError("manifest_sha256s must contain tune, cal, and test")
    for role, value in receipt.manifest_sha256s:
        _require_sha256(value, f"manifest_sha256s[{role}]")


def split_receipt_sha256(receipt: SplitBuildReceipt) -> str:
    """Return the canonical construction receipt identity."""
    _validate_receipt(receipt)
    return hashlib.sha256(_canonical_json(asdict(receipt))).hexdigest()


def _safe_artifact_filename(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"unsafe {field_name}: {value!r}")
    return value


def _resolved_artifact_path(
    parent: Path,
    value: object,
    field_name: str,
) -> Path:
    filename = _safe_artifact_filename(value, field_name)
    resolved_parent = parent.resolve()
    resolved_path = (parent / filename).resolve()
    if resolved_path.parent != resolved_parent:
        raise ValueError(f"unsafe {field_name}: path escapes artifact directory")
    return resolved_path


def _validate_cluster_manifest_payload(
    payload: object,
    receipt: SplitBuildReceipt,
) -> set[tuple[str, str, str]]:
    if not isinstance(payload, dict) or set(payload) != {
        "construction_schema_version",
        "clusters",
    }:
        raise ValueError("cluster manifest has unexpected fields")
    if payload["construction_schema_version"] != SPLIT_CONSTRUCTION_VERSION:
        raise ValueError("cluster manifest construction_schema_version mismatch")
    raw_clusters = payload["clusters"]
    if not isinstance(raw_clusters, list):
        raise ValueError("cluster manifest clusters must be a list")
    if len(raw_clusters) != receipt.num_clusters:
        raise ValueError("cluster manifest num_clusters mismatch")
    seen_clusters: set[str] = set()
    seen_doc_ids: set[str] = set()
    representatives: set[tuple[str, str, str]] = set()
    member_count = 0
    for raw_cluster in raw_clusters:
        if not isinstance(raw_cluster, dict) or set(raw_cluster) != {
            "cluster_id",
            "representative_doc_id",
            "members",
        }:
            raise ValueError("cluster manifest row has unexpected fields")
        cluster_id = raw_cluster["cluster_id"]
        representative_doc_id = raw_cluster["representative_doc_id"]
        _require_sha256(cluster_id, "cluster_id")
        _require_nonempty(representative_doc_id, "representative_doc_id")
        if cluster_id in seen_clusters:
            raise ValueError(f"duplicate cluster_id: {cluster_id!r}")
        seen_clusters.add(cluster_id)
        raw_members = raw_cluster["members"]
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError("cluster members must be a non-empty list")
        members = []
        for raw_member in raw_members:
            if not isinstance(raw_member, dict) or set(raw_member) != {
                "doc_id",
                "content_sha256",
            }:
                raise ValueError("cluster member has unexpected fields")
            doc_id = raw_member["doc_id"]
            content_hash = raw_member["content_sha256"]
            _require_nonempty(doc_id, "doc_id")
            _require_sha256(content_hash, "content_sha256")
            if doc_id in seen_doc_ids:
                raise ValueError(f"duplicate cluster member doc_id: {doc_id!r}")
            seen_doc_ids.add(doc_id)
            members.append((doc_id, content_hash))
        member_count += len(members)
        member_by_id = dict(members)
        if representative_doc_id not in member_by_id:
            raise ValueError("representative_doc_id is not a cluster member")
        expected_representative = min(
            members,
            key=lambda item: (item[1], item[0]),
        )[0]
        if representative_doc_id != expected_representative:
            raise ValueError("representative_doc_id violates representative_policy")
        expected_cluster_id = _cluster_id(
            member_by_id.values(),
            receipt.cluster_namespace_sha256,
        )
        if cluster_id != expected_cluster_id:
            raise ValueError("cluster_id does not match receipt namespace and members")
        representatives.add(
            (representative_doc_id, member_by_id[representative_doc_id], cluster_id)
        )
    if member_count != receipt.num_input_documents:
        raise ValueError("cluster manifest num_input_documents mismatch")
    return representatives


def _validate_split_role_assignments(
    manifests: Mapping[str, DocumentManifest],
    receipt: SplitBuildReceipt,
) -> None:
    for role in ("tune", "cal", "test"):
        manifest = manifests[role]
        for document in manifest.documents:
            expected_role = split_role_for_cluster(
                document.doc_id,
                receipt.global_salt,
            )
            if expected_role != role:
                raise ValueError(
                    "split role mismatch: "
                    f"doc_id={document.doc_id!r} recorded={role!r} "
                    f"expected={expected_role!r}"
                )


def save_split_artifacts(result: SplitBuildArtifacts, output_dir: Path) -> Path:
    """Save role manifests, cluster membership, and a canonical receipt wrapper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_split_role_assignments(result.manifests, result.receipt)
    expected_hashes = dict(result.receipt.manifest_sha256s)
    manifest_files = {}
    for role in ("tune", "cal", "test"):
        manifest = result.manifests[role]
        if manifest.role != role:
            raise ValueError(f"role mismatch for manifest {role!r}")
        if manifest.source != result.receipt.source:
            raise ValueError(f"source mismatch for manifest {role!r}")
        if manifest.protocol_version != "icml2027-pr1a":
            raise ValueError(f"protocol_version mismatch for manifest {role!r}")
        actual_hash = manifest_sha256(manifest)
        if expected_hashes.get(role) != actual_hash:
            raise ValueError(f"manifest_sha256 mismatch for role {role!r}")
        filename = f"{role}_manifest.json"
        save_manifest(manifest, output_dir / filename)
        manifest_files[role] = filename

    cluster_manifest = _canonical_cluster_manifest(result.clusters)
    cluster_hash = hashlib.sha256(_canonical_json(cluster_manifest)).hexdigest()
    if cluster_hash != result.receipt.cluster_manifest_sha256:
        raise ValueError("cluster_manifest_sha256 mismatch")
    representatives = _validate_cluster_manifest_payload(
        cluster_manifest,
        result.receipt,
    )
    manifest_representatives = {
        (item.doc_id, item.content_sha256, item.cluster_id)
        for manifest in result.manifests.values()
        for item in manifest.documents
    }
    if representatives != manifest_representatives:
        raise ValueError("cluster representatives do not match role manifests")
    cluster_filename = "cluster_manifest.json"
    (output_dir / cluster_filename).write_text(
        json.dumps(
            {
                "cluster_manifest_sha256": cluster_hash,
                "cluster_manifest": cluster_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    wrapper = {
        "receipt_sha256": split_receipt_sha256(result.receipt),
        "receipt": asdict(result.receipt),
        "cluster_manifest_file": cluster_filename,
        "manifest_files": manifest_files,
    }
    receipt_path = output_dir / "split_receipt.json"
    receipt_path.write_text(json.dumps(wrapper, indent=2, sort_keys=True) + "\n")
    return receipt_path


def _receipt_from_dict(payload: object) -> SplitBuildReceipt:
    if not isinstance(payload, dict):
        raise ValueError("receipt must be a JSON object")
    expected = set(SplitBuildReceipt.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("receipt fields do not match SplitBuildReceipt schema")
    values = dict(payload)
    try:
        values["split_bands"] = tuple(tuple(item) for item in values["split_bands"])
        values["manifest_sha256s"] = tuple(
            tuple(item) for item in values["manifest_sha256s"]
        )
        receipt = SplitBuildReceipt(**values)
        _validate_receipt(receipt)
        return receipt
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed split receipt: {exc}") from exc


def load_split_receipt(
    receipt_path: Path,
) -> tuple[SplitBuildReceipt, dict[str, DocumentManifest]]:
    """Load a receipt and verify its cluster and role-manifest bindings."""
    receipt_path = Path(receipt_path)
    with receipt_path.open() as handle:
        wrapper = json.load(handle)
    if not isinstance(wrapper, dict) or set(wrapper) != {
        "receipt_sha256",
        "receipt",
        "cluster_manifest_file",
        "manifest_files",
    }:
        raise ValueError("split receipt wrapper has unexpected fields")
    receipt = _receipt_from_dict(wrapper["receipt"])
    actual_receipt_hash = split_receipt_sha256(receipt)
    if wrapper["receipt_sha256"] != actual_receipt_hash:
        raise ValueError(
            "receipt_sha256 mismatch: "
            f"recorded={wrapper['receipt_sha256']!r} actual={actual_receipt_hash!r}"
        )

    cluster_path = _resolved_artifact_path(
        receipt_path.parent,
        wrapper["cluster_manifest_file"],
        "cluster manifest path",
    )
    with cluster_path.open() as handle:
        cluster_wrapper = json.load(handle)
    if not isinstance(cluster_wrapper, dict) or set(cluster_wrapper) != {
        "cluster_manifest_sha256",
        "cluster_manifest",
    }:
        raise ValueError("cluster manifest wrapper has unexpected fields")
    cluster_hash = hashlib.sha256(
        _canonical_json(cluster_wrapper["cluster_manifest"])
    ).hexdigest()
    if cluster_wrapper["cluster_manifest_sha256"] != cluster_hash:
        raise ValueError("cluster_manifest_sha256 wrapper mismatch")
    if receipt.cluster_manifest_sha256 != cluster_hash:
        raise ValueError("cluster_manifest_sha256 receipt mismatch")
    representatives = _validate_cluster_manifest_payload(
        cluster_wrapper["cluster_manifest"],
        receipt,
    )

    raw_manifest_files = wrapper["manifest_files"]
    if not isinstance(raw_manifest_files, dict) or set(raw_manifest_files) != {
        "tune",
        "cal",
        "test",
    }:
        raise ValueError("manifest_files must contain tune, cal, and test")
    expected_hashes = dict(receipt.manifest_sha256s)
    manifests = {}
    for role in ("tune", "cal", "test"):
        manifest_path = _resolved_artifact_path(
            receipt_path.parent,
            raw_manifest_files[role],
            "manifest path",
        )
        manifest = load_manifest(manifest_path)
        if manifest.role != role:
            raise ValueError(f"role mismatch for {manifest_path.name}: {manifest.role!r}")
        if manifest.source != receipt.source:
            raise ValueError(
                f"source mismatch for {manifest_path.name}: {manifest.source!r}"
            )
        if manifest.protocol_version != "icml2027-pr1a":
            raise ValueError(
                f"protocol_version mismatch for {manifest_path.name}: "
                f"{manifest.protocol_version!r}"
            )
        actual_hash = manifest_sha256(manifest)
        if expected_hashes[role] != actual_hash:
            raise ValueError(f"manifest_sha256 receipt mismatch for role {role!r}")
        manifests[role] = manifest
    assert_pairwise_disjoint(manifests)
    _validate_split_role_assignments(manifests, receipt)
    manifest_representatives = {
        (item.doc_id, item.content_sha256, item.cluster_id)
        for manifest in manifests.values()
        for item in manifest.documents
    }
    if representatives != manifest_representatives:
        raise ValueError("cluster representatives do not match role manifests")
    return receipt, manifests


def _validated_exclusions(values: Collection[int]) -> frozenset[int]:
    exclusions = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("excluded_target_ids must contain integers")
        exclusions.add(value)
    return frozenset(exclusions)


def _validated_token_ids(values: Sequence[int]) -> tuple[int, ...]:
    token_ids = tuple(values)
    for value in token_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("token_ids must contain integers")
    return token_ids


def _unbiased_index(size: int, *, doc_id: str, salt: str) -> int:
    limit = (1 << 256) - ((1 << 256) % size)
    counter = 0
    while True:
        payload = (
            b"icml2027-guarantee-position-v1\x00"
            + salt.encode("utf-8")
            + b"\x00"
            + doc_id.encode("utf-8")
            + b"\x00"
            + str(counter).encode("ascii")
        )
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < limit:
            return value % size
        counter += 1


def select_guarantee_position(
    document: ManifestDocument,
    token_ids: Sequence[int],
    *,
    salt: str,
    excluded_target_ids: Collection[int] = (),
    min_context: int = 16,
) -> SelectedPosition:
    """Select exactly one deterministic uniform eligible target for a [G] row."""
    _require_nonempty(document.doc_id, "doc_id")
    _require_nonempty(document.cluster_id, "cluster_id")
    _require_nonempty(salt, "salt")
    if isinstance(min_context, bool) or not isinstance(min_context, int) or min_context < 0:
        raise ValueError("min_context must be a non-negative integer")
    exclusions = _validated_exclusions(excluded_target_ids)
    validated_token_ids = _validated_token_ids(token_ids)
    eligible = tuple(
        index
        for index in range(min_context, len(validated_token_ids))
        if validated_token_ids[index] not in exclusions
    )
    if not eligible:
        raise ValueError(
            f"no eligible target position for doc_id={document.doc_id!r}"
        )
    target_index = eligible[
        _unbiased_index(len(eligible), doc_id=document.doc_id, salt=salt)
    ]
    return SelectedPosition(
        doc_id=document.doc_id,
        cluster_id=document.cluster_id,
        target_index=target_index,
        evidence_grade="G",
    )


def select_guarantee_positions(
    documents: Sequence[ManifestDocument],
    token_ids_by_doc: Mapping[str, Sequence[int]],
    *,
    salt: str,
    excluded_target_ids: Collection[int] = (),
    min_context: int = 16,
) -> tuple[SelectedPosition, ...]:
    """Select one [G] target for every document with exact input binding."""
    if not isinstance(token_ids_by_doc, Mapping):
        raise ValueError("token_ids_by_doc must be a mapping")
    by_id = {}
    for document in documents:
        if not isinstance(document, ManifestDocument):
            raise ValueError("documents must contain ManifestDocument values")
        if document.doc_id in by_id:
            raise ValueError(f"duplicate doc_id: {document.doc_id!r}")
        by_id[document.doc_id] = document
    document_ids = set(by_id)
    token_ids = set(token_ids_by_doc)
    missing = sorted(document_ids - token_ids)
    extra = sorted(token_ids - document_ids)
    if missing:
        raise ValueError(f"token mapping has missing document IDs: {missing!r}")
    if extra:
        raise ValueError(f"token mapping has extra document IDs: {extra!r}")
    return tuple(
        select_guarantee_position(
            by_id[doc_id],
            token_ids_by_doc[doc_id],
            salt=salt,
            excluded_target_ids=excluded_target_ids,
            min_context=min_context,
        )
        for doc_id in sorted(by_id)
    )


def pooled_positions(
    document: ManifestDocument,
    token_ids: Sequence[int],
    *,
    excluded_target_ids: Collection[int] = (),
    min_context: int = 16,
    stride: int = 4,
) -> tuple[SelectedPosition, ...]:
    """Return document-identified stride positions for empirical [E] rows."""
    _require_nonempty(document.doc_id, "doc_id")
    _require_nonempty(document.cluster_id, "cluster_id")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if isinstance(min_context, bool) or not isinstance(min_context, int) or min_context < 0:
        raise ValueError("min_context must be a non-negative integer")
    exclusions = _validated_exclusions(excluded_target_ids)
    validated_token_ids = _validated_token_ids(token_ids)
    return tuple(
        SelectedPosition(
            doc_id=document.doc_id,
            cluster_id=document.cluster_id,
            target_index=index,
            evidence_grade="E",
        )
        for index in range(min_context, len(validated_token_ids), stride)
        if validated_token_ids[index] not in exclusions
    )
