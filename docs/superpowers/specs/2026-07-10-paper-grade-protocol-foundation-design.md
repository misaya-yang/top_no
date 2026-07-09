# Paper-Grade Protocol Foundation Design

**Date:** 2026-07-10  
**Status:** Approved for autonomous execution under the user's 10-hour goal  
**Scope:** PR-1a only — frequency-table artifacts, provenance, and fail-closed
tripwires

## 1. Goal

Eliminate the confirmed A1 frequency-leakage path and establish a verifiable
artifact contract for every future paper-grade run. The change must make it
impossible to silently rebuild token counts from calibration, test, or
downstream text while reusing a conformal threshold calibrated with a different
score.

This slice deliberately does **not** unlock paper-grade prediction-set runs.
Document clustering, deterministic `D_tune/D_cal/D_test` assignment, and
one-position-per-document evaluation remain PR-1b and keep the A2 blocker
closed.

## 2. Why This Slice Comes First

Three implementation routes were considered:

1. **Full PR-1 in one change.** This removes A1 and A2 together, but combines
   artifact provenance, MinHash clustering, split assignment, document-aware
   forward passes, and sampling changes in one hard-to-review diff.
2. **PR-1a vertical slice (selected).** Build and validate immutable external
   frequency tables, wire their identity into the evaluator and downstream
   consumers, and retain the paper-grade block until PR-1b lands.
3. **Phase-0-first experiment.** This would produce a quick signal, but the
   current frequency counts and sequential split make that signal unsuitable
   for paper claims and potentially misleading for method selection.

The selected route is the smallest independently testable change that removes
one confirmed validity defect without weakening the existing safety gate.

## 3. Scientific Contract

For any calibrated score using token frequency, the frequency table is part of
the score definition. Therefore the following tuple is immutable across tuning,
calibration, evaluation, and downstream corroboration:

```text
(model_id, tokenizer_id, tokenizer_revision, vocab_size,
 exclusion_token_ids, counts_sha256, source_manifest_sha256)
```

Changing any field creates a different score and invalidates the old
`q_hat`. A downstream run must refuse to proceed rather than rebuild counts or
silently accept a mismatch.

The artifact proves provenance and integrity, not scientific adequacy. A valid
artifact can still be scientifically unsuitable if its source corpus is poorly
chosen. Paper-grade configs must eventually reference a pre-registered,
disjoint `D_freq` artifact built from a documented corpus shard.

## 4. Artifact Format

Add `experiments/freq_table.py` with one public data object and explicit I/O:

```python
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


def save_frequency_table(
    counts: torch.Tensor,
    metadata: FrequencyTableMetadata,
    output_dir: Path,
) -> Path:
    """Write `<artifact_id>.counts.pt` and `<artifact_id>.json`."""


def load_frequency_table(
    metadata_path: Path,
    *,
    expected_model_id: str,
    expected_tokenizer_id: str,
    expected_vocab_size: int,
    expected_exclusion_token_ids: Collection[int],
) -> tuple[torch.Tensor, FrequencyTableMetadata]:
    """Load only after all hashes and compatibility checks pass."""
```

Rules:

- Counts are canonicalized to contiguous CPU `torch.int64` before hashing and
  saving.
- `counts_sha256` hashes canonical tensor bytes plus shape and dtype.
- The metadata filename stem is a deterministic artifact ID derived from the
  canonical metadata payload; timestamps are excluded from identity.
- The sidecar stores the counts filename relative to itself. Absolute local
  paths are never part of artifact identity.
- `torch.load(..., weights_only=True, map_location="cpu")` is used when the
  installed Torch supports it. Loaded content must be a one-dimensional tensor;
  arbitrary pickled objects are rejected.
- Negative or fractional counts, wrong vocabulary length, hash mismatch,
  tokenizer/model mismatch, and exclusion mismatch are fatal.
- Excluded PAD/BOS/control IDs are frozen in metadata and their counts are
  forced to zero by the builder.

## 5. Manifest Contract and Tripwire

Add the minimal manifest contract to `experiments/splits.py`; PR-1b will extend
the same module with deduplication and split construction.

```python
@dataclass(frozen=True)
class DocumentManifest:
    protocol_version: str
    role: str
    source: str
    documents: tuple[ManifestDocument, ...]


load_manifest(path: Path) -> DocumentManifest
manifest_sha256(manifest: DocumentManifest) -> str
assert_pairwise_disjoint(manifests: Mapping[str, DocumentManifest]) -> None
```

Each document entry carries `doc_id`, `content_sha256`, and `cluster_id`.
Disjointness is checked on all three identifiers. Repeated identifiers inside a
single manifest are also fatal. This is stricter than checking only document
IDs and catches copied content under different source IDs when a prior
clustering step has supplied a shared cluster ID.

PR-1a validates manifests but does not claim to generate trustworthy clusters.
Only PR-1b may create paper-grade manifests via MinHash-LSH and salted
cluster-level assignment.

## 6. Evaluator Data Flow

`experiments/eval_prediction_sets.py` keeps two visibly separate paths:

```text
legacy smoke:
  loaded text pool -> inline counts -> sequential calibration/eval
  requires allow_legacy_protocol=true
  output evidence_grade = "legacy-smoke"

future paper path:
  D_freq artifact + tune/cal/test manifests -> tripwire -> model load
  remains blocked_pending_pr1b in this slice
```

The evaluator gains config fields for `frequency_table` and the four manifest
paths. Protocol validation runs before loading a model or dataset. A supplied
frequency table is loaded through `freq_table.py`; inline count construction is
never selected implicitly for a non-legacy run.

The existing `build_token_counts()` symbol remains temporarily available for
legacy callers because three downstream scripts import it. It is renamed
internally/documented as legacy and cannot be used by a paper-grade config.
Deleting that compatibility surface is deferred until all consumers use the
artifact loader.

Every result JSON records:

```text
protocol_version
evidence_grade
frequency_table metadata path and artifact ID
counts_sha256
source_manifest_sha256
tune/cal/test manifest hashes when present
config_sha256
```

No output from PR-1a is labeled `[G]` or citable; A2 is still unresolved.

## 7. Downstream Consistency

`eval_reasoning_self_consistency.py` and `eval_openended_quality.py` currently
rebuild counts from WikiText while reusing `kappa`, `alpha`, and `q_hat`. That
changes the nonconformity score and invalidates the calibration.

PR-1a changes these consumers to one of two explicit behaviors:

- load the exact artifact referenced by the upstream metrics file and validate
  its hash; or
- refuse to run if the metrics file lacks a valid artifact reference.

Legacy smoke behavior may remain only behind an explicit legacy flag and must
write `evidence_grade = "legacy-smoke"`. No downstream command may silently
fall back to rebuilding counts.

Controlled-channel experiments remain archived and are removed from the
one-shot paper queue; they do not consume this contract in the current claim
stack.

## 8. Error Handling

All protocol errors fail before model allocation and use messages that identify
the violated invariant without dumping source documents or credentials.

Fatal cases include:

- missing counts or sidecar file;
- mismatched or tampered tensor hash;
- model/tokenizer/vocabulary/exclusion mismatch;
- source manifest hash mismatch;
- any `D_freq/D_tune/D_cal/D_test` document, content, or cluster intersection;
- missing artifact reference in a non-legacy downstream run;
- artifact hash different from the one used to produce `q_hat`;
- paper-mode request while PR-1b is still incomplete.

## 9. Tests and Acceptance Criteria

New CPU-only tests must cover:

1. deterministic artifact identity and round-trip;
2. rejection of modified counts and modified metadata;
3. tokenizer, vocabulary, and exclusion-list mismatches;
4. manifest canonical hashing and pairwise disjoint success;
5. intersections by `doc_id`, `content_sha256`, and `cluster_id`;
6. evaluator protocol validation before model loading;
7. legacy smoke remains explicit and is labeled non-citable;
8. downstream artifact-hash mismatch fails instead of rebuilding counts;
9. summary provenance contains all available hashes.

Repository checks:

```bash
python3 -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover tests
```

The server check additionally loads the downloaded Qwen tokenizer/config with
`local_files_only=True` and verifies that an artifact built for it passes the
model/tokenizer/vocabulary contract.

## 10. Explicit Non-Goals

- No paper-grade GPU run or claim unlock.
- No MinHash-LSH implementation, cluster assignment, or one-position-per-doc
  sampler; those are PR-1b.
- No Mondrian quantiles, methods registry, suffstats, Phase-0 estimator, or gate
  rewrite; those remain PR-2 through PR-4.
- No new dependency.
- No reinterpretation of legacy result artifacts.

## 11. Follow-On Gate

After PR-1a is merged, PR-1b must make evaluator inputs manifest-driven rather
than merely validating manifest files. Paper-grade execution stays blocked
until:

1. MinHash-LSH clustering and salted 40/25/35 assignment are implemented;
2. `[G]` rows sample exactly one eligible position per document;
3. pooled `[E]` rows preserve document/cluster identifiers;
4. PR-2 conformal/equivalence tests and PR-3 calibrated gate are green.
