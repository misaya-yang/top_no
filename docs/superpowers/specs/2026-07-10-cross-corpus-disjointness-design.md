# PR-1d Cross-Corpus Disjointness Design

## Goal

Close the last PR-1 provenance gap: no document used to build the immutable
token-frequency table may be an exact or near duplicate of a retained
`D_tune`, `D_cal`, or `D_test` representative. A saved hash is not trusted as
proof; normal protocol validation must reproduce the comparison from frozen
source text.

## Frozen comparison contract

- Left side: every document in the `freq` manifest, with exact raw text bound
  from its source JSONL by `doc_id` and SHA-256.
- Right side: every document in the exact evaluation input JSONL bound by the
  PR-1b split receipt and PR-1c document store, including discarded members of
  retained clusters.
- Normalization: NFKC, case-fold, whitespace tokenization.
- Similarity: set Jaccard over 13-token shingles, with the paper threshold
  represented exactly as `4/5`.
- Candidates: union of fixed-seed MinHash-LSH and the deterministic
  threshold-complete prefix index, followed by exact integer comparison.
- Passing condition: zero pairs satisfy `5 * intersection >= 4 * union`.

Scanning all input members is necessary because PR-1b clusters are transitive
connected components: a discarded endpoint can be below the threshold against
the chosen representative while still being linked through an intermediate
member. Restricting the audit to representatives would therefore permit a
frequency document to duplicate an evaluation-cluster member. The cross audit
does not rewrite either corpus or silently drop collisions; any collision is a
hard failure that requires rebuilding the upstream artifacts.

## Receipt and validation

The JSON receipt binds both corpus hashes, the frequency and role-manifest
hashes, the split receipt, fixed algorithm parameters, candidate-pair and exact
comparison counts, a canonical exact-comparison transcript hash, and the full
match list. The CLI writes an artifact only when the match list is empty.

`validate_cross_corpus_audit` first validates the wrapper and serialized
matches, then recomputes the entire audit from the configured inputs and
requires structural equality. This rejects a forged zero-match receipt even if
its internal hashes are self-consistent. PR-1c's bound document snapshot carries
the exact split receipt and full source-document tuple from one load, so scan
parameters, manifest bindings, and compared text cannot come from separate
receipt reads. The prediction-set protocol performs this recomputation before
model allocation and then stops at `blocked_pending_pr2_pr3`.

## Non-goals

- PR-2 score calibration, methods registry, and suffstats replay.
- PR-3 calibrated-vs-calibrated gate logic.
- Treating legacy smoke results as paper evidence.
