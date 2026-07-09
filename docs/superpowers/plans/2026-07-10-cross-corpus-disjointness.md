# PR-1d Cross-Corpus Disjointness Plan

- [x] Add threshold-complete cross-corpus candidate generation and exact
  Jaccard confirmation at the frozen 13-shingle, 0.8 protocol.
- [x] Bind the frequency manifest to exact source JSONL and reuse PR-1c binding
  for every evaluation input document, including discarded cluster members.
- [x] Serialize a content-addressed audit with transcript and match hashes.
- [x] Recompute saved audits and reject forged zero-match receipts.
- [x] Add a CLI that refuses to write a passing artifact on overlap.
- [x] Require the audit, both source JSONLs, and metadata document-count
  agreement in normal protocol validation.
- [x] Advance the fail-closed terminal reason to `blocked_pending_pr2_pr3`.
- [x] Cover exact duplicates, the Jaccard-0.8 LSH miss counterexample,
  tampering, weak thresholds, CLI behavior, and protocol integration.
- [x] Obtain independent code review and run local/server verification before
  merge and push.
