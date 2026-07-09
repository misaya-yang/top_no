# Manifest-To-Text Binding Implementation Plan

### Task 1: Strict frozen source store

- [x] Add strict JSONL loading and public semantic document hashing.
- [x] Reject blank/malformed/extra-field rows, duplicate IDs, and empty input.
- [x] Reuse the loader in the split-preparation CLI.

### Task 2: Receipt/manifests/text binder

- [x] Add immutable bound-document and bound-split types.
- [x] Verify input hash, source, configured manifest hashes, representative IDs,
      and raw content hashes.
- [x] Test row-order invariance and all missing/extra/tampered failure modes.

### Task 3: Fail-closed protocol integration

- [x] Add receipt, document JSONL, and position-salt config/CLI inputs.
- [x] Bind text before dataset/model allocation for nonlegacy requests.
- [x] Add prefix-only independent cal/test `[G]` forward helpers.
- [x] Advance only to `blocked_pending_cross_corpus_cluster`; keep legacy smoke.

### Task 4: Review and integration

- [ ] Update roadmap with the achieved boundary and remaining PR-1d work.
- [ ] Run all tests/compile/shell checks locally and on the server.
- [ ] Fix independent Critical/Important review findings.
- [ ] Commit, push, merge to `main`, verify, and push.
