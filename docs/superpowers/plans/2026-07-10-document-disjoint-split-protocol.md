# Document-Disjoint Split Protocol Implementation Plan

**Goal:** Build deterministic near-duplicate-aware tune/cal/test artifacts,
construction receipts, and document-position selectors while keeping the
evaluator blocked until PR-1c binds manifests to actual forward-pass text.

**Constraints:** standard library only; preserve legacy CLI behavior; use TDD;
do not edit generated `results/`; never label development output paper-grade.

### Task 1: Construction metadata and deterministic clustering

**Files:** `experiments/splits.py`, `tests/test_split_construction.py`

- [x] Add canonical construction receipt metadata without changing the PR-1a
      manifest wire schema.
- [x] Add normalized 13-gram shingles, MinHash-LSH candidate generation,
      exact Jaccard confirmation, union-find clusters, and canonical reps.
- [x] Add salted fixed-band role assignment and order-invariance tests.
- [x] Test exact duplicates, near duplicates, threshold separation, and
      construction-parameter hash drift.

### Task 2: Cryptographic build receipt and CLI

**Files:** `experiments/splits.py`, `experiments/prepare_document_splits.py`,
`tests/test_split_artifacts.py`

- [x] Bind input/source hashes, every algorithm parameter, membership digest,
      and role-manifest hash into a canonical receipt.
- [x] Reject receipt/manifest drift and unsafe manifest paths.
- [x] Add a JSONL-to-manifests-and-receipt CLI and end-to-end CLI test.

### Task 3: Document-aware position selection

**Files:** `experiments/splits.py`, `tests/test_document_sampling.py`

- [x] Add deterministic uniform one-position selection with 16-token context.
- [x] Add stride-4 pooled indices with explicit target-exclusion policy.
- [x] Test bounds, order independence, approximate uniformity, and fatal
      handling of empty eligible position sets.

### Task 4: Fail-closed boundary update

**Files:** `experiments/protocol.py`, `tests/test_prediction_protocol.py`

- [x] Advance the terminal reason to `blocked_pending_pr1c` after PR-1a checks.
- [x] Document the exact remaining manifest-to-text/evaluator invariants.
- [x] Keep the block before dataset/model allocation and preserve legacy smoke.

### Task 5: Documentation, review, and integration

- [ ] Update roadmap/mainline docs with the exact achieved boundary.
- [ ] Run `compileall`, shell syntax checks, full unit suite, and a reduced
      document-store smoke.
- [ ] Request independent review; fix all Critical/Important findings.
- [ ] Commit, push feature branch, merge to `main`, rerun checks, and push.
