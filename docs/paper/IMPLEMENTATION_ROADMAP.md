# Implementation Roadmap

This roadmap reconciles the active paper plan with the pushed repository state.
The full external review is stored in
`docs/fable5/2026-07-09-repo-reconciliation-addendum-v1.1.md`.

## Current Stop Condition

No paper-grade GPU run should be launched until PR-2 and PR-3 are complete.
PR-1a now provides immutable external frequency artifacts, PR-1b provides
deterministic split construction receipts and position selectors, and PR-1c
binds those artifacts to exact text and a prefix-only `[G]` forward helper.
PR-1d adds a recomputed, threshold-complete cross-corpus disjointness proof.
The normal `experiments/eval_prediction_sets.py` path remains blocked because:

1. the calibrated methods registry and suffstats replay are not implemented;
2. the current gate compares one calibrated method against mostly uncalibrated
   baselines.

The runner refuses nonlegacy execution with
`blocked_pending_pr2_pr3` before model allocation. Explicit
`allow_legacy_protocol=true` retains the old sequential path only for
noncitable smoke tests.

## PR-0: Narrative And Legacy Guardrails

Status: mostly complete.

- README and `docs/paper/CLAIM_STACK.md` use the frequency-offset margin framing.
- Legacy reports are marked retired.
- `nu_topp_floor`, `nu_entropy`, and `nu_mathboost` require `legacy=True`.
- The Fable5 plan, stress spec, and repo reconciliation addendum live under
  `docs/fable5/`.

## PR-1a: Frequency Artifacts

Status: complete.

- `experiments/freq_table.py` loads token counts from `D_freq` only,
  with content-hashed metadata and manifest-disjointness checks.
- Downstream evaluators must reuse the exact referenced count artifact.
- Nonlegacy requests fail before model allocation.

## PR-1b: Deterministic Split Artifacts

Status: complete.

- Normalized 13-gram MinHash-LSH plus threshold-complete prefix candidates,
  followed by exact Jaccard `>= 0.8` confirmation and deterministic connected
  components.
- One canonical representative per cluster and salted 40/25/35 cluster-level
  tune/cal/test assignment.
- A cryptographic construction receipt binds source/input identity, cluster
  namespace, every algorithm parameter, membership, salt/bands, and manifests.
- `[G]` one-position-per-document and `[E]` stride-4 selectors preserve
  document/cluster identity and use explicit target exclusions.

## PR-1c: Manifest-To-Forward Binding

Status: exact text binding and the lower-level `[G]` forward are complete.

- Nonlegacy configs require the PR-1b receipt, frozen source JSONL, and distinct
  calibration/test position salts.
- Source documents are bound by semantic input hash, manifest `doc_id`, and raw
  UTF-8 content hash before any model allocation.
- The document-aware forward consumes prefix-only context windows, never feeds
  the target token, and carries document/cluster/position metadata.
- Calibration/test manifests and salts produce independent model calls; the
  legacy sequential skip/take path is not used by this lower-level API.

## PR-1d: Cross-Corpus Near-Duplicate Proof

Status: complete in code; paper execution remains fail-closed.

- The audit compares every frequency-manifest document against every document
  in the frozen evaluation input JSONL, including non-representative members of
  retained clusters.
- Candidate generation unions deterministic MinHash-LSH with the
  threshold-complete prefix index; exact integer `5I >= 4U` confirmation fixes
  the paper threshold at Jaccard `>= 0.8` over normalized 13-token shingles.
- The receipt binds both source JSONLs, the frequency manifest, the split
  receipt and role manifests, all algorithm parameters, candidate/exact counts,
  the comparison transcript, and the zero-match verdict.
- Protocol validation loads the saved receipt and independently recomputes the
  full audit before advancing to `blocked_pending_pr2_pr3`.
- Frequency-table `num_documents` must equal the bound frequency manifest.

Required tests: disjoint manifest tripwire, one-position-per-document
determinism, exact receipt/text binding, and refusal on count/eval near-duplicate
intersection. These are covered by the PR-1a through PR-1d tests.

## PR-2: Conformal Core And Methods Registry

- Add `mondrian_quantiles`, score dithering, and a tuning path that reads
  `D_tune` only.
- Add `experiments/methods.py` for calibrated baselines: C-margin, C-logprob,
  C-zmargin, APS, RAPS, TS+APS, CNS, entropy-Mondrian, frequency-Mondrian,
  learned-h/g, and C-nu.
- Add suffstats write/replay so Phase 0 and Phase 1 can share forward passes.

Required tests: synthetic exchangeable coverage, equivalence tests
(`C-nu(kappa=0) == C-margin`, APS as conformal top-p), and replay/direct
agreement.

## PR-3: Gate Rewrite

Replace substring-based gate logic with calibrated-vs-calibrated criteria:

- enforce minimum test size and document count;
- require green split/count tripwires and frozen config hashes;
- compute G1/G2 verdicts with document-clustered bootstrap CIs;
- emit `PASS`, `G2-only`, or `FAIL` with `[G]` and `[E]` evidence labels.

## PR-4: Phase 0 Diagnostic

Implement `experiments/phase0_reliability.py` for the decisive
margin-by-frequency diagnostic:

```text
h(m, n) = P(Y = i | margin m_i = m, frequency n_i = n)
```

The decision memo should determine whether frequency adds information at fixed
margin and what sign/shape any offset should take.
