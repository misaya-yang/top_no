# Implementation Roadmap

This roadmap reconciles the active paper plan with the pushed repository state.
The full external review is stored in
`docs/fable5/2026-07-09-repo-reconciliation-addendum-v1.1.md`.

## Current Stop Condition

No paper-grade GPU run should be launched until PR-1 through PR-3 are complete.
The current `experiments/eval_prediction_sets.py` path is a legacy link-test
runner only because:

1. token counts are built from the same loaded text pool used for calibration
   and evaluation;
2. calibration/evaluation positions are consumed sequentially rather than drawn
   from disjoint document-level split manifests;
3. the current gate compares a calibrated method against mostly uncalibrated
   baselines.

The runner now refuses paper-grade execution unless `allow_legacy_protocol=true`
is explicitly set for smoke tests.

## PR-0: Narrative And Legacy Guardrails

Status: mostly complete.

- README and `docs/paper/CLAIM_STACK.md` use the frequency-offset margin framing.
- Legacy reports are marked retired.
- `nu_topp_floor`, `nu_entropy`, and `nu_mathboost` require `legacy=True`.
- The Fable5 plan, stress spec, and repo reconciliation addendum live under
  `docs/fable5/`.

## PR-1: Frequency Tables And Splits

Create the protocol layer that makes conformal claims defensible:

- `experiments/freq_table.py`: load or build token counts from `D_freq` only,
  with content-hashed metadata and manifest-disjointness checks.
- `experiments/splits.py`: document/cluster-level `D_tune`, `D_cal`, `D_test`
  manifests with deterministic seeded sampling.
- Replace the tiny eval-pool frequency buckets with corpus-scale log buckets and
  method-side mass-quantile buckets.
- Delete the inline `build_token_counts()` and sequential skip/take protocol
  from `eval_prediction_sets.py`.

Required tests: disjoint manifest tripwire, one-position-per-document
determinism, and refusal on count/eval manifest intersection.

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
