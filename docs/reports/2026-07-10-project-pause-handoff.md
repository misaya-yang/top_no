# Project Pause Handoff

**Date:** 2026-07-10

**Repository state:** `main` at `7e88b68` before this handoff

**Decision:** Pause the ICML 2027 project and do not spend more rented-GPU time.

## Why the project is paused

The active method-paper claim requires external-corpus token frequency to add
useful predictive information beyond logit margin. The completed Qwen2.5 3B/7B
Phase-0 pilot found suggestive conditional structure on the frozen web and math
domains, but it did not clear its own pilot decision rule:

- all four cells completed and were individually informative;
- cross-domain and cross-scale signs agreed;
- only three of four cells separated from the single permutation control;
- both web cells failed the half-stability check because the selected reference
  group changed between halves;
- the 7B-math permutation maximum was driven by a different bucket rather than
  the rare-token contrast;
- the aggregate verdict was `INSUFFICIENT`.

These outputs are `E-pilot`, `paper_citable=false`. They establish neither a
coverage/size improvement nor a valid additive frequency-offset rule. Continuing
directly to a broad GPU benchmark would therefore spend compute before the
central empirical premise is strong enough.

## Preserved evidence

The compact server summaries are committed under:

```text
results/server_runs/topno_phase0_20260710_7e88b68_summary/
```

They include the four cell summaries and `full_tune_decision_memo.json`. The
large raw archive and high-frequency GPU telemetry remain local-only and are
ignored by Git:

```text
results/server_runs/topno_phase0_20260710_7e88b68.tar.gz
results/server_runs/topno_phase0_20260710_7e88b68_summary/phase0_gpu.csv
```

The summaries bind the experiment code commit, exact model revisions, frequency
artifact identity, runtime receipt, document/position counts, and the pilot
evidence grade. No model weights, frozen corpora, credentials, or server
connection details are committed.

## Code state at pause

The repository remains deliberately fail-closed for paper-grade prediction-set
runs:

- PR-1 provenance, deterministic document splits, manifest-to-forward binding,
  and the cross-corpus audit are implemented;
- the conformal core and several calibrated registry methods are implemented;
- mandatory `cns`, `learned_h`, and `ts_aps` methods remain unavailable;
- evaluator suffstats/replay, frozen CUDA evidence, and the PR-3
  calibrated-vs-calibrated gate remain incomplete;
- `experiments/protocol.py` must continue to raise
  `blocked_pending_pr2_pr3` for nonlegacy paper execution.

Legacy-protocol smoke outputs and the preserved Phase-0 pilot must not be used
as paper evidence.

## Restart criteria

Do not resume implementation or rent a GPU merely to finish the existing
roadmap. Resume only if all of the following are accepted in advance:

1. The next run is a falsification-first experiment, not a broad benchmark.
2. Its primary comparison tests whether frequency adds out-of-sample value
   after controlling for margin, context entropy/logit scale, and tokenizer
   morphology.
3. `D_tune`, `D_cal`, and `D_test` roles remain disjoint, with the test split
   opened only after the score and thresholds are frozen.
4. The real-frequency gain must beat multiple pre-registered token-frequency
   permutations and document-clustered bootstrap uncertainty.
5. A method-paper continuation requires a meaningful calibrated coverage/size
   improvement over the strongest non-frequency baseline in both domains. A
   correlation plot alone is not sufficient.
6. Model caches, tokenized inputs, artifact hashes, disk capacity, deterministic
   runtime settings, checkpointing, and the complete launch command are verified
   before a paid GPU is started.
7. Failure of the frozen gate ends the current method-paper direction.

If those conditions are not worth the remaining engineering and compute cost,
the correct action is to leave the repository paused.

## Lightweight verification

The paused repository should continue to pass:

```bash
python3 -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover tests
```

No GPU experiment is required to verify this handoff.
