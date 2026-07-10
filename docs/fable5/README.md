# Fable5 Research Notes

This directory collects Fable5-facing prompts, reviews, and research plans for
the ICML 2027 decoding/calibration project.

## Reading Order

1. `2026-07-09-fable5-theory-results-brief.md`
   - Context package prepared for Fable5.
   - Summarizes the repo state, current claim stack, code entrypoints, smoke
     result, and known weaknesses.

2. `2026-07-09-fable5-analysis.md`
   - Fable5's critical review of the original framing.
   - Key conclusion: conformal coverage is not novelty; the frequency signal and
     calibrated-vs-calibrated efficiency comparison must carry the paper.

3. `icml2027_theory_and_experiment_plan.md`
   - Fable5's stronger global plan.
   - Reframes the work as frequency-offset margin rules, Phase 0
     margin-by-frequency reliability diagnostics, Mondrian baselines, and a
     calibrated gate before downstream generation.

4. `icml2027_stress_test_spec.md`
   - Fable5's stress-test specification for hardening the theory, gate,
     baselines, and implementation plan before spending larger GPU budget.

5. `2026-07-09-repo-reconciliation-addendum-v1.1.md`
   - Fable5's reconciliation after inspecting the pushed GitHub `main`.
   - Upgrades two risks to confirmed defects: eval-pool frequency-count leakage
     and sequential calibration/eval splitting.

6. `2026-07-10-topno-deep-review.md`
   - Fable5's 2026 decoding-landscape review after the Phase-0 pilot.
   - Recommends a conformal scheduler for risk-controlled test-time compute as
     the flagship direction, with the repaired margin/frequency audit retained
     as the lower-risk fallback.

7. `2026-07-10-final-method-paper-draft-prompt.md`
   - Final adjudication prompt to use with the Fable5 and GPT-5.6 Pro reviews.
   - Requires one selected method and a detailed paper-grade English draft,
     rather than another menu of possible directions.

## Current Takeaway

The frequency-offset sampler is paused after the Phase-0 result. Both current
deep reviews recommend moving any flagship claim to a decision-relevant
test-time-compute consumer, while retaining the confound-complete frequency
audit only as a lower-risk empirical path. The final method has not yet been
selected; the adjudication prompt above is intended to make that decision once.

## Immediate Implementation Focus

- Do not restart GPU experiments or extend the old sampler pipeline before the
  final paper method and risk contract are selected.
- Preserve the Phase-0 evidence, protocol repairs, and audit infrastructure as
  reusable research assets.
- After adjudication, replace this section with one implementation and
  experiment plan for the selected paper only.
