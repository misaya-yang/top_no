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

## Current Takeaway

The project should not currently claim a universal new sampler. The stronger
path is to test whether token frequency adds information beyond logit margin for
calibrated support construction. If it does, the method paper is viable. If it
does not, the fallback is a conformal audit / measurement paper.

## Immediate Implementation Focus

- Implement the Phase 0 margin x frequency diagnostic.
- Add calibrated baselines: C-margin, conformal top-p / APS, RAPS, CNS, signed
  C-nu, frequency-Mondrian margin, and learned-g.
- Replace the current gate with calibrated-vs-calibrated Pareto and CovGap
  criteria.
- Keep downstream generation gated and secondary.
