# Related Work Positioning

Use decoding rules as support-construction baselines:

- `top-k`: cardinality constraint.
- `top-p`: cumulative probability-mass constraint.
- `typical`: local information-rate criterion.
- `Mirostat`: sequence-level perplexity control.
- `eta` / desmoothing: probability-space support recovery.
- `min-p`: fixed logit-margin special case.
- `top-nsigma`: context-global logit-margin special case.
- `nu`: token-wise frequency-calibrated logit-margin score.

## Positioning Sentence

```text
Rather than proposing another heuristic truncation threshold, we evaluate truncation as prediction-set construction: the central object is the retained token set, measured by true-token coverage and support efficiency before any downstream generation metric is claimed.
```

## Baseline Priority

The first table should compare calibrated methods against calibrated methods.
Uncalibrated truncation rules are useful diagnostics, but not enough for the
paper claim:

- `top_k_50`
- `top_p_0.95`
- `min_p_0.05`
- `fixed_margin_3`
- `top_nsigma_2`
- `nu_k10_m3`
- calibrated margin / `kappa=0`
- conformal top-p / APS-style cumulative-mass baseline
- frequency-Mondrian margin baseline
- `conformal_nu`

Do not lead with Distinct-n. It is a downstream surface-diversity metric, not the core claim. Do not present conformal coverage alone as evidence for frequency-aware decoding; coverage is generic once a valid split-conformal score is chosen.
