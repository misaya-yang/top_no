# PR-2a Conformal Core Plan

- [x] Add regression tests for the impossible finite `n+1` quantile.
- [x] Return positive infinity instead of clamping a missing order statistic.
- [x] Add auditable Mondrian group thresholds, counts, absent/small-group
  reasons, and the `ceil(5 / delta)` default floor.
- [x] Add explicit float64 score dithering without hidden RNG state.
- [x] Add C-margin helpers and make C-nu share the score definition.
- [x] Add APS scores with explicit order and boundary uniforms.
- [x] Freeze stable top-p tie ordering and test crossing-token equivalence.
- [x] Test C-margin/min-p and C-nu(0)/C-margin equivalence.
- [x] Obtain independent review and run local/server verification.
- [ ] Merge and push the reviewed slice.
