# PR-2b Calibrated Method Registry: Margin-Family Slice

## Scope

Introduce stable method keys and one calibration/prediction interface for the
already implemented conformal primitives. This slice implements `c_margin`,
`aps`, `c_nu`, `frequency_mondrian_margin`, and
`entropy_mondrian_margin`. The remaining mandatory baselines stay registered
but explicitly unavailable, so paper-grade execution remains blocked.

## Contract

- Registry keys, rather than display-name substrings, identify methods.
- Calibration returns an immutable `MethodCalibration` containing the method
  key, finite-sample threshold(s), group semantics, score parameters, and
  explicit tie-randomization policy.
- Non-APS methods require caller-supplied score uniforms and use float64
  dithering. APS requires its own caller-supplied boundary uniforms and never
  receives a second dither.
- Frequency Mondrian groups candidates by token identity (`V` group IDs).
  Entropy Mondrian groups complete contexts (`N` group IDs). Calibration uses
  the corresponding true-token or context group IDs.
- Absent or under-sized Mondrian groups retain the conformal `+inf` threshold.
  Prediction rejects unknown group IDs instead of inventing a fallback.
- Unknown keys, registered-but-unimplemented methods, malformed group shapes,
  missing frequency tables, and missing uniforms fail closed.

## Non-goals

This slice does not implement RAPS, TS+APS, CNS, learned-h/g, suffstats replay,
tuning artifacts, the calibrated gate, or runner integration. It therefore
does not remove `blocked_pending_pr2_pr3` and cannot produce paper evidence.

