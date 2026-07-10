# PR-2g RAPS Validation

RAPS is now executable through the canonical method registry using

```text
A_RAPS(x, i) = A_APS(x, i; U_i)
               + lambda * max(rank_1based(x, i) - k_reg, 0).
```

The order is logit-descending with token-ID-stable ties. `lambda` is finite and
non-negative; `k_reg` is a positive integer. Both are calibration identity,
require a tune artifact in gate evidence, and never receive the independent
score dither used by margin-family methods.

## Numerical Correction Found During Review

The first implementation promoted only after APS scoring. At Qwen2.5 width
(`V=152064`), constant fp16 logits then produced only 8,355 unique APS scores,
total probability 0.9970, and maximum cumulative-mass error about 0.00323.
Independent review classified this as Important: lost softmax/cumulative-mass
precision cannot be restored after the fact.

APS now promotes every non-fp64 input before softmax, cumulative mass, and
boundary scoring. On the RTX 5090, fp16/bfloat16/fp32 each produced 152,064
unique APS values, all 152,064 half-boundary uniforms changed their score, and
the last-prefix error was `1.60e-7`; fp64 error was below `5e-16`.

Default CUDA floating-point scans varied by at most one ULP across independent
calls. With `torch.use_deterministic_algorithms(True, warn_only=False)`, repeated
APS and `RAPS(lambda=0)` were bitwise equal for fp16, bfloat16, fp32, and fp64.
The reviewer recommends configuring and binding determinism at the future
paper-runner boundary, not mutating process-global policy inside tensor helpers.
This is recorded as a paper-grade runtime blocker.

## Automated Checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python3 -m unittest discover tests                        192/192 PASS
git diff --check                                          PASS
```

RTX 5090 server:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         192 run, 191 PASS,
                                                         1 MPS-only skip
```

Tests cover the one-based penalty formula, `lambda=0` APS limit, target/full
agreement, malformed parameters, low-precision 70k-vocabulary resolution,
canonical calibration/prediction, tuning evidence, boundary-only
randomization, and exact remaining paper-method blockers. Independent review
reports no remaining Critical or Important issue in the PR-2g slice.
