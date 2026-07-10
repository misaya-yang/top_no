# PR-2g RAPS Baseline Design

RAPS uses the same randomized APS boundary score and explicit total order as
APS, then adds a rank penalty:

```text
A_RAPS(x, i) = A_APS(x, i; U_i)
               + lambda * max(rank_1based(x, i) - k_reg, 0).
```

`rank_1based=1` is the highest-logit token. Equal logits are ordered by token
ID through the existing stable descending sort. `lambda` must be finite and
non-negative; `k_reg` must be a positive integer. At `lambda=0`, RAPS is APS
after dtype alignment.

RAPS uses only the APS boundary uniform. It must not receive the independent
score dither used by margin/log-probability methods. Candidate and calibration
target scores share the same full-score implementation, so the target path is
exactly a gather from the candidate tensor.

For fp16/bfloat16/fp32 logits, APS promotes logits and uniforms before softmax,
cumulative mass, and boundary scoring, then adds the rank penalty in fp32;
fp64 is preserved. This is required because Qwen2.5 has a 152,064-token
vocabulary, far beyond the range where fp16 can preserve cumulative mass or
distinguish adjacent integer ranks.

CUDA floating-point scans may differ by one ULP across calls unless PyTorch
deterministic algorithms are enabled. The tensor helper does not silently
change process-global runtime policy. A future paper runner must enable and
verify deterministic CUDA execution before model initialization and bind the
runtime policy/version/device receipt into gate provenance.

The canonical method parameters are `lambda` and `k_reg`. They are selected on
`D_tune`, stored in `MethodCalibration.params`, and gate evidence requires a
non-null tuning artifact. This slice does not select a parameter grid, emit
evaluator rows, or unblock paper-grade execution.
