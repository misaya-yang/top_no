# PR-2e C-logprob Validation

Validation target: `codex/pr2e-c-logprob`.

## Implementation

C-logprob now uses the canonical registry key `c_logprob` and the stable score
`logsumexp(logits) - logit_i`. Target calibration gathers only the observed
target; prediction constructs full candidate scores. Low-precision inputs
accumulate in float32 and float64 inputs remain float64. The same explicit
dither and finite-sample quantile path as the other global methods applies.

The paper-grade blocker now lists five missing mandatory methods:
`c_zmargin`, `cns`, `learned_h`, `raps`, and `ts_aps`. This slice does not
change the runner or gate.

## Automated checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
git diff --check                                          PASS
python3 -m unittest discover tests                        179/179 PASS
```

RTX 5090 server:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         179 run, 178 PASS,
                                                         1 MPS-only skip
```

## Real-vocabulary CUDA smoke

With synthetic fp16 logits at Qwen2.5-7B vocabulary width (`V=152064`), 19
calibration rows produced `q_hat=14.382911682128906`; two prediction rows had
set sizes 148,127 and 148,160. This deliberately tiny finite-sample smoke only
checks CUDA shape/numerics and is not empirical or paper evidence.

An equal-logit fp16 CUDA row at the same vocabulary width and common offset
60,000 produced `11.932056427001953`, agreeing with
`log(152064)=11.932056763842207`.

Tests also verify exact epsilon-set equivalence at zero dither, additive-logit
shift invariance, finite fp16 scores at a 100-nat gap, target-only scoring, and
the registry/protocol blocker transition.

Independent review found and drove two Important numerical/memory repairs. The
full path now centers logits before `logsumexp`, preserving `log(K)` at huge
common offsets. The target-only path performs fp32/fp64 accumulation in
4096-token chunks instead of materializing a full fp32 `(N,V)` working tensor.
The reviewer additionally checked chunk boundaries, fp16/bfloat16/fp32/fp64
reference agreement, finite autograd, and 64 targeted tests, then reported no
remaining Critical/Important issues.
