# PR-2f C-zmargin Validation

Validation target: `codex/pr2f-c-zmargin`.

C-zmargin uses centered sample standard deviation and the canonical registry.
Target calibration computes mean/variance in two 4096-token passes without a
full fp32 `(N,V)` working tensor. Zero-variance and singleton rows receive zero
scores. At zero dither, the set is exactly top-nsigma with
`n_sigma=q_hat`.

The paper-grade blocker now lists four missing mandatory methods: `cns`,
`learned_h`, `raps`, and `ts_aps`. Runner and gate behavior are unchanged.

## Automated checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
git diff --check                                          PASS
python3 -m unittest discover tests                        183/183 PASS
```

RTX 5090 server:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         183 run, 182 PASS,
                                                         1 MPS-only skip
```

At Qwen2.5-7B vocabulary width (`V=152064`), a synthetic fp16 CUDA smoke gave
`q_hat=6.270020008087158` and set sizes 148,342/147,423. A constant 60,000-logit
row produced zero target score and only zero full scores. These tiny synthetic
runs verify device/shape/numerics only and are not paper evidence.

Tests cover top-nsigma equivalence, common-offset stability, zero variance,
singleton vocabulary, target/full agreement across the 4096-token chunk
boundary, registry execution, and continued protocol blocking.

Independent review found one Important score-symmetry issue: separate full-row
and chunked reductions differed by up to `9.54e-7`, enough to interact with the
`1e-6` dither at a threshold. Full and target paths now share the identical
chunked row-statistics helper. The reviewer reran fp16/fp32/fp64 at `V=152064`
over multiple seeds and observed bitwise equality with worst difference zero;
no Critical/Important issues remain.
