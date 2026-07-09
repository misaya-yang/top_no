# PR-2a Conformal Core Validation

Validation target: `codex/pr2-conformal-core`, including independent-review
repairs after the initial `027d28d` implementation commit.

## Mathematical regression coverage

The new core tests cover:

- the invalid finite-threshold case (`n=18, delta=0.05 -> +inf`) and the exact
  finite boundary (`n=19 -> max`);
- exhaustive leave-one-out ranks for five distinct exchangeable scores;
- NaN rejection and explicit infinite thresholds;
- Mondrian finite, below-floor, rank-exceeding, and absent groups with counts;
- explicit float64 dithering and invalid uniform/epsilon inputs;
- C-nu at zero equals C-margin;
- target-only C-margin/C-nu scoring cannot call the full `(N,V)` score helper;
- C-margin equals min-p under `p_min=exp(-q_hat)`;
- deterministic APS equals crossing-token top-p under one explicit order;
- APS boundary uniforms and explicit tie-order behavior.

## Local and server checks

macOS, Python 3.9.6, Torch 2.8.0:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python3 -m unittest discover tests                        144/144 PASS
```

RTX 5090 server, project virtualenv:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         144 run, 143 PASS,
                                                         1 MPS-only skip
```

## RTX 5090 vocabulary-scale smoke

A direct CUDA smoke used four random rows over the Qwen-size 151,643-token
vocabulary. It computed stable descending orders, full APS scores, C-margin
scores, explicit float64 dither, and Mondrian thresholds with group IDs supplied
from CPU. APS produced a finite `(4, 151643)` CUDA tensor; dither stayed on CUDA
in float64; both 10-sample Mondrian groups returned their finite maxima at
`delta=0.1`. This is a functional device/shape smoke, not paper evidence or a
performance claim.

The local MPS smoke separately verified stable ordering, APS, margin, and
Mondrian paths. Float64 dithering is intentionally rejected on MPS with a
controlled instruction to move cached scores/uniforms to CPU.

## Independent review

Review found and drove fixes for four Important numerical/device issues:

1. target-only margin/C-nu scoring accidentally materialized full `(N,V)`
   score tensors;
2. fp16 probability underflow broke min-p/C-margin equivalence;
3. subtracting the current mass from a rounded APS cumulative sum broke an
   fp16 boundary equivalence with top-p;
4. MPS cannot represent the required float64 dither result.

After repair, the reviewer reported no remaining Critical/Important findings
and passed random float16/32/64 property checks for the core equivalences.
