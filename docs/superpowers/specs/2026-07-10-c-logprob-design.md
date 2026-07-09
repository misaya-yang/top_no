# PR-2e C-logprob Baseline Design

C-logprob uses the canonical nonconformity score
`A(x,i) = -log p_i(x) = logsumexp(s(x)) - s_i(x)`. Target calibration gathers
only the observed target logit and never materializes an `(N,V)` score matrix.
Candidate construction uses the full score only at prediction time.

Float16/bfloat16/float32 inputs accumulate in float32; float64 is preserved.
The registry applies the same explicit score dither and finite-sample quantile
as other global methods. With zero dither, `A <= q_hat` is exactly the
epsilon-probability set `p_i >= exp(-q_hat)`.

This implements one mandatory non-frequency baseline. It does not change the
runner, gate, or paper-grade stop condition.

