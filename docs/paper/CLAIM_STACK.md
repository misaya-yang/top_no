# Claim Stack

## Working Title

Primary:

```text
Frequency-Offset Margin Rules for Calibrated Language Model Decoding
```

Backup:

```text
Does Corpus Frequency Add Information Beyond Logit Margin?
A Conformal Audit of Next-Token Prediction Sets
```

## Core Claim

```text
Token truncation in language-model decoding constructs a next-token support set. The core research question is whether fixed, external-corpus token frequency carries predictive information beyond logit margin. If it does, general rules measurable in `(margin, frequency)` may improve the calibrated coverage/size frontier; additive offsets are justified only if the measured reliability surface has an approximately horizontal-shift structure.
```

## Formal Object

Let `m_i(x) = s_max(x) - s_i(x)` be the logit margin for token `i` in context
`x`, and let `n_i` be a fixed external-corpus count. The general scientific
object is:

```text
h(m, n) = P(Y = i | m_i(X) = m, n_i = n)
A_h(x, i) = -h_hat(m_i(x), n_i)
S_h(x) = { i : A_h(x, i) <= q_hat }
```

`h_hat` is fit only on `D_tune`, and `q_hat` is calibrated only on `D_cal`.
The interpretable additive subfamily is:

```text
A_g(x, i) = m_i(x) - g(n_i)
S_g(x) = { i : A_g(x, i) <= q_hat }
```

It is oracle-shaped only under a single-index condition such as
`h(m,n) = rho(m-g(n))` with `rho` monotone. Frequency interaction is necessary,
not sufficient, for an additive offset to be optimal. Candidate restrictions
include `g=0` (C-margin), a signed inverse-square-root or log-frequency ansatz,
a learned additive `g`, and frequency-Mondrian bucket thresholds. General
learned-`h` and additive learned-`g` must be reported separately.

Conformal calibration gives finite-sample marginal coverage for any measurable
score under the split-exchangeability assumptions. Coverage by itself is not
evidence for the score. Evidence must be measured through set size,
coverage-size Pareto frontier, or conditional/frequency-bucket behavior.

## Falsifiable Premise

The entire novelty license is:

```text
h(m, n) = P(Y = i | margin m_i = m, frequency n_i = n)
```

varies with `n` at fixed `m`. Phase 0 should estimate this surface directly
before expensive downstream generation. If `h(m, n)` is effectively a function
of `m` alone at the study's power, the frequency method becomes a
negative/audit result. This null does not by itself prove C-margin is
frontier-optimal: Phase 0 must also test monotonicity of `h_m(m)` and report the
smallest interaction the experiment could reliably detect.

## Contributions

1. Recast truncation decoding as token-level prediction-set construction over next-token logits.
2. Provide a direct `margin x frequency` diagnostic for whether frequency carries additional information.
3. Separate the general learned-`h(m,n)` oracle approximation from additive learned-`g`, frequency-Mondrian, and signed nu/log-frequency ansatzes.
4. Compare calibrated methods against calibrated baselines on coverage/size frontier and frequency-bucket coverage before claiming downstream generation value.

## Claims Allowed After Current Code

- min-p is a fixed logit-margin rule.
- top-nsigma is a context-global logit-margin rule.
- nu is a token-specific frequency-indexed logit-margin score; only an independently calibrated version is “calibrated.”
- the current conformal-nu helpers implement a marginal score/set construction, but the paper runner remains blocked pending protocol repairs.
- current smoke outputs are engineering checks, not paper evidence.

## Claims Not Allowed Yet

- Do not claim conformal coverage itself as novelty; it is generic for any valid nonconformity score.
- Do not claim frequency helps until the `margin x frequency` diagnostic and calibrated baselines support it.
- Do not claim that any frequency interaction proves an additive offset is optimal; test the horizontal-shift structure and retain learned-`h` as the general comparator.
- Do not claim a null frequency effect proves C-margin is optimal without testing margin monotonicity and statistical power.
- Do not claim that nu improves GSM8K single-sample accuracy until the self-consistency script produces real pass@k/maj@k evidence.
- Do not claim the real LLM noise channel is identified.
- Do not use legacy `results/` artifacts as paper-ready evidence.
- Do not claim mathboost widens low-frequency math-token support without rechecking the formula and evidence.
