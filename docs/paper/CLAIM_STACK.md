# Claim Stack

## Working Title

Primary:

```text
Frequency-Offset Margin Rules for Calibrated Language Model Decoding
```

Backup:

```text
Token Truncation as Prediction-Set Construction for Language Model Decoding
```

## Core Claim

```text
Token truncation in language-model decoding is prediction-set construction. The core research question is whether token frequency carries information beyond the logit margin used by standard truncation rules. If it does, frequency-offset margin rules can improve the calibrated coverage/size frontier or frequency-bucket coverage at matched marginal coverage.
```

## Formal Object

Let `m_i(x) = s_max(x) - s_i(x)` be the logit margin for token `i` in context
`x`. The active family is:

```text
A(x, i) = m_i(x) - g(n_i)
S(x) = { i : A(x, i) <= q_hat }
```

where `n_i` is a fixed side-information count and `q_hat` is a split-conformal
quantile. Candidate instantiations:

```text
g(n) = 0                         calibrated margin / min-p null
g(n) = kappa / sqrt(n + alpha)   signed nu ansatz
g(n) = bucket-specific offset    frequency-Mondrian / grouped calibration
g(n) = learned offset            tuning-split plug-in score
```

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
of `m` alone, the frequency-offset method should be treated as a negative/audit
result rather than sold as a new top decoding method.

## Contributions

1. Recast truncation decoding as token-level prediction-set construction over next-token logits.
2. Provide a direct `margin x frequency` diagnostic for whether frequency carries additional information.
3. Define a calibrated family of frequency-offset margin scores, with nu as one signed parametric ansatz rather than the whole method.
4. Compare calibrated methods against calibrated baselines on coverage/size frontier and frequency-bucket coverage before claiming downstream generation value.

## Claims Allowed After Current Code

- min-p is a fixed logit-margin rule.
- top-nsigma is a context-global logit-margin rule.
- nu is a token-specific frequency-offset logit-margin score.
- conformal nu can be evaluated as a split-calibrated prediction set with measured true-token coverage and support size.
- current smoke outputs are engineering checks, not paper evidence.

## Claims Not Allowed Yet

- Do not claim conformal coverage itself as novelty; it is generic for any valid nonconformity score.
- Do not claim frequency helps until the `margin x frequency` diagnostic and calibrated baselines support it.
- Do not claim that nu improves GSM8K single-sample accuracy until the self-consistency script produces real pass@k/maj@k evidence.
- Do not claim the real LLM noise channel is identified.
- Do not use legacy `results/` artifacts as paper-ready evidence.
- Do not claim mathboost widens low-frequency math-token support without rechecking the formula and evidence.
