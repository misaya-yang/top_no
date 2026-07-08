# Claim Stack

## Working Title

Primary:

```text
nu-Sampling: Frequency-Calibrated Logit Prediction Sets for Language Model Decoding
```

Backup:

```text
Token-Level Truncation as Calibrated Support Testing for Language Model Decoding
```

## Core Claim

```text
Truncation decoding can be formulated as token-level prediction-set construction over next-token logits. Fixed truncation rules impose global support constraints, while nu uses token-frequency calibrated margins. Split calibration turns this into a target-coverage prediction-set method and exposes the coverage/efficiency tradeoff directly.
```

## Contributions

1. Recast truncation decoding as token-level support testing / prediction-set construction.
2. Define a nu nonconformity score: a frequency-calibrated token-wise logit margin.
3. Validate with finite-sample token coverage, support-size efficiency, and downstream sampling metrics gated by prediction-set evidence.

## Claims Allowed After Current Code

- min-p is a fixed logit-margin rule.
- top-nsigma is a context-global logit-margin rule.
- nu is a token-specific frequency-calibrated logit-margin rule.
- conformal nu can be evaluated as a split-calibrated prediction set with measured true-token coverage and support size.

## Claims Not Allowed Yet

- Do not claim that nu improves GSM8K single-sample accuracy until the self-consistency script produces real pass@k/maj@k evidence.
- Do not claim the real LLM noise channel is identified.
- Do not use legacy `results/` artifacts as paper-ready evidence.
- Do not claim mathboost widens low-frequency math-token support without rechecking the formula and evidence.
