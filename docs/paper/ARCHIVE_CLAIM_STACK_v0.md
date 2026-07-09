# Archived Claim Stack v0

This file preserves the pre-reconciliation claim stack for provenance only. It
is not the active paper framing. The current claim stack is
`docs/paper/CLAIM_STACK.md`.

## Retired Working Title

```text
nu-Sampling: Frequency-Calibrated Logit Prediction Sets for Language Model Decoding
```

Backup:

```text
Token-Level Truncation as Calibrated Support Testing for Language Model Decoding
```

## Retired Core Claim

```text
Truncation decoding can be formulated as token-level prediction-set construction over next-token logits. Fixed truncation rules impose global support constraints, while nu uses token-frequency calibrated margins. Split calibration turns this into a target-coverage prediction-set method and exposes the coverage/efficiency tradeoff directly.
```

## Why Retired

This stack centered `nu` too strongly and did not make the falsifiable premise
explicit. The current project is broader and stricter: frequency-offset margin
rules are useful only if token frequency adds information beyond logit margin,
and all evidence must compare calibrated methods against calibrated methods.
