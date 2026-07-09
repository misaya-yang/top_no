# PR-2d Frequency Bucket Separation Validation

Validation target: `codex/pr2d-frequency-buckets`.

## Scope

The Phase-0 diagnostic uses exact integer B0..B8 log-count bands. Method-side
Mondrian/CovGap buckets use a separate canonical kind and a committed v1 policy
fitted from true-token mass on `D_tune`. The policy freezes K0=8, equal-count
ties, unseen handling, the strictest paper delta floor (`500` targets per
bucket), deterministic adjacent merging, and fail-closed minimum bucket count.

This slice does not build the method bucket artifact. That builder must bind
the exact `D_tune` target rows, pinned `D_freq` artifact, and policy hash; ad hoc
arrays remain unable to produce paper evidence.

## Automated checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
git diff --check                                          PASS
python3 -m unittest discover tests                        177/177 PASS
```

RTX 5090 server:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         177 run, 176 PASS,
                                                         1 MPS-only skip
```

## Real Qwen2.5-7B artifact smoke

The strict frequency loader reloaded the existing two-document Qwen2.5-7B
artifact (`V=152064`) and applied the diagnostic groups. It assigned 152,039
zero-count token types to B0 and 25 observed types (29 total token occurrences)
to B1. The committed method policy loaded with canonical SHA-256
`afc33682ee27ebbc75e29686438a5701bb03a9c9877520a67bb689cbbeac1636`.

This tiny artifact only verifies real-vocabulary shape, strict identity loading,
and boundary execution. Its distribution is not a corpus result or paper
evidence.

Independent review found and drove three repairs: small integer dtypes that
could overflow bucket boundaries are now rejected while int32/int64 compare in
int64; changing the method bucket kind bumped gate evidence to schema v2; and
the strict policy loader now locks exact numeric choices rather than merely
accepting positive variants. The reviewer found no remaining Critical/Important
issues and agreed that deferring the unbound mass-quantile builder is the
correct fail-closed choice.

