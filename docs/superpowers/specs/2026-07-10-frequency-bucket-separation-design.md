# PR-2d Frequency Bucket Separation

## Frozen diagnostic policy

Phase 0 uses the exact Fable5 log-count bands derived only from the pinned
`D_freq` count vector:

```text
B0: 0
B1: 1..9
B2: 10..99
B3: 100..999
B4: 1,000..9,999
B5: 10,000..99,999
B6: 100,000..999,999
B7: 1,000,000..9,999,999
B8: >=10,000,000
```

The implementation is an integer boundary map, not floating-point `log10`, so
large counts and exact edges are deterministic across devices.

## Method buckets are a different artifact

Frequency-Mondrian calibration and CovGap use true-token-mass quantiles fitted
on `D_tune`, with every final group satisfying the pre-registered
`ceil(5 / delta)` floor. They must never reuse the diagnostic group IDs.

Fable5 fixes the data source, ordering variable, and floor, but does not freeze
all construction choices needed for a deterministic artifact. This repository
therefore labels the following as new protocol decisions, not Fable5 claims,
and freezes them in `configs/frequency_bucket_policy_v1.json`:

- initial `K0=8`, with equal-frequency levels never split;
- cumulative true-token-mass right-crossing cuts on `D_tune`;
- `n=0` merged into the lowest nonzero-frequency bucket;
- one bucket table for the paper delta grid, using its strictest
  `delta=0.01`, hence the hard floor `ceil(5 / delta)=500`;
- repeatedly merge the smallest under-floor bucket (lower-frequency tie-break)
  into the lower-mass neighbor (lower-frequency tie-break);
- reject if at least two valid buckets cannot remain.

The strict loader and canonical hash freeze these decisions. The builder is
still deliberately deferred: it must bind exact `D_tune` target rows and the
pinned frequency artifact rather than accept ad hoc arrays. The gate-evidence
schema imports the canonical method kind instead of duplicating a similar
string.
