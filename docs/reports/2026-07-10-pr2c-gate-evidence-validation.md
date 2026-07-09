# PR-2c Gate Evidence Contract Validation

Validation target: `codex/pr2c-gate-evidence`.

## Contract

The content-addressed artifact stores lossless per-position rows for one
`(model, revision, domain, method, delta, evidence grade)` cell. It binds the
canonical method calibration, `[G]`/`[E]` position policy, model/domain
snapshot, frozen configs, PR-1 frequency/split/cross-corpus receipts, method
frequency buckets, randomization, tuning, and calibration inputs.

`[G]` requires exactly one unique document and cluster per test-manifest
document. `[E]` permits multiple positions but preserves document-to-cluster
identity. Comparators must share the same ordered test-row hash and core
provenance. Coverage, mean size, and frequency-group summaries are recomputed
from raw rows; old aggregate metrics are rejected.

The artifact does not implement bootstrap inference, gate thresholds, or
verdicts. Its references must be resolved and revalidated by the future PR-3
loader. Paper-grade execution remains blocked.

## Automated checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
git diff --check                                          PASS
python3 -m unittest discover tests                        171/171 PASS
```

The ten new tests cover canonical `+inf` serialization, content/filename hash
binding, tamper rejection, strict wrapper schemas, `[G]`/`[E]` identity rules,
vocabulary/provenance bounds, tuning and calibration identity, lossless summary
recomputation, shared test-row pairing, canonical order, and registry-role
partitioning without method-name substrings.

