# PR-2c Gate Evidence Contract Design

## Purpose

Freeze the lossless, per-position handoff from calibrated prediction-set
evaluation to the future PR-3 gate. Existing aggregate metrics cannot support
document-clustered bootstrap inference and must never be accepted as gate
evidence.

## Artifact contract

One content-addressed JSON artifact represents one
`(model, revision, domain, method, delta, evidence grade)` cell. It binds:

- canonical method registry version/key and its immutable calibration result;
- canonical method-registry content hash, preventing role/status drift without
  an explicit artifact change;
- `[G]` or `[E]` evidence grade;
- effective config, frequency table, split receipt, cross-corpus receipt,
  calibration/test manifests, and method-side frequency-bucket artifact IDs;
- code/model commits, model family, frozen domain snapshot, primary-config
  preregistration, and gate-threshold hashes;
- optional tuning artifact ID, required for methods with selected
  hyperparameters or buckets;
- model/domain identity and vocabulary size;
- one lossless record per selected test position: document, cluster, target
  position/token, coverage, set size, and method-side frequency group.

Threshold `+inf` is serialized as the JSON string `+inf`; NaN and `-inf` are
invalid. The wrapper and filename both bind the canonical SHA-256 identity.
The method-independent ordered test-row hash must match across comparators.

## Invariants

- `[G]` requires exactly one row per unique document and cluster.
- `[E]` may contain multiple positions per document, but `(doc_id, position)`
  is unique and each document maps to one cluster.
- Method/calibration keys, deltas, randomization policy, parameters, Mondrian
  axes/counts, and finite/vacuous thresholds must agree with the registry.
- Target IDs and set sizes must fit the recorded vocabulary.
- Paper provenance IDs are canonical SHA-256 strings; model revision is a
  pinned 40-hex commit.
- Old aggregate metrics, unknown fields, tampering, registered-but-unavailable
  methods, and tuned methods without a tuning artifact fail closed.

## Derived views

Coverage, mean set size, and frequency-group coverage/size are recomputed from
raw records. Gate comparator partitions use registry `paper_role`, never
display names or substrings. PR-3 bootstrap and verdict logic remain out of
scope, and the paper runner remains blocked. The future gate loader must also
resolve these references and rerun the existing PR-1 validators; a matching
wrapper hash alone is never a green provenance verdict.
