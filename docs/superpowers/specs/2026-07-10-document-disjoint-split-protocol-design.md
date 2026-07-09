# Document-Disjoint Split Protocol Design

## Goal

Build the pure deterministic library needed to make evaluation documents,
near-duplicate clusters, and sampled next-token positions content-addressed
and auditable. Connecting materialized source text to evaluator forward passes
is deliberately isolated as PR-1c; this slice therefore does not unlock model
runs or paper claims.

## Fixed protocol

- Input is JSONL with exactly one non-empty `doc_id` and `text` per row.
- Exact content identity is SHA-256 over UTF-8 text bytes.
- Near duplicates use normalized token 13-gram shingles, deterministic
  MinHash-LSH candidates, and exact Jaccard confirmation at `>= 0.8`.
- Near-duplicate connected components are clusters. Only the canonical
  representative (minimum `(content_sha256, doc_id)`) is retained.
- The representative ID and a committed global salt are hashed into fixed
  bands: tune `[0, 40)`, calibration `[40, 65)`, test `[65, 100)`.
- Calibration and guarantee-grade test rows use exactly one target position
  per document. The target is drawn uniformly from token indices
  `{16, ..., length - 1}` using a SHA-256-derived per-document seed.
- Empirical pooled rows use target indices `16, 20, 24, ...` (stride 4) and
  retain document and cluster identity. They are never described as having the
  one-position exchangeability guarantee.

The existing manifest wire schema remains unchanged so PR-1a frequency
artifacts retain their identity. Instead, a mandatory `SplitBuildReceipt`
cryptographically binds source/input hashes, normalization, MinHash seeds and
layout, exact threshold, representative policy, salt, bands, cluster membership
digest, and all three role-manifest hashes. Even when a parameter change happens
to produce the same assignments, the receipt identity changes.

## Artifacts

`experiments/splits.py` owns three layers:

1. canonical manifest I/O and pairwise intersection checks;
2. deterministic clustering and tune/cal/test construction;
3. a canonical build receipt and receipt hashing/I/O.

`experiments/prepare_document_splits.py` is the reproducible CLI from source
JSONL to manifests plus receipt. It uses only the standard library and exposes
protocol-changing knobs explicitly.

## Evaluator boundary

PR-1b does not pretend that a manifest proves which text was fed to a model.
The evaluator remains fail-closed, with its terminal reason advanced to
`blocked_pending_pr1c`. PR-1c must require the receipt, bind manifest rows to
materialized source text by ID and content hash, run independent cal/test
forwards, preserve document/cluster/evidence labels, and remove sequential
skip/take from every nonlegacy path.

## Error handling

Fatal errors name the invariant and never print source text. Duplicate IDs,
invalid parameters, empty eligible position sets, receipt/manifest mismatch,
and frequency/evaluation intersections all stop construction.

## Acceptance criteria

- Split artifacts are identical under input reordering.
- Exact and near duplicates cannot straddle roles.
- Observed assignment follows the fixed 40/25/35 hash bands.
- Receipt or manifest tampering fails closed.
- One-position sampling is deterministic, in range, and approximately uniform
  across many document IDs; pooled sampling uses stride 4.
- The evaluator remains blocked before model allocation with an explicit PR-1c
  reason; no partial library is mislabeled as end-to-end data binding.
- Full compile, shell syntax, and unit-test suites pass.

Cluster IDs are meaningful only inside the receipt's cluster namespace.
Independently clustering `D_freq` and evaluation data cannot prove that a near
duplicate is absent merely because component IDs differ. PR-1c must either
require a joint four-way construction receipt or perform cross-corpus LSH plus
exact-Jaccard checking and rebuild the frequency manifest/table if it changes.

## Explicit non-goals

- No frequency/Mondrian bucket redesign in this slice.
- No materialized text store or evaluator rewiring; those are PR-1c.
- No calibrated method registry, suffstats cache, frozen-config registry, or
  gate rewrite; those remain PR-2 and PR-3.
- No paper-grade GPU run and no reinterpretation of legacy results.
