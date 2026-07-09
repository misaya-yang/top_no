# Manifest-To-Text Binding Design

## Goal

Make a split receipt prove which frozen source text corresponds to every
tune/calibration/test manifest row before any model is allocated. This closes
the identity gap between PR-1b artifacts and future forward passes without
pretending that independently constructed `D_freq` and evaluation cluster IDs
prove cross-corpus near-duplicate disjointness.

The lower-level `[G]` forward tokenizes with `add_special_tokens=false` and no
truncation, selects one target per document, feeds only the bounded prefix
ending at `target_index - 1`, and retains document/cluster/position metadata.
Calibration and test roles use distinct salts and independent model calls.

## Frozen document store

The source store is strict JSONL with exactly `doc_id` and `text` per row. Its
semantic identity is the PR-1b order-independent hash of sorted
`(doc_id, raw-UTF8-content-sha256)` rows. JSON whitespace and row order may
change; IDs or text may not.

`experiments/document_store.py` loads the store, verifies that semantic hash
against `SplitBuildReceipt.input_documents_sha256`, then binds each retained
manifest representative by exact ID and raw-text SHA-256. Missing, extra,
duplicate, modified, or wrong-source documents fail without printing text.

The binder returns immutable `BoundDocument` rows carrying role, document ID,
cluster ID, content hash, and text. These are the only admissible inputs to the
future nonlegacy document-aware forward helper.

## Protocol boundary

Nonlegacy configs add `split_receipt` and `document_jsonl`. Validation order is:

1. frequency artifact and four role manifests;
2. pairwise ID/content/cluster tripwire;
3. split receipt and its role-manifest hashes;
4. frozen JSONL semantic hash and representative text binding;
5. explicit cross-corpus cluster-namespace stop.

The stop remains before model allocation:
`blocked_pending_cross_corpus_cluster`. PR-1d must build a joint four-way
cluster proof, or run cross-corpus threshold-complete candidate checks and
rebuild the frequency manifest/table. Merely comparing independently generated
cluster-ID strings is forbidden.

## Acceptance criteria

- Input JSONL row order does not affect its semantic identity.
- Any text/ID change, duplicate, missing, or extra document is rejected.
- Receipt role manifests must exactly match configured manifests.
- Every retained representative binds to exactly one source row and content
  hash before model allocation.
- Prefix-only forward batches never contain the target token, use explicit
  left-padding masks/position IDs, and emit exactly one `[G]` row per document.
- Valid inputs advance the fail-closed reason from PR-1c to the explicit
  cross-corpus clustering blocker.
- Legacy smoke behavior remains unchanged and noncitable.

## Non-goals

- No paper-grade run or cross-corpus disjointness claim.
- No tokenizer-specific eligibility artifact; target exclusions remain an
  explicit input until the EOS-preserving policy is frozen.
- No PR-2 method registry, suffstats, or gate rewrite.
