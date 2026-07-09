# PR-1d Validation Record

Validation target: `codex/pr1d-cross-corpus`, including independent-review
repairs after the initial `1188654` implementation commit.

## Correctness checks

Local macOS, Python 3.9.6, Torch 2.8.0:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python3 -m unittest discover tests                        121/121 PASS
```

RTX 5090 server, isolated worktree, project virtualenv:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         121/121 PASS
```

The focused tests include exact duplicates, normalized duplicates, a pair at
the exact Jaccard 0.8 boundary that MinHash-LSH alone misses, a pair below the
threshold, input reordering, missing/extra/modified rows, weak split parameters,
wrapper tampering, a self-consistent forged zero-match artifact, fixed scope,
CLI refusal, protocol integration, and pre-model fail-closed behavior.

## Synthetic scaling probe

The server probe built a PR-1b split for 2,000 distinct synthetic evaluation
documents and then recomputed PR-1d against 2,000 distinct frequency documents.
Each document contained 48 whitespace tokens; the corpora were deliberately
disjoint. This is an engineering benchmark, not paper evidence.

```json
{
  "candidate_pairs": 0,
  "cross_audit_seconds": 6.475,
  "evaluation_input_documents": 2000,
  "evaluation_documents": 2000,
  "exact_comparisons": 0,
  "frequency_documents": 2000,
  "matches": 0,
  "max_rss_mb": 66.9,
  "split_seconds": 3.211
}
```

Interpretation: the fixed completeness fallback stays practical on a sparse,
fully disjoint 2k-by-2k workload. This does not establish worst-case scaling;
real corpus preparation should record candidate density and memory before
expanding to substantially larger manifests.
