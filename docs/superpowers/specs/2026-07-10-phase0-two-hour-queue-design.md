# Two-Hour Phase-0 Queue Design

## Goal

Prepare a fail-closed GPU queue that uses roughly two hours of RTX 5090 time to
answer the project's decisive pilot question:

```text
At fixed raw-logit margin, does pinned external-corpus token frequency carry a
replicable signal about the realized next token?
```

The queue is a decision pilot. Its outputs may guide Plan A versus Plan B, but
they are not paper evidence until the pre-registered 300,000-position,
3,000-document cells are rerun at full scale.

## Chosen Approach

Use a replication-first 2 x 2 matrix instead of one high-powered cell or the
unfinished calibrated prediction-set gate:

1. Qwen2.5-3B on the frozen web domain.
2. Qwen2.5-3B on the frozen math domain.
3. Qwen2.5-7B on the same web domain.
4. Qwen2.5-7B on the same math domain.

Each cell receives a 25-minute graceful wall-time budget. The queue reserves
the remaining time for preflight, model teardown, replay analysis, and the
decision memo. Cells process deterministic stride-4 empirical `[E]` positions
from `D_tune`; this gives substantially more power than one position per
document while keeping theorem-grade `[G]` rows untouched.

The same forward pass must support four analyses without additional GPU work:

- the margin-by-frequency `NUM / DEN` reliability grid;
- deterministic tune-half stability by document hash;
- a fixed frequency-label permutation negative control;
- the signed horizontal-offset direction and a non-additivity flag.

## Alternatives Rejected

### One large model/domain cell

This maximizes within-cell power but cannot show cross-domain or scale
replication. A frequency effect seen only once is too easy to explain as a
tokenizer, domain, or data artifact.

### Complete the calibrated gate before using the server

This is the paper-grade destination, but `ts_aps`, `cns`, `learned_h`, method
buckets, suffstats replay, evaluator emission, and clustered gate inference are
not all integrated. Finishing them first would miss the user's two-hour server
window and still leave the central frequency premise unmeasured.

### Run the legacy GPU queue

The legacy queue uses pre-PR-1 data flow and an uncalibrated comparison gate.
It cannot support the active claim and must remain blocked.

## Inputs And Provenance

The server data root defaults to `/root/autodl-tmp/top_no_phase0` and may be
overridden with `TOPNO_PHASE0_DATA_ROOT`. It must contain:

```text
domains/web/documents.jsonl
domains/web/splits/receipt.json
domains/web/splits/tune.manifest.json
domains/web/splits/cal.manifest.json
domains/web/splits/test.manifest.json
domains/web/cross_corpus.json
domains/math/documents.jsonl
domains/math/splits/receipt.json
domains/math/splits/tune.manifest.json
domains/math/splits/cal.manifest.json
domains/math/splits/test.manifest.json
domains/math/cross_corpus.json
frequency/documents.jsonl
frequency/freq.manifest.json
frequency/qwen2.5-3b/<artifact-id>.json
frequency/qwen2.5-7b/<artifact-id>.json
```

There must be exactly one frequency-table sidecar in each model directory.
Every sidecar must retain its content-addressed filename. The queue validates
the frozen split receipt, exact text binding, pairwise manifest disjointness,
frequency-table identity, model/tokenizer revision, and recomputed cross-corpus
near-duplicate receipt before loading model weights.

The fixed revisions are:

```text
Qwen/Qwen2.5-3B  3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
Qwen/Qwen2.5-7B  d149729398750b98c0af14eb82c78cfe92750796
```

Missing or ambiguous artifacts are fatal. There is no network download,
evaluation-corpus frequency fallback, sequential split, or synthetic fallback.

## Components

### `experiments/protocol.py`

Add `validate_phase0_inputs(config)`. It performs the PR-1 provenance checks
currently embedded in `validate_protocol_inputs` and returns a Phase-0 protocol
receipt instead of raising the PR-2/PR-3 blocker. The paper-grade prediction-set
entrypoint continues to call the same shared validation and then raises
`blocked_pending_pr2_pr3` until its own blockers are complete.

### `experiments/phase0_stats.py`

Provide pure tensor/statistics code with no model or dataset dependency:

- frozen margin edges and diagnostic B0..B8 frequency groups;
- per-document `NUM` and `DEN` accumulation;
- deterministic document-half assignment;
- deterministic token-bucket permutation control;
- cell masking with a minimum true-token numerator count;
- aligned-offset estimation over margin `[2, 12]`;
- non-additivity detection from sign changes across valid subwindows;
- canonical JSON serialization and resumable additive merging.

All counters use int64. Raw margins are computed in fp32. Excluded special and
control token IDs contribute to neither `NUM` nor `DEN`.

### `experiments/phase0_reliability.py`

Load one pinned local-only model and one validated cell. Iterate documents in
canonical `doc_id` order, tokenize without special-token injection, choose
stride-4 eligible positions after 16 context tokens, and perform prefix-only
next-token forwards. The runner streams rows into `phase0_stats` and never
stores full-vocabulary logits on disk.

After every completed document it writes an atomic checkpoint containing
per-document sufficient statistics and provenance hashes. `--wall-seconds`
causes a graceful checkpoint and a successful `PARTIAL` result; integrity or
compatibility errors fail the cell. A resumed run refuses any changed config,
commit, runtime receipt, input artifact, or model identity.

### `experiments/summarize_phase0_queue.py`

Read the four cell summaries and emit one machine-readable decision memo. A
pilot cell is informative only when both tune halves contain at least three
valid cells in margin `[2, 12]`. The memo reports effect magnitude, direction,
half stability, permutation-control separation, cross-domain sign agreement,
cross-scale sign agreement, and `PLAN_A_PILOT`, `PLAN_B_PILOT`, or
`INSUFFICIENT`.

The pilot verdict never claims the full pre-registered Plan-A criterion. It
unlocks only a recommendation to run the full Phase-0 matrix.

### `scripts/run_phase0_two_hour_queue.sh`

Set offline Hugging Face mode, deterministic CUDA environment variables,
project paths, and a 110-minute queue deadline. Run the four isolated Python
processes in this order:

1. 3B web;
2. 3B math;
3. 7B web;
4. 7B math.

The first two cells receive priority. If startup or slower-than-expected
throughput exhausts the global budget, later cells are marked `NOT_STARTED`
instead of overrunning the two-hour window. Existing valid checkpoints resume.
The final summarizer always runs when at least one cell completed or checkpointed.

### Configs

Commit one matrix config containing model IDs, revisions, relative artifact
paths, stride, context length, batch sizes, per-cell wall time, margin edges,
valid-cell floor, seed, and output names. Paths resolve under the server data
root, so no machine-specific secret or mutable absolute path enters evidence.

## Output Contract

Each cell writes under `results/phase0_two_hour/<cell-key>/`:

```text
checkpoint.pt
phase0_summary.json
runtime/<runtime-artifact-id>.json
run.log
```

The queue writes:

```text
results/phase0_two_hour/queue_status.json
results/phase0_two_hour/decision_memo.json
```

Every summary records `evidence_grade="E-pilot"`, `paper_citable=false`, git
commit, runtime receipt ID, effective config hash, model revision, frequency
artifact ID, split and manifest hashes, position count, document count, stride,
elapsed time, completion status, and every analysis statistic.

## Error Handling

- Preflight failure: exit before CUDA/model allocation.
- CUDA or pinned-model cache mismatch: fail the affected cell and stop the queue.
- Wall-time exhaustion: atomically checkpoint and mark `PARTIAL`; continue only
  when enough global time remains.
- Corrupt or stale checkpoint: refuse resume and preserve the file for audit.
- Numerical NaN/Inf or counter overflow: fail the cell; never coerce or skip.
- Fewer than two completed domain cells: summarizer returns `INSUFFICIENT`.

## Verification

Tests cover exact margin boundaries, B0..B8 mapping reuse, exclusion handling,
int64 accumulation, document-half determinism, permutation determinism,
additive checkpoint merge, aligned-shift recovery on synthetic `+/-0.5`-nat
signals, null behavior, non-additivity flagging, provenance refusal, wall-time
partial completion, and queue budget/order parsing.

Required local checks:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests
git diff --check
```

The server preflight additionally checks the RTX 5090, available VRAM and disk,
the exact cached Qwen revisions, deterministic CUDA policy, and all input
artifacts before the user-funded wall clock starts.
