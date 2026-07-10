# Two-Hour Phase-0 Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, resumable four-cell GPU queue that spends at most 110 minutes measuring whether external token frequency adds stable signal beyond raw-logit margin.

**Architecture:** Extract a Phase-0 boundary from the audited PR-1 validation, stream causal document logits into pure int64 sufficient statistics, and analyze every claim diagnostic from cached counts. Isolated model/domain processes run under one shell deadline and feed a conservative Python decision memo.

**Tech Stack:** Python 3.12, PyTorch 2.9+, Transformers, NumPy, repository artifact modules, `unittest`, Bash.

## Global Constraints

- Preserve existing CLI and artifact contracts, especially `blocked_pending_pr2_pr3`.
- Use frozen `D_freq` and `D_tune`; no legacy, network, synthetic, or eval-pool fallback.
- Pin Qwen2.5-3B to `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` and Qwen2.5-7B to `d149729398750b98c0af14eb82c78cfe92750796`.
- Mark outputs `evidence_grade="E-pilot"`, `paper_citable=false`.
- Use fp32 margins, int64 counters, deterministic CUDA, atomic checkpoints, and fail-closed validation.
- Add no dependency and no generated `results/` file.

---

### Task 1: Expose Phase-0 PR-1 Validation

**Files:** Modify `experiments/protocol.py`; modify `tests/test_prediction_protocol.py`.

**Interfaces:** Produce `validate_phase0_inputs(config: dict[str, Any]) -> dict[str, Any]`. Keep `validate_protocol_inputs` blocked after it reuses the same checks.

- [ ] **Step 1: Write the failing test**

```python
def test_phase0_protocol_returns_after_pr1_checks(self):
    receipt = validate_phase0_inputs(self.valid_nonlegacy_config())
    self.assertEqual(receipt["protocol_version"], "icml2027-phase0-pilot-v1")
    self.assertEqual(receipt["evidence_grade"], "E-pilot")
    self.assertFalse(receipt["paper_citable"])
```

- [ ] **Step 2: Verify RED**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_prediction_protocol.PredictionProtocolTests.test_phase0_protocol_returns_after_pr1_checks`

Expected: import failure because `validate_phase0_inputs` is missing.

- [ ] **Step 3: Implement the extraction**

```python
def validate_phase0_inputs(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("allow_legacy_protocol", False) is not False:
        raise ValueError("Phase-0 pilot forbids allow_legacy_protocol")
    reference, manifests, cross_receipt = _validate_pr1_inputs(config)
    return {
        "protocol_version": "icml2027-phase0-pilot-v1",
        "evidence_grade": "E-pilot",
        "paper_citable": False,
        "effective_config_sha256": effective_config_sha256(config),
        "frequency_table": reference,
        "manifest_sha256s": {role: manifest_sha256(manifests[role]) for role in ("freq", "tune", "cal", "test")},
        "cross_corpus_receipt_sha256": cross_corpus_receipt_sha256(cross_receipt),
    }
```

Move the current nonlegacy checks into `_validate_pr1_inputs`; do not reorder them.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_prediction_protocol`

Commit: `git add experiments/protocol.py tests/test_prediction_protocol.py && git commit -m "feat: expose phase0 protocol validation"`

### Task 2: Implement Pure Phase-0 Statistics

**Files:** Create `experiments/phase0_stats.py`; create `tests/test_phase0_stats.py`.

**Interfaces:** Produce `GridSpec`, `DocumentGridStats`, `accumulate_document`, `merge_document_stats`, `analyze_grid`, `document_half`, `permuted_frequency_groups`.

- [ ] **Step 1: Write RED counting tests**

```python
def test_counts_allowed_candidates_and_true_target(self):
    result = accumulate_document("doc-a", torch.tensor([[4., 3., 1., 0.]]), torch.tensor([1]), torch.tensor([0, 9, 10, 10_000_000]), grid=GridSpec.default(), excluded_token_ids={3}, permutation_seed=17)
    self.assertEqual(int(result.den.sum()), 3)
    self.assertEqual(int(result.num.sum()), 1)
    self.assertEqual(result.num.dtype, torch.int64)
```

Add one-behavior tests for exact margin edges, B0..B8 reuse, deterministic halves/permutation, additive merge, overflow, and NaN rejection.

- [ ] **Step 2: Verify RED**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_stats`

Expected: missing-module failure.

- [ ] **Step 3: Implement accumulation**

```python
@dataclass(frozen=True)
class GridSpec:
    margin_edges: tuple[float, ...]
    frequency_labels: tuple[str, ...] = DIAGNOSTIC_BUCKET_LABELS
    min_true_count: int = 20

@dataclass(frozen=True)
class DocumentGridStats:
    doc_id: str
    half: int
    num: torch.Tensor
    den: torch.Tensor
    perm_num: torch.Tensor
    perm_den: torch.Tensor
    n_positions: int
```

Flatten `(margin_bin, frequency_group)` and use `torch.bincount`; check int64 capacity before merging.

- [ ] **Step 4: Write RED shift tests**

Construct `log h_b(m)=c-(m-g_b)` grids for `g_b=+0.5`, `-0.5`, and `0`; assert recovery within `0.10`, null near zero, and sign-changing subwindows produce `non_additive=true`.

- [ ] **Step 5: Implement analysis**

```python
def analyze_grid(num: torch.Tensor, den: torch.Tensor, *, grid: GridSpec, reference_group: int | None = None) -> dict[str, object]:
    """Mask underfloor cells and fit horizontal shifts on margin [2,12]."""
```

Interpolate the reference log-rate curve on a fixed `[-2,2]` shift grid with `0.05` spacing and minimize denominator-weighted squared error.

- [ ] **Step 6: Verify GREEN and commit**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_stats`

Commit: `git add experiments/phase0_stats.py tests/test_phase0_stats.py && git commit -m "feat: add phase0 reliability statistics"`

### Task 3: Implement Resumable Cell Runner

**Files:** Create `experiments/phase0_reliability.py`; create `tests/test_phase0_runner.py`.

**Interfaces:** Consume `--config`, `--cell`, `--data-root`, `--output-root`, `--wall-seconds`, `--created-by-commit`, `--preflight-only`. Produce `checkpoint.pt`, `phase0_summary.json`, runtime receipt, and `COMPLETE` or graceful `PARTIAL`.

- [ ] **Step 1: Write RED runner tests**

With fake tokenizer/model, assert target `ids[t]` uses `logits[t-1]`, stride-4 starts after 16 tokens, excluded targets disappear, and a fake clock checkpoints after one document.

```python
rows = list(iter_document_logits(model, tokenizer, documents, max_length=64, stride=4, batch_size=2, excluded_target_ids={0}))
self.assertTrue(all(row.targets.numel() == row.logits.shape[0] for row in rows))
```

- [ ] **Step 2: Verify RED**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_runner`

- [ ] **Step 3: Implement preflight and causal extraction**

Resolve paths below `data_root`, require exactly one content-addressed frequency sidecar, call `validate_phase0_inputs`, bind tune text, and load local-only config/tokenizer. Tokenize without special injection, right-pad, run one causal document forward, and gather `output.logits[row,target_index-1,:]`.

- [ ] **Step 4: Implement atomic resume**

```python
def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
```

Bind config hash, commit, runtime artifact, protocol receipt, cell, revision, processed docs, and serialized statistics; refuse any mismatch.

- [ ] **Step 5: Verify GREEN and commit**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_runner`

Commit: `git add experiments/phase0_reliability.py tests/test_phase0_runner.py && git commit -m "feat: add resumable phase0 cell runner"`

### Task 4: Implement Conservative Summarizer

**Files:** Create `experiments/summarize_phase0_queue.py`; create `tests/test_phase0_summary.py`.

**Interfaces:** Consume cell summaries; produce `decision_memo.json` with `PLAN_A_PILOT`, `PLAN_B_PILOT`, or `INSUFFICIENT`.

- [ ] **Step 1: Write RED verdict tests**

```python
def test_four_replicated_cells_pass_plan_a(self):
    self.assertEqual(summarize_cells(self.four_cells(effect=0.45, perm_effect=0.05))["verdict"], "PLAN_A_PILOT")

def test_one_cell_is_insufficient(self):
    self.assertEqual(summarize_cells(self.one_cell())["verdict"], "INSUFFICIENT")
```

Add a two-domain small-effect `PLAN_B_PILOT` test.

- [ ] **Step 2: Verify RED**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_summary`

- [ ] **Step 3: Implement rule and commit**

Plan A requires four informative cells, effect `>=0.30`, stable half signs, cross-domain/scale sign agreement, and permutation effect below half the real effect in three cells. Plan B requires two informative domains with effect `<0.30`; otherwise insufficient.

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_summary`

Commit: `git add experiments/summarize_phase0_queue.py tests/test_phase0_summary.py && git commit -m "feat: summarize phase0 pilot evidence"`

### Task 5: Add Frozen Matrix And Deadline Queue

**Files:** Create `configs/phase0_two_hour_qwen.json`; create `scripts/run_phase0_two_hour_queue.sh`; create `tests/test_phase0_queue_config.py`; modify `README.md`.

**Interfaces:** Consume `TOPNO_PHASE0_DATA_ROOT`, optional `PHASE0_OUTPUT_ROOT`, `PYTHON_BIN`; produce four cell runs, `queue_status.json`, and `decision_memo.json` within 6,600 seconds.

- [ ] **Step 1: Write RED config tests**

Assert order `3b_web,3b_math,7b_web,7b_math`, pinned revisions, cell seconds 1500, queue seconds 6600, stride 4, relative paths, 3B batch 2, 7B batch 1, and no legacy flag.

- [ ] **Step 2: Verify RED**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_queue_config`

- [ ] **Step 3: Implement config, queue, and docs**

Use `max_length=512`, `min_context=16`, 80,000 positions for 3B and 50,000 for 7B. Preflight all cells before starting the deadline. Set `PYTHONHASHSEED=1729`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline Hub variables, and `HF_HOME=/root/autodl-tmp/huggingface`. Reserve 300 seconds for summary and skip cells below 360 remaining seconds.

Document:

```bash
cd /root/neurips2027
source /root/autodl-tmp/venvs/neurips2027/bin/activate
TOPNO_PHASE0_DATA_ROOT=/root/autodl-tmp/top_no_phase0 bash scripts/run_phase0_two_hour_queue.sh
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest tests.test_phase0_queue_config`

Run: `bash -n scripts/run_phase0_two_hour_queue.sh`

Commit: `git add configs/phase0_two_hour_qwen.json scripts/run_phase0_two_hour_queue.sh tests/test_phase0_queue_config.py README.md && git commit -m "feat: add two-hour phase0 GPU queue"`

### Task 6: Full Verification And Server Handoff

**Files:** Modify only Phase-0 files required to fix verification failures.

- [ ] **Step 1: Run all gates**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m compileall experiments`

Run: `for script in scripts/*.sh; do bash -n "$script"; done`

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests`

Run: `git diff --check`

- [ ] **Step 2: Prove the old runner remains blocked**

Run: `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python experiments/eval_prediction_sets.py --config configs/prediction_sets_qwen3b.json`

Expected: fails before model allocation and never starts legacy work.

- [ ] **Step 3: Review and push**

Run: `git status --short --branch && git log --oneline --decorate -8`

Expected: only design, plan, Phase-0 code/tests/config/script/docs changed; no result or secret exists.

Run: `git push origin main`

Expected: server can fast-forward `/root/neurips2027`, run preflight, and start the funded queue only after data/disk inspection.
