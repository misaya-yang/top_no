# Calibrated Support Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the project from a ν-sampling heuristic into a calibrated token-level logit prediction-set framework, then run experiments that directly test coverage, efficiency, and downstream sampling value.

**Architecture:** Keep `experiments/samplers.py` as the shared decoding surface, add conformal scoring/calibration helpers, and introduce a separate prediction-set evaluation pipeline before rerunning generation metrics. Treat existing `results/` as legacy until regenerated with the fixed sampler/generation path.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Datasets, NumPy, Matplotlib, existing `experiments/data_utils.py`, and local/GPU-server Hugging Face caches.

---

## Source Review

Raw GPT-5.5 Pro review saved at:

- `docs/reviews/2026-07-08-gpt55pro-research-review.md`

## Current Repo Alignment

### Already Fixed

- Shared sampler/generation core exists in `experiments/samplers.py`.
- Standard top-p now keeps the crossing token.
- Sampler registry now includes `top_k`, `min_p`, and `fixed_margin`.
- Conformal helpers now exist in `experiments/conformal.py`.
- Prediction-set evaluator now exists in `experiments/eval_prediction_sets.py`.
- Prediction-set plotting and gate checking now exist in `experiments/plot_prediction_sets.py`
  and `experiments/check_prediction_set_gate.py`.
- Smoke and Qwen3B configs now exist under `configs/`.
- Runnable entrypoints now exist in `scripts/run_prediction_sets_smoke.sh` and
  `scripts/run_prediction_sets_qwen3b.sh`.
- Downstream GPU entrypoints now exist for reasoning self-consistency,
  open-ended quality, controlled channels, and the full gated queue.
- Paper-facing V2 docs now exist under `docs/paper/`.
- Downstream decoding paths use raw-logit truncation before temperature sampling.
- Batch generation uses left padding, explicit `position_ids`, EOS-aware stopping, and generated-token-only metrics.
- Silent fallback to untruncated softmax is removed.
- Tests cover top-p crossing, ν margin direction, left-padded generation, `position_ids`, and EOS slicing.
- `README.md`, `AGENTS.md`, and `docs/reports/REPRODUCIBILITY_AUDIT.md` mark current `results/` as legacy.

### Not Yet Complete

- `samplers.py` does not yet include typical or eta/epsilon as first-class generation strategies.
- Qwen3B GPU prediction-set results have not been run yet; current committed smoke
  output is only a local link test.
- Reasoning, open-ended, and controlled-channel GPU scripts exist but have not
  produced paper-ready Qwen3B result artifacts yet.
- Quantization residual and bootstrap-model channel probes are still planned.
- Old reports remain useful as historical artifacts, not final claims.

## Core Claim

Use this as the working primary claim:

> ν-sampling turns truncation decoding into frequency-calibrated logit prediction sets, generalizing min-p and top-nσ from global margins to token-wise uncertainty margins with calibratable coverage.

Operational version:

> We recast truncation sampling as token-level prediction set construction over next-token logits. Min-p is a fixed logit-margin rule; top-nσ is a context-global logit-margin rule; ν introduces frequency-indexed token-wise margins. With split calibration, ν-conformal provides target next-token coverage while improving support-set efficiency.

Avoid these claims until new evidence exists:

- "We identified the real LLM noise channel."
- "ν-sampling beats greedy/top-p on GSM8K single-sample accuracy."
- "Current legacy JSON results are paper-ready."
- "Mathboost rescues low-frequency math tokens by widening their ν margin." The current formula does the opposite for boosted counts: it contracts the uncertainty margin.

## Method Target

Define the ν nonconformity score:

```text
A_kappa(x, i) = s_max(x) - s_i(x) - kappa / sqrt(n_i + alpha)
```

Calibrate on held-out true next tokens:

```text
q_hat = Quantile_{1 - delta}(A_kappa(x_t, y_t))
```

Retain at test time:

```text
S_nu(x) = { i : A_kappa(x, i) <= q_hat }
```

Decode from the retained set:

```text
p_tilde_i = exp(s_i / T) 1[i in S_nu(x)] / sum_j exp(s_j / T) 1[j in S_nu(x)]
```

Special-case positioning:

- `min-p = alpha_p` is equivalent to fixed logit margin `m = -log(alpha_p)`.
- `ν(kappa=0, q=3)` is approximately `min-p=0.05`.
- `top-nsigma` is a context-global margin; ν is token-specific.
- `ν-recall` with the current formula controls false exclusions by widening low-frequency uncertainty margins.
- A future dual-channel ν can add a separate reliability/prior penalty instead of overloading frequency count.

## Task 1: Finish Sampler Registry

**Files:**
- Modify: `experiments/samplers.py`
- Modify: `tests/test_samplers.py`
- Optional docs: `docs/reports/REPRODUCIBILITY_AUDIT.md`

- [x] **Step 1: Add first-class strategies**

Add strategies to `apply_truncation`:

```python
elif strategy == "top_k":
    k = int(kwargs.get("k", 50))
    top_values, _ = logits.topk(max(k, 1), dim=-1)
    threshold = top_values[..., -1:].expand_as(logits)
    keep = logits >= threshold

elif strategy == "min_p":
    p_min = kwargs.get("p_min", 0.05)
    probs = F.softmax(logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    keep = probs >= p_min * p_max

elif strategy == "fixed_margin":
    margin = kwargs.get("margin", 3.0)
    s_max = logits.max(dim=-1, keepdim=True).values
    keep = (s_max - logits) <= margin
```

- [x] **Step 2: Add equivalence tests**

Add tests:

```python
def test_fixed_margin_matches_min_p_threshold(self):
    logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
    min_p = apply_truncation(logits, "min_p", p_min=math.exp(-3.0))
    fixed = apply_truncation(logits, "fixed_margin", margin=3.0)
    self.assertEqual(torch.isfinite(min_p).tolist(), torch.isfinite(fixed).tolist())

def test_nu_kappa_zero_matches_fixed_margin(self):
    logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
    freqs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    nu = apply_truncation(logits, "nu", token_freq_table=freqs, kappa=0.0, m0=3.0)
    fixed = apply_truncation(logits, "fixed_margin", margin=3.0)
    self.assertEqual(torch.isfinite(nu).tolist(), torch.isfinite(fixed).tolist())
```

- [x] **Step 3: Run sampler tests**

Run:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests
```

Expected: all tests pass.

## Task 2: Add Conformal Calibration Helpers

**Files:**
- Create: `experiments/conformal.py`
- Create: `tests/test_conformal.py`

- [x] **Step 1: Implement nonconformity and quantile helpers**

Create `experiments/conformal.py`:

```python
from __future__ import annotations

import math
import torch


def nu_nonconformity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    s_max = logits.max(dim=-1).values
    target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    freqs = token_freq_table.to(logits.device).float()[target_ids]
    return s_max - target_logits - kappa / torch.sqrt(freqs + alpha)


def conformal_quantile(scores: torch.Tensor, delta: float) -> float:
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if scores.numel() == 0:
        raise ValueError("scores must be non-empty")
    n = scores.numel()
    rank = min(math.ceil((n + 1) * (1 - delta)), n)
    sorted_scores = torch.sort(scores.flatten()).values
    return float(sorted_scores[rank - 1].item())


def conformal_nu_keep_mask(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    q_hat: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    s_max = logits.max(dim=-1, keepdim=True).values
    freqs = token_freq_table.to(logits.device).float().unsqueeze(0).expand_as(logits)
    scores = s_max - logits - kappa / torch.sqrt(freqs + alpha)
    return scores <= q_hat
```

- [x] **Step 2: Add deterministic tests**

Create `tests/test_conformal.py`:

```python
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    from conformal import conformal_quantile, nu_nonconformity, conformal_nu_keep_mask
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ConformalTests(unittest.TestCase):
    def test_quantile_uses_conformal_rank(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(conformal_quantile(scores, delta=0.25), 4.0)

    def test_nu_nonconformity_uses_target_frequency(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]])
        target_ids = torch.tensor([1])
        freqs = torch.tensor([100.0, 0.0, 10.0])
        score = nu_nonconformity(logits, target_ids, freqs, kappa=1.0)
        self.assertAlmostEqual(float(score[0]), 2.0)

    def test_conformal_mask_keeps_tokens_below_threshold(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]])
        freqs = torch.tensor([100.0, 0.0, 10.0])
        keep = conformal_nu_keep_mask(logits, freqs, kappa=1.0, q_hat=2.0)
        self.assertEqual(keep.tolist(), [[True, True, False]])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 3: Run tests**

Run:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests
```

Expected: all tests pass.

## Task 3: Build Prediction-Set Evaluation

**Files:**
- Create: `experiments/eval_prediction_sets.py`
- Create: `configs/prediction_sets_smoke.json`
- Create: `configs/prediction_sets_qwen3b.json`
- Modify: `README.md`

- [x] **Step 1: Add evaluator outputs**

The evaluator must write JSON with these fields:

```json
{
  "model": "Qwen/Qwen2.5-3B",
  "dataset": "wikitext",
  "n_positions": 1000,
  "methods": {
    "top_p_0.95": {
      "coverage": 0.0,
      "avg_set_size": 0.0,
      "median_set_size": 0.0,
      "avg_retained_mass": 0.0,
      "low_freq_bucket_coverage": {},
      "config": {}
    }
  }
}
```

- [x] **Step 2: Implement minimum methods first**

Initial methods:

```text
top_k_50
top_p_0.95
min_p_0.05
fixed_margin_3
top_nsigma_2
nu_k10_q3
conformal_nu_k10_delta05
```

- [x] **Step 3: Smoke run with cached small model**

Use `gpt2` for local smoke because it is cached on this Mac:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python experiments/eval_prediction_sets.py \
  --model gpt2 \
  --dataset wikitext \
  --n-calibration 64 \
  --n-eval 64 \
  --batch-size 2 \
  --output-dir ./results/smoke_prediction_sets
```

Expected:

```text
results/smoke_prediction_sets/prediction_set_metrics.json
```

- [ ] **Step 4: GPU run after renting card**

Run:

```bash
python experiments/eval_prediction_sets.py \
  --model Qwen/Qwen2.5-3B \
  --dataset wikitext \
  --n-calibration 5000 \
  --n-eval 20000 \
  --batch-size 8 \
  --output-dir ./results/prediction_sets_qwen3b_wikitext
```

Expected: coverage-size Pareto tables and figures.

## Task 4: Define Main Figures

**Files:**
- Create: `experiments/plot_prediction_sets.py`
- Create: `docs/reports/FIGURE_PLAN.md`

- [x] **Step 1: Produce coverage-efficiency figure**

Required plot:

```text
x-axis: average set size
y-axis: true-token coverage
series: top-k, top-p, min-p, top-nsigma, ν, conformal-ν
```

- [x] **Step 2: Produce frequency-bucket coverage figure**

Required buckets:

```text
count=0
count=1-2
count=3-10
count=11-100
count>100
```

- [x] **Step 3: Produce retained-mass/set-size distribution figure**

Required panels:

```text
retained probability mass distribution
support size histogram
support size by token-frequency bucket
```

## Task 5: Reasoning Self-Consistency Evaluation

**Files:**
- Create: `experiments/eval_reasoning_self_consistency.py`
- Create: `configs/reasoning_self_consistency_qwen3b.json`

- [x] **Step 1: Replace single-sample claim**

Report:

```text
acc@1
pass@4, pass@8, pass@16
maj@4, maj@8, maj@16
invalid answer rate
unique answer count
answer entropy
```

- [x] **Step 2: Use real datasets only**

Required datasets:

```text
GSM8K
MATH-500
SVAMP
```

If a dataset cannot load, the script must fail with a clear error instead of silently using synthetic fallback for paper runs.

- [ ] **Step 3: GPU run**

Run:

```bash
python experiments/eval_reasoning_self_consistency.py \
  --model Qwen/Qwen2.5-3B \
  --datasets gsm8k math500 svamp \
  --samples-per-question 16 \
  --n-questions 500 \
  --batch-size 8 \
  --output-dir ./results/reasoning_self_consistency_qwen3b
```

Expected: JSON with pass@k/maj@k and invalid-rate tables.

## Task 6: Open-Ended Quality Evaluation

**Files:**
- Create: `experiments/eval_openended_quality.py`
- Create: `configs/openended_quality_qwen3b.json`

- [x] **Step 1: Keep Distinct but demote it**

Report Distinct-n as surface diversity only.

- [x] **Step 2: Add quality metrics**

At minimum:

```text
self-BLEU
repetition rate
external LM perplexity
length-normalized unique token ratio
```

Add MAUVE and LLM-as-judge only after dependencies and evaluator prompts are stable.

- [ ] **Step 3: GPU run**

Run:

```bash
python experiments/eval_openended_quality.py \
  --model Qwen/Qwen2.5-3B \
  --datasets writingprompts alpacaeval_creative \
  --n-prompts 300 \
  --batch-size 8 \
  --max-new-tokens 256 \
  --output-dir ./results/openended_quality_qwen3b
```

## Task 7: Controlled Channel Evidence

**Files:**
- Create: `experiments/exp5b_controlled_channels.py`
- Create: `docs/reports/CHANNEL_EVIDENCE_PLAN.md`

- [x] **Step 1: Hidden-state Gaussian channel**

Measure:

```text
Var((W_i(h + xi)) - W_i h) by token frequency bucket
```

- [ ] **Step 2: Quantization residual channel**

Measure:

```text
logit residual fp16 vs int8/int4 by token frequency bucket
```

- [x] **Step 3: Dropout/activation perturbation ensemble**

Measure:

```text
same-prefix logit variance across perturbation ensemble
```

- [x] **Step 4: Report honestly**

Accepted claim only if at least two controlled channels show stable frequency-dependent sensitivity:

```text
Frequency is a useful proxy for token-wise logit sensitivity across controlled channels.
```

Do not claim:

```text
The real LLM noise channel is identified.
```

## Task 8: Rewrite Paper-Facing Narrative

**Files:**
- Create: `docs/paper/CLAIM_STACK.md`
- Create: `docs/paper/RELATED_WORK_POSITIONING.md`
- Create: `docs/paper/EXPERIMENT_MAINLINE.md`

- [x] **Step 1: Claim stack**

Use three contributions:

```text
1. Truncation decoding as token-level support testing / prediction-set construction.
2. ν nonconformity score: frequency-calibrated token-wise logit margin.
3. Calibrated and empirical validation: finite-sample token coverage plus downstream sampling value.
```

- [x] **Step 2: Related-work positioning**

Required framing:

```text
top-k: cardinality constraint
top-p: cumulative mass constraint
typical: local information-rate criterion
Mirostat: sequence-level perplexity control
eta/desmoothing: probability-space support recovery
min-p: fixed logit-margin special case
top-nsigma: context-global logit-margin special case
ν: token-wise frequency-calibrated logit-margin score
```

- [x] **Step 3: Replace old title**

Preferred title:

```text
ν-Sampling: Frequency-Calibrated Logit Prediction Sets for Language Model Decoding
```

Backup title:

```text
Token-Level Truncation as Calibrated Support Testing for Language Model Decoding
```

## Execution Order

1. Task 1: finish sampler registry.
2. Task 2: conformal helpers and tests.
3. Task 3: prediction-set evaluator with gpt2 smoke, then Qwen3B GPU run.
4. Task 4: main figures.
5. Task 5: reasoning self-consistency.
6. Task 6: open-ended quality.
7. Task 7: controlled channel evidence.
8. Task 8: paper-facing narrative rewrite.

## Decision Gate Before Expensive Runs

Do not spend GPU on generation-quality experiments until Task 3 produces at least one of these signals:

```text
At fixed 95% true-token coverage, conformal-ν has lower average support size than min-p/top-p/top-nsigma.
```

or

```text
At matched support size, conformal-ν improves low-frequency bucket coverage without degrading overall coverage.
```

If neither holds, revise the score before running downstream generation.

## Verification Commands

Local:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests
python3 -m compileall experiments tests
for script in scripts/*.sh; do bash -n "$script"; done
git diff --check
```

GPU server after implementation:

```bash
python experiments/eval_prediction_sets.py --model Qwen/Qwen2.5-3B --dataset wikitext --n-calibration 5000 --n-eval 20000 --batch-size 8 --output-dir ./results/prediction_sets_qwen3b_wikitext
```
