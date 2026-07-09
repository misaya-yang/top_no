# top_no

Research repository for the ICML 2027 draft:
**Frequency-Offset Margin Rules for Calibrated Language Model Decoding**.

Retired framing: earlier drafts used the title **Truncation Sampling as
Hypothesis Testing** and argued for an identified frequency-dependent noise
channel. That framing is now legacy. The active project treats token truncation
as prediction-set construction and asks a falsifiable question: does token
frequency carry information beyond the logit margin?

## Current Thesis

Token truncation in language-model decoding constructs a candidate prediction
set. Standard truncation rules can be read as margin rules over
`m_i = s_max - s_i`; the current method family studies frequency-offset scores:

```text
A(x, i) = m_i(x) - g(n_i)
S(x) = { i : A(x, i) <= q_hat }
```

where `q_hat` is a split-conformal quantile and `g` may be zero, a signed
inverse-frequency offset, a frequency-bucket/Mondrian offset, or a learned
offset. Conformal calibration supplies coverage for any score, so the paper
claim must come from improved coverage/size tradeoffs or better
frequency-bucket behavior at matched coverage.

## Current Status

This repository has been patched for the first reproducibility audit and the
first calibrated prediction-set pipeline:

- Decoding experiments now share `experiments/samplers.py`.
- `min_p`, `fixed_margin`, and `conformal_nu` are available in the shared
  sampler surface.
- `experiments/conformal.py` contains the split-conformal scoring helpers.
- `top_p` uses the standard nucleus rule and keeps the crossing token.
- Batch generation uses left padding, EOS-aware stopping, and generated-token-only
  metrics.
- Logit-space truncation runs on raw logits; temperature is applied after
  truncation for sampling.
- Invalid truncated distributions now fail instead of silently falling back to
  untruncated softmax.

Existing files in `results/` and the old final reports were produced before
these fixes. Treat them as legacy artifacts until the relevant experiments are
rerun under the calibrated protocol.

Protocol blocker: the current prediction-set runner is intentionally blocked
for paper-grade runs until the PR-2 conformal core and PR-3 calibrated gate
land. PR-1 now binds immutable frequency counts, deterministic document splits,
exact source text, and a recomputed cross-corpus near-duplicate receipt. The
legacy runner builds token counts from the loaded calibration/eval text pool and
uses a sequential calibration/eval split, so its outputs are smoke-test
artifacts only.

## Layout

- `experiments/`: runnable experiment scripts and shared experiment utilities.
- `experiments/samplers.py`: shared truncation and batch generation utilities.
- `tests/`: lightweight sampler correctness tests.
- `scripts/`: suite-level shell entrypoints.
- `results/`: generated figures and JSON result artifacts.
- `docs/paper/`: active claim stack, experiment mainline, and positioning notes.
- `docs/fable5/`: external research critiques and planning documents.
- `docs/reports/`: experiment writeups and summary reports.
- `requirements.txt`: Python runtime dependencies.
- `AGENTS.md`: working instructions for repository agents and contributors.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full model-backed experiments expect the relevant Hugging Face model cache to
exist locally. The suite scripts default to offline Hugging Face mode for the GPU
server environment.

Build a small pinned frequency table from an already frozen `freq` manifest and
matching JSONL without loading model weights:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 experiments/prepare_frequency_table.py \
  --document-jsonl /data/frozen_d_freq.jsonl \
  --source-manifest /data/freq_manifest.json \
  --model-id Qwen/Qwen2.5-7B \
  --revision <pinned-hugging-face-commit> \
  --output-dir /data/frequency_table
```

This is a single-process artifact builder. Multi-billion-token paper tables
still require a later sharded/resumable production pipeline and provenance
receipt.

## Run

Run the main experiment suite:

```bash
bash scripts/run_all.sh
```

Run supplementary experiments:

```bash
bash scripts/run_supplementary.sh
```

Run one experiment directly from the repository root:

```bash
PYTHONPATH="$PWD/experiments" python3 experiments/exp1_topk_bias.py --output-dir ./results
```

Run the local prediction-set smoke test with the verified conda Torch
environment. This is a legacy-protocol link test, not paper evidence:

```bash
bash scripts/run_prediction_sets_smoke.sh
```

The rented-GPU Qwen2.5-3B prediction-set experiment is blocked by default until
the split/count/gate repairs land:

```bash
PYTHON_BIN=python bash scripts/run_prediction_sets_qwen3b.sh
```

Plot prediction-set figures and run the decision gate:

```bash
PYTHON_BIN=python bash scripts/run_prediction_set_plots.sh
python experiments/check_prediction_set_gate.py --metrics results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json
```

Run the full GPU queue. This stops after the prediction-set gate if the core
signal is not strong enough for downstream generation experiments:

```bash
PYTHON_BIN=python bash scripts/run_icml2027_gpu_queue.sh
```

The prediction-set experiment writes:

- `prediction_set_metrics.json`
- `coverage_size_pareto.png`
- `prediction_set_coverage_efficiency.png`
- `prediction_set_bucket_coverage.png`
- `prediction_set_distribution_summary.png`

Downstream GPU scripts:

- `scripts/run_reasoning_self_consistency_qwen3b.sh`: GSM8K, MATH-500, and SVAMP pass@k/maj@k evaluation.
- `scripts/run_openended_quality_qwen3b.sh`: open-ended quality with self-BLEU, repetition, unique-token ratio, and perplexity.
- `scripts/run_controlled_channels_qwen3b.sh`: controlled perturbation evidence by target-token frequency bucket.

## Lightweight Verification

These checks verify the repository structure and syntax without rerunning the
expensive GPU experiments:

```bash
python3 -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover tests
```
