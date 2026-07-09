# Experiment Mainline

## Stage 0: Margin x Frequency Diagnostic

Before spending GPU on downstream decoding claims, estimate:

```text
h(m, n) = P(Y = i | margin m_i = m, frequency n_i = n)
```

using held-out next-token positions. The diagnostic should answer whether token
frequency changes true-token probability at fixed logit margin and should also
determine the empirical sign of any frequency offset. If this surface is flat in
`n` after conditioning on `m`, the project becomes a calibrated audit /
negative-result paper rather than a new decoding-method paper.

Phase 0 must also test whether `h_m(m)` is monotone, quantify held-out
predictive gain from adding frequency, and distinguish a general
two-dimensional learned-`h` surface from an additive horizontal shift
`m-g(n)`. A null frequency interaction is not enough to call C-margin
frontier-optimal without the monotonicity and power checks.

## Stage 1: Prediction-Set Gate

Target entrypoints after PR-1 through PR-3 are complete (they are intentionally
blocked today):

```bash
PYTHON_BIN=python bash scripts/run_prediction_sets_qwen3b.sh
PYTHON_BIN=python bash scripts/run_prediction_set_plots.sh
python experiments/check_prediction_set_gate.py --metrics results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json
```

Required signal before spending more GPU:

```text
At fixed target coverage, a frequency-offset rule should improve the coverage/size frontier against calibrated margin-only baselines, or reduce frequency-bucket coverage gaps at matched marginal coverage and mean set size.
```

Coverage attainment alone is not evidence because split conformal gives
coverage for any score. The gate should compare calibrated methods against
calibrated methods; uncalibrated top-p/min-p/fixed-margin numbers are diagnostic
context only.

## Stage 2: Reasoning Self-Consistency

Run:

```bash
PYTHON_BIN=python bash scripts/run_reasoning_self_consistency_qwen3b.sh
```

Report:

- `acc@1`
- `pass@4`, `pass@8`, `pass@16`
- `maj@4`, `maj@8`, `maj@16`
- invalid answer rate
- unique answer count
- answer entropy

Datasets:

- GSM8K
- MATH-500
- SVAMP

## Stage 3: Open-Ended Quality

Run:

```bash
PYTHON_BIN=python bash scripts/run_openended_quality_qwen3b.sh
```

Report:

- Distinct-n as surface diversity only.
- self-BLEU.
- repetition rate.
- evaluator LM perplexity.
- length-normalized unique token ratio.

## One-Shot GPU Queue

This is also a target post-PR-3 entrypoint. Controlled-channel experiments are
archived and intentionally absent from the active paper queue.

Run:

```bash
PYTHON_BIN=python bash scripts/run_icml2027_gpu_queue.sh
```

The queue stops after Stage 1 if the prediction-set gate fails.
