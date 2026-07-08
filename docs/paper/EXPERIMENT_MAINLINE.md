# Experiment Mainline

## Stage 1: Prediction-Set Gate

Run:

```bash
PYTHON_BIN=python bash scripts/run_prediction_sets_qwen3b.sh
PYTHON_BIN=python bash scripts/run_prediction_set_plots.sh
python experiments/check_prediction_set_gate.py --metrics results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json
```

Required signal before spending more GPU:

```text
At fixed target coverage, conformal-nu should reduce average support size against at least one strong baseline, or at matched support size it should improve low-frequency coverage without degrading overall coverage.
```

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

## Stage 4: Controlled Channels

Run:

```bash
PYTHON_BIN=python bash scripts/run_controlled_channels_qwen3b.sh
```

Report:

- hidden-noise target-logit sensitivity by target-token frequency.
- dropout-ensemble target-logit variance by target-token frequency.

Quantization residuals are a planned extension and require an explicit dependency decision.

## One-Shot GPU Queue

Run:

```bash
PYTHON_BIN=python bash scripts/run_icml2027_gpu_queue.sh
```

The queue stops after Stage 1 if the prediction-set gate fails.
