#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$REPO_ROOT/experiments:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PRED_DIR="results/prediction_sets_qwen3b_wikitext"
PRED_METRICS="$PRED_DIR/prediction_set_metrics.json"

echo "============================================================"
echo "ICML 2027 GPU queue started: $(date)"
echo "Python: $("$PYTHON_BIN" --version)"
echo "============================================================"

echo "[1/5] Prediction-set evaluation"
"$PYTHON_BIN" experiments/eval_prediction_sets.py \
  --config configs/prediction_sets_qwen3b.json

echo "[2/5] Prediction-set figures"
"$PYTHON_BIN" experiments/plot_prediction_sets.py \
  --metrics "$PRED_METRICS"

echo "[3/5] Decision gate"
"$PYTHON_BIN" experiments/check_prediction_set_gate.py \
  --metrics "$PRED_METRICS"

echo "[4/5] Reasoning self-consistency"
"$PYTHON_BIN" experiments/eval_reasoning_self_consistency.py \
  --config configs/reasoning_self_consistency_qwen3b.json

echo "[5/5] Open-ended quality"
"$PYTHON_BIN" experiments/eval_openended_quality.py \
  --config configs/openended_quality_qwen3b.json

echo "============================================================"
echo "ICML 2027 GPU queue complete: $(date)"
echo "============================================================"
