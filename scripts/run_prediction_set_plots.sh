#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
METRICS="${METRICS:-results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json}"
export PYTHONPATH="$REPO_ROOT/experiments:${PYTHONPATH:-}"

"$PYTHON_BIN" experiments/plot_prediction_sets.py \
  --metrics "$METRICS"
