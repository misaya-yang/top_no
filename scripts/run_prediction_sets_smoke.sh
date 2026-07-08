#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python}"
export PYTHONPATH="$REPO_ROOT/experiments:${PYTHONPATH:-}"

"$PYTHON_BIN" experiments/eval_prediction_sets.py \
  --config configs/prediction_sets_smoke.json
