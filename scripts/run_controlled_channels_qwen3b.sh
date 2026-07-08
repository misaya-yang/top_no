#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$REPO_ROOT/experiments:${PYTHONPATH:-}"

"$PYTHON_BIN" experiments/exp5b_controlled_channels.py \
  --config configs/controlled_channels_qwen3b.json
