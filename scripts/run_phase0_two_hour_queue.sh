#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${TOPNO_PHASE0_DATA_ROOT:-/root/autodl-tmp/top_no_phase0}"
OUTPUT_ROOT="${PHASE0_OUTPUT_ROOT:-$REPO_ROOT/results/phase0_two_hour}"
CONFIG="${PHASE0_CONFIG:-$REPO_ROOT/configs/phase0_two_hour_qwen.json}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/experiments:${PYTHONPATH:-}"
export PYTHONHASHSEED=1729
export CUBLAS_WORKSPACE_CONFIG=:4096:8

read_matrix_value() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$CONFIG" "$1"
}

QUEUE_WALL_SECONDS="$(read_matrix_value queue_wall_seconds)"
CELL_WALL_SECONDS="$(read_matrix_value cell_wall_seconds)"
SUMMARY_RESERVE_SECONDS="$(read_matrix_value summary_reserve_seconds)"
MINIMUM_START_SECONDS="$(read_matrix_value minimum_start_seconds)"
COMMIT="$(git rev-parse HEAD)"
CELLS=(3b_web 3b_math 7b_web 7b_math)

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[phase0-queue] current commit is not pinned" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/.queue_status.tsv"
: > "$STATUS_TSV"

echo "[phase0-queue] repository=$REPO_ROOT"
echo "[phase0-queue] commit=$COMMIT"
echo "[phase0-queue] data_root=$DATA_ROOT"
echo "[phase0-queue] output_root=$OUTPUT_ROOT"

echo "[phase0-queue] validating all artifacts and cached revisions before funded time"
for cell in "${CELLS[@]}"; do
  echo "[phase0-queue] preflight $cell"
  if ! "$PYTHON_BIN" experiments/phase0_reliability.py \
    --config "$CONFIG" \
    --cell "$cell" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --preflight-only; then
    echo "[phase0-queue] preflight failed for $cell; no GPU queue started" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$((START_EPOCH + QUEUE_WALL_SECONDS))"
STOP_AFTER_FAILURE=0

for cell in "${CELLS[@]}"; do
  if (( STOP_AFTER_FAILURE != 0 )); then
    printf '%s\t%s\t%s\n' "$cell" "NOT_STARTED" "earlier_cell_failed" >> "$STATUS_TSV"
    continue
  fi
  NOW_EPOCH="$(date +%s)"
  REMAINING="$((DEADLINE_EPOCH - NOW_EPOCH))"
  USABLE="$((REMAINING - SUMMARY_RESERVE_SECONDS))"
  if (( USABLE < MINIMUM_START_SECONDS )); then
    printf '%s\t%s\t%s\n' "$cell" "NOT_STARTED" "global_deadline" >> "$STATUS_TSV"
    continue
  fi
  WALL_SECONDS="$CELL_WALL_SECONDS"
  if (( WALL_SECONDS > USABLE )); then
    WALL_SECONDS="$USABLE"
  fi
  mkdir -p "$OUTPUT_ROOT/$cell"
  echo "[phase0-queue] start $cell wall_seconds=$WALL_SECONDS remaining=$REMAINING"
  set +e
  "$PYTHON_BIN" experiments/phase0_reliability.py \
    --config "$CONFIG" \
    --cell "$cell" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --wall-seconds "$WALL_SECONDS" \
    --created-by-commit "$COMMIT" \
    2>&1 | tee "$OUTPUT_ROOT/$cell/run.log"
  CELL_EXIT="${PIPESTATUS[0]}"
  set -e
  if (( CELL_EXIT != 0 )); then
    printf '%s\t%s\t%s\n' "$cell" "FAILED" "exit_$CELL_EXIT" >> "$STATUS_TSV"
    STOP_AFTER_FAILURE=1
    continue
  fi
  CELL_STATUS="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["completion_status"])' "$OUTPUT_ROOT/$cell/phase0_summary.json")"
  printf '%s\t%s\t%s\n' "$cell" "$CELL_STATUS" "ok" >> "$STATUS_TSV"
done

"$PYTHON_BIN" experiments/summarize_phase0_queue.py --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" - "$STATUS_TSV" "$OUTPUT_ROOT/queue_status.json" "$START_EPOCH" "$DEADLINE_EPOCH" <<'PY'
import json
import os
import sys
import time

rows = []
with open(sys.argv[1]) as handle:
    for line in handle:
        cell, status, reason = line.rstrip("\n").split("\t")
        rows.append({"cell": cell, "status": status, "reason": reason})
payload = {
    "schema_version": "icml2027-phase0-queue-status-v1",
    "evidence_grade": "E-pilot",
    "paper_citable": False,
    "started_epoch": int(sys.argv[3]),
    "deadline_epoch": int(sys.argv[4]),
    "finished_epoch": int(time.time()),
    "cells": rows,
}
temporary = sys.argv[2] + ".tmp"
with open(temporary, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, sys.argv[2])
PY
rm -f "$STATUS_TSV"

echo "[phase0-queue] complete"
echo "[phase0-queue] decision=$OUTPUT_ROOT/decision_memo.json"
echo "[phase0-queue] status=$OUTPUT_ROOT/queue_status.json"
