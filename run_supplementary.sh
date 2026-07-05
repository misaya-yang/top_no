#!/bin/bash
# Run all supplementary experiments on GPU server
set -e

export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Starting Supplementary Experiments ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

# ── Exp 3C+: Dynamic step size (no model needed, ~2 min) ──
echo "=== [1/4] Exp 3C+: Dynamic Step Size ==="
python exp3c_dynamic_step.py --output-dir ./results 2>&1
echo ""

# ── Exp 1B: Vocabulary ablation (GPU Monte Carlo, ~5 min) ──
echo "=== [2/4] Exp 1B: Vocabulary Size Ablation ==="
python exp1b_vocab_ablation.py --output-dir ./results 2>&1
echo ""

# ── Exp 5: Natural heteroscedastic channel (needs model, ~15 min) ──
echo "=== [3/4] Exp 5: Natural Alignment Channel ==="
python exp5_alignment_channel.py \
    --model Qwen/Qwen2.5-3B \
    --n-samples 3000 \
    --max-length 256 \
    --batch-size 16 \
    --output-dir ./results 2>&1
echo ""

# ── Exp 4C: Downstream evaluation (needs model, ~15 min) ──
echo "=== [4/4] Exp 4C: Downstream Task Evaluation ==="
python exp4c_downstream_eval.py \
    --model Qwen/Qwen2.5-3B \
    --n-gsm8k 50 \
    --n-creative 30 \
    --max-new-tokens 200 \
    --batch-size 8 \
    --output-dir ./results 2>&1
echo ""

echo "=== All supplementary experiments complete! ==="
echo "Results in ./results/"
ls -la results/exp* results/fig* 2>/dev/null
