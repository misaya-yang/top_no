#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results

# Use local HF cache (no internet needed)
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "============================================================"
echo " Experiment Suite — $(date)"
echo " Paper: Truncation Sampling as Hypothesis Testing"
echo "============================================================"
echo ""
echo "[check] Python:  $(python3 --version)"
echo "[check] PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "[check] CUDA:    $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# ── Experiment 1: Top-K Bias + K* Scaling (~30 min) ──
echo "============================================================"
echo "▶ Exp 1: Top-K Bias + Zipf + Synthetic K* + Coverage + Rank Ablation"
echo "============================================================"
time python3 exp1_topk_bias.py \
    --model "Qwen/Qwen2.5-3B" \
    --n-samples 2000 \
    --max-length 256 \
    --batch-size 8
echo ""

# ── Experiment 2: Identifiability + V_eff + Two-Point (~70 min) ──
echo "============================================================"
echo "▶ Exp 2: Identifiability + n-Sweep + Full-KL + Two-Point + Corollary"
echo "============================================================"
time python3 exp2_identifiability.py \
    --teacher "Qwen/Qwen2.5-7B" \
    --student "Qwen/Qwen2.5-3B" \
    --n-samples 2000 \
    --max-length 256 \
    --batch-size 4
echo ""

# ── Experiment 3: Lyapunov + Bimodality + Online + Falsification (~70 min) ──
echo "============================================================"
echo "▶ Exp 3: Lyapunov + Bimodality + Online Margin + Burst-Calm + Long Seq"
echo "============================================================"
time python3 exp3_lyapunov.py \
    --model "Qwen/Qwen2.5-3B" \
    --n-samples 50 \
    --min-tokens 200 \
    --max-tokens 300 \
    --epsilon 1e-3
echo ""

echo "============================================================"
echo "✓ All experiments complete — $(date)"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh results/*.png results/*.json 2>/dev/null || echo "(no outputs found)"
