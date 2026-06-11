#!/usr/bin/env bash
# =============================================================================
# train_runpod.sh — Full training run on RunPod RTX 5090
#
# Usage:
#   bash scripts/train_runpod.sh                  # full run
#   bash scripts/train_runpod.sh --smoke-test      # 5-step sanity check
#   bash scripts/train_runpod.sh --epochs 5        # override epochs
#   bash scripts/train_runpod.sh --lr 1e-4         # override LR
#
# All extra args are forwarded to src/train.py.
# =============================================================================

set -euo pipefail

# ── RTX 5090 environment ──────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# Reduces fragmentation on large VRAM GPUs; critical when batch_size >= 8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Avoid HuggingFace tokenizer parallelism warnings during data loading
export TOKENIZERS_PARALLELISM=false

# Force Python stdout/stderr to flush immediately (prevents silent logs when piped through tee)
export PYTHONUNBUFFERED=1

# Load secrets from .env if it exists (RunPod alternative to shell secrets)
if [ -f .env ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# ── Validation ────────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[error] HF_TOKEN is not set."
    echo "        Set it in your RunPod secrets or create a .env file."
    exit 1
fi

# ── Info banner ───────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Path-VQA Med-GaMMa — Training on RunPod             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  Config : config/config.yaml"
echo "  Extra  : $*"
echo ""

# ── Timestamps & logging ──────────────────────────────────────────────────────
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "  Log    : $LOG_FILE"
echo ""

# ── Run training (tee to log + screen) ───────────────────────────────────────
python -m src.train \
    --config config/config.yaml \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training finished. Log saved to: $LOG_FILE"
echo ""
