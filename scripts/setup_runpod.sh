#!/usr/bin/env bash
# =============================================================================
# setup_runpod.sh — One-shot environment setup for RunPod RTX 5090
#
# Run once after spinning up a new pod:
#   bash scripts/setup_runpod.sh
#
# Requires:  HF_TOKEN set as RunPod secret or exported in the shell.
# =============================================================================

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Path-VQA Med-GaMMa — RunPod Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  CUDA  : $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | cut -d',' -f1)"
echo "  Python: $(python --version)"
echo ""

# ── 1. Flash Attention 2 (must be first, heavy C++ compile) ──────────────────
echo "[1/4] Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation -q
echo "      Flash Attention 2 installed."
echo ""

# ── 2. Project dependencies ───────────────────────────────────────────────────
echo "[2/4] Installing project requirements..."
pip install -r requirements.txt -q
echo "      Dependencies installed."
echo ""

# ── 3. Pre-cache Med-GaMMa base weights ──────────────────────────────────────
echo "[3/4] Pre-fetching Med-GaMMa 4B base model weights..."
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
token = os.getenv("HF_TOKEN")
if not token:
    print("  [warn] HF_TOKEN not set — download may fail for gated model")
snapshot_download(
    "google/medgemma-4b-it",
    token=token,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
)
print("  Base model cached.")
PYEOF
echo ""

# ── 4. Create output directories ──────────────────────────────────────────────
echo "[4/4] Creating workspace directories..."
mkdir -p outputs wandb metrics logs
echo "      Directories ready."
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup complete!  Start training with:                       ║"
echo "║                                                              ║"
echo "║    bash scripts/train_runpod.sh                              ║"
echo "║    bash scripts/train_runpod.sh --smoke-test  (quick check)  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
