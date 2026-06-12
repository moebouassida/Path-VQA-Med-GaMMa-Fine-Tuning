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

# Load .env if present (same as train_runpod.sh)
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Path-VQA Med-GaMMa — RunPod Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  CUDA  : $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | cut -d',' -f1)"
echo "  Python: $(python --version)"
echo ""

# ── 1. Project dependencies ───────────────────────────────────────────────────
echo "[1/3] Installing project requirements..."

# Detect the CUDA driver version and pick a matching torch index.
# The driver caps which CUDA runtime version PyTorch can use — installing a
# torch built for a newer CUDA than the driver supports breaks CUDA visibility.
DRIVER_CUDA=$(python -c "
import subprocess, re
out = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
m = re.search(r'CUDA Version: (\d+\.\d+)', out)
if m:
    major, minor = m.group(1).split('.')
    print(f'cu{major}{minor.zfill(2)}')
else:
    print('cu124')
" 2>/dev/null || echo "cu124")
echo "      Driver CUDA: $DRIVER_CUDA — installing matching torch..."

# Install torch first from the driver-compatible index, then everything else.
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${DRIVER_CUDA} \
    --timeout 120 -q

# Install remaining project deps (torch already satisfied above)
pip install --timeout 120 -r requirements.txt -q
echo "      Dependencies installed."
echo ""

# ── 2. Pre-cache Med-GaMMa base weights ──────────────────────────────────────
echo "[2/3] Pre-fetching Med-GaMMa 4B base model weights..."
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

# ── 3. Create output directories ──────────────────────────────────────────────
echo "[3/3] Creating workspace directories..."
mkdir -p outputs wandb metrics logs
echo "      Directories ready."
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup complete!  Start training with:                       ║"
echo "║                                                              ║"
echo "║    bash scripts/train_runpod.sh --smoke-test  (quick check)  ║"
echo "║    bash scripts/train_runpod.sh --epochs 1    (full run)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
