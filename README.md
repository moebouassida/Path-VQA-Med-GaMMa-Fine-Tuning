# 🔬 Path-VQA Med-GaMMa Fine-Tuning

**Fine-tune Med-GaMMa 4B on PathVQA Enhanced for pathology Visual Question Answering.**
Answers yes/no and open-ended clinical questions from H&E and other pathology images using DoRA fine-tuning on [moebouassida/path-vqa-enhanced](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced).

[![CI](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-medgemma--4b--path--vqa-yellow)](https://huggingface.co/moebouassida/medgemma-4b-path-vqa)
[![Dataset on HF](https://img.shields.io/badge/🤗%20Dataset-path--vqa--enhanced-yellow)](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced)

---

## Architecture

```
PathVQA Enhanced (HF Hub)
        │
        ▼
data_processing.py  ← conversation format + label masking
        │
        ▼
train.py  ← DoRA + RSLoRA fine-tuning (4-bit NF4, Flash Attention 2)
        │
        ▼
outputs/final/
        ├── src/main.py        ← FastAPI production API
        ├── src/inference.py   ← CLI inference
        ├── src/evaluate.py    ← evaluation + quality gates
        └── hf_spaces_app.py   ← Gradio demo (HF Spaces)
```

| Component | Details |
|-----------|---------|
| Base model | Med-GaMMa 4B (`google/medgemma-4b-it`) |
| Fine-tuning | DoRA + RSLoRA · r=16, α=32 · HF Trainer |
| Quantization | 4-bit NF4 + double quant · bfloat16 compute |
| Attention | Flash Attention 2 (~2× speedup) |
| Dataset | PathVQA Enhanced — ~32K LLM-enriched clinical QA pairs |
| Tracking | W&B (metrics + model artifacts) |
| Serving | FastAPI + Gradio |

---

## Setup

```bash
git clone https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning.git
cd Path-VQA-Med-GaMMa-Fine-Tuning
pip install -r requirements.txt

# Flash Attention 2 (optional but recommended — needs CUDA headers)
pip install flash-attn --no-build-isolation
```

**Requirements:** Python 3.10+, CUDA GPU (RTX 5090 recommended), HuggingFace token for Med-GaMMa access.

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...    # for experiment tracking
```

---

## Training on RunPod (RTX 5090)

The recommended training path is RunPod Community Cloud with a single RTX 5090 (32 GB VRAM).

### 1. Create a pod on runpod.io

| Setting | Value |
|---------|-------|
| GPU | RTX 5090 (32 GB GDDR7) |
| Template | RunPod PyTorch 2.x (CUDA 12.1+) |
| Container disk | 20 GB |
| Volume disk | 50 GB |
| Secrets | `HF_TOKEN`, `WANDB_API_KEY` |

### 2. Inside the pod terminal

```bash
git clone https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning
cd Path-VQA-Med-GaMMa-Fine-Tuning

# Install flash-attn + deps + pre-cache base model weights (~10 min)
bash scripts/setup_runpod.sh

# Quick sanity check (5 steps, ~2 min)
bash scripts/train_runpod.sh --smoke-test

# Full training run (~2–3 hours, ~$4–5 on Community Cloud)
bash scripts/train_runpod.sh
```

> **Tip:** Wrap in `tmux` so training keeps going after you disconnect:
> ```bash
> tmux new -s train
> bash scripts/train_runpod.sh
> # Ctrl+B then D to detach
> ```

### 3. Override hyperparameters

```bash
bash scripts/train_runpod.sh --epochs 5 --lr 1e-4
```

All key settings live in [`config/config.yaml`](config/config.yaml).

---

## Training locally

```bash
make smoke-test          # 5-step pipeline check
make train               # full run
```

Or directly:

```bash
python -m src.train --smoke-test
python -m src.train --config config/config.yaml
python -m src.train --epochs 5 --lr 1e-4
```

---

## Experiment Tracking (W&B)

Every training run logs to [Weights & Biases](https://wandb.ai):

- Train / eval loss curves per step
- Learning rate schedule
- GPU VRAM utilisation
- All hyperparameters from config

**Model versioning via W&B Artifacts** — after each completed run the adapter weights are saved as a versioned artifact (`v1`, `v2`, `v3` …):

```python
# Download a specific version for inference
import wandb
api = wandb.Api()
artifact = api.artifact("your-username/path-vqa-medgemma/medgemma-4b-path-vqa:v2")
artifact.download("outputs/v2")
```

Set `wandb_entity` in `config/config.yaml` to your W&B username or org.

---

## Evaluation

```bash
make evaluate
# or
python -m src.evaluate --model outputs/final --config config/config.yaml
```

Quality gates (configurable in `config.yaml`):

| Metric | Threshold |
|--------|-----------|
| Yes/No accuracy (exact match) | ≥ 0.55 |
| Open-ended BLEU-4 | ≥ 0.20 |
| Open-ended token F1 | reported (no gate) |

Exit code `0` = all gates passed, `1` = failed.

---

## Inference

**CLI:**
```bash
python -m src.inference \
  --model outputs/final \
  --image path/to/slide.jpg \
  --question "Is there evidence of malignancy?"

# With sampling
python -m src.inference --model outputs/final --image slide.jpg \
  --question "Describe the cellular architecture." \
  --sample --temperature 0.4
```

**Gradio demo:**
```bash
make demo
# → http://localhost:7860
```

**FastAPI server:**
```bash
make serve
# → http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + GPU stats |
| `GET` | `/info` | Model metadata + disclaimer |
| `POST` | `/predict` | Image URL + question → answer |
| `POST` | `/predict/upload` | Image file upload + question → answer |
| `GET` | `/docs` | Swagger UI |

```bash
# Image URL
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://...", "question": "Is this benign?"}'

# File upload
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@slide.jpg" \
  -F "question=What type of tissue is shown?"
```

---

## Docker (GPU)

```bash
docker build -f Docker/Dockerfile -t pathvqa-medgemma .
docker run --gpus all -p 8000:8000 \
  -v ./outputs:/app/outputs:ro \
  -e HF_TOKEN=$HF_TOKEN \
  pathvqa-medgemma
```

---

## Project Structure

```
Path-VQA-Med-GaMMa-Fine-Tuning/
├── src/
│   ├── train.py           # DoRA fine-tuning + W&B artifact logging
│   ├── inference.py       # Model loading + single/batch inference
│   ├── data_processing.py # Dataset loader + conversation format
│   ├── evaluate.py        # Evaluation + quality gates
│   ├── metrics.py         # Exact match, token F1, BLEU
│   └── main.py            # FastAPI server
├── hf_spaces_app.py       # Gradio demo (HF Spaces entry point)
├── config/config.yaml     # All hyperparameters
├── scripts/
│   ├── setup_runpod.sh    # One-shot RunPod environment setup
│   └── train_runpod.sh    # Training launcher for RunPod
├── Makefile               # Common tasks
├── Docker/Dockerfile      # GPU Docker image
├── tests/                 # Unit tests
└── .github/workflows/     # CI pipeline
```

---

## Citation

```bibtex
@software{pathvqa_medgemma2026,
  author = {Bouassida, Moez},
  title  = {Path-VQA Med-GaMMa Fine-Tuning},
  year   = {2026},
  url    = {https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

> ⚠️ **Research use only.** This is not a medical device. All model outputs must be reviewed by qualified healthcare professionals before any clinical use.
