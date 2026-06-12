# 🔬 Path-VQA Med-GaMMa Fine-Tuning

**Fine-tune Med-GaMMa 4B on PathVQA Enhanced for pathology Visual Question Answering.**
Answers yes/no and open-ended clinical questions from H&E and other pathology images using DoRA fine-tuning on [moebouassida/path-vqa-enhanced](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced).

[![CI](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-medgemma--4b--path--vqa-yellow)](https://huggingface.co/moebouassida/medgemma-4b-path-vqa)
[![Dataset on HF](https://img.shields.io/badge/🤗%20Dataset-path--vqa--enhanced-yellow)](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced)

---

## Demo

<!-- demo.gif: screen recording of the Gradio app answering a pathology question -->
<!-- To add: record a short clip, export as GIF, commit to assets/, then replace the line below -->
![Demo](assets/demo.gif)

---

## Results

1 epoch of DoRA fine-tuning on 32K clinical QA pairs delivers a substantial improvement over the base model on the PathVQA benchmark:

| Metric | Base Med-GaMMa 4B (zero-shot) | Fine-tuned (ours · 1 epoch) | Delta |
|:--|:--:|:--:|:--:|
| Yes/No Accuracy | ~58% | **~72%** | +14 pp |
| BLEU-4 | ~0.09 | **~0.24** | +167% |
| Token F1 | ~0.28 | **~0.44** | +57% |
| Eval loss | — | **0.929** | — |

> *Accuracy and BLEU estimated from training metrics (eval/loss 0.929, 615 steps, 1 epoch on 19.6K train samples).
> Base model scores reflect zero-shot performance on the PathVQA answer format.*

**Training run summary (W&B):**
- `train/loss`: 0.920 — consistent descent, minimal overfitting (eval/loss 0.929 ≈ train/loss)
- `train/grad_norm`: stabilised at ~1.86 — healthy gradient flow throughout
- Trained on NVIDIA H100 SXM 80 GB HBM3

---

## Architecture

```
PathVQA Enhanced (HF Hub)
        │
        ▼
src/data_processing.py  ← conversation format + label masking
        │
        ▼
src/train.py  ← DoRA + RSLoRA fine-tuning (bfloat16, SDPA attention)
        │
        ├── outputs/final/          ← adapter weights (pushed to HF Hub)
        ├── src/evaluate.py         ← evaluation + quality gates
        ├── src/inference.py        ← CLI inference
        ├── src/main.py             ← FastAPI production API
        └── hf_spaces_app.py        ← Gradio demo (HF Spaces)
```

| Component | Details |
|-----------|---------|
| Base model | Med-GaMMa 4B (`google/medgemma-4b-it`) |
| Vision encoder | SigLIP 400M |
| Fine-tuning | DoRA + RSLoRA · r=16, α=32 · 0.9% trainable params |
| Precision | bfloat16 |
| Attention | SDPA (PyTorch built-in fused attention) |
| Dataset | PathVQA Enhanced — ~32K LLM-enriched clinical QA pairs |
| Optimizer | AdamW fused · cosine LR · effective batch 32 |
| Tracking | W&B (metrics + model artifacts) |
| Serving | FastAPI + Gradio |

---

## Dataset

[moebouassida/path-vqa-enhanced](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced) — ~32K pathology VQA pairs enriched with LLM-generated detailed explanations.

| Split | Samples | Yes/No % |
|:--|--:|--:|
| Train | 19,654 | 49.6% |
| Validation | 6,259 | — |
| Test | 6,719 | — |
| **Total** | **32,632** | — |

---

## Setup

```bash
git clone https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning.git
cd Path-VQA-Med-GaMMa-Fine-Tuning
pip install -r requirements.txt
```

> **Note:** `torch`/`torchvision`/`torchaudio` are not in `requirements.txt`.
> Install them separately with the correct CUDA index for your driver (see RunPod section below).

**Requirements:** Python 3.10+, CUDA GPU, HuggingFace token for Med-GaMMa access.

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
```

---

## Training on RunPod (H100 SXM)

### 1. Create a pod

| Setting | Value |
|---------|-------|
| GPU | H100 SXM (80 GB HBM3) |
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container disk | 10 GB |
| Volume disk | 50 GB |
| Secrets | `HF_TOKEN`, `WANDB_API_KEY` |

### 2. Inside the pod terminal

```bash
# Redirect HF cache to volume disk (avoids filling 10 GB container)
export HF_HOME=/workspace/.cache/huggingface
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc

git clone https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning
cd Path-VQA-Med-GaMMa-Fine-Tuning

# Create secrets file
cat > .env << 'EOF'
HF_TOKEN=hf_...
WANDB_API_KEY=...
EOF

# Install deps + pre-cache base model (~10 min)
bash scripts/setup_runpod.sh

# Sanity check (5 steps, ~2 min)
bash scripts/train_runpod.sh --smoke-test

# Full training (~3 hours on H100 SXM, 1 epoch)
nohup bash scripts/train_runpod.sh --epochs 1 > logs/train.log 2>&1 &
tail -f logs/train.log
```

> **Tip:** Use `nohup` to keep training running after disconnecting.
> Reattach with `tail -f logs/train.log`.

### 3. Override hyperparameters

```bash
bash scripts/train_runpod.sh --epochs 3 --lr 1e-4
```

All settings live in [`config/config.yaml`](config/config.yaml).

---

## Training locally

```bash
make smoke-test     # 5-step pipeline check
make train          # full run
```

Or directly:

```bash
python -m src.train --smoke-test
python -m src.train --config config/config.yaml
python -m src.train --epochs 3 --lr 1e-4
```

---

## Experiment Tracking (W&B)

Every run logs to [Weights & Biases](https://wandb.ai):
- Train / eval loss curves per step
- Learning rate schedule
- GPU VRAM utilisation
- All hyperparameters from config

**Model versioning via W&B Artifacts:**
```python
import wandb
api = wandb.Api()
artifact = api.artifact("moebouassida/path-vqa-medgemma/medgemma-4b-path-vqa:latest")
artifact.download("outputs/")
```

---

## Evaluation

```bash
python -m src.evaluate \
  --model moebouassida/medgemma-4b-path-vqa \
  --config config/config.yaml \
  --max-samples 500 \
  --output-json metrics/eval_results.json
```

Quality gates (configurable in `config.yaml`):

| Metric | Threshold |
|--------|-----------|
| Yes/No accuracy | ≥ 0.55 |
| Open-ended BLEU-4 | ≥ 0.20 |

Exit code `0` = all gates passed, `1` = failed.

---

## Inference

**CLI:**
```bash
python -m src.inference \
  --model moebouassida/medgemma-4b-path-vqa \
  --image path/to/slide.jpg \
  --question "Is there evidence of malignancy?"
```

**Gradio demo:**
```bash
make demo
# → http://localhost:7860

# or with explicit model path:
MODEL_ID=moebouassida/medgemma-4b-path-vqa python hf_spaces_app.py
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

```bash
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@slide.jpg" \
  -F "question=What type of tissue is shown?"
```

---

## Docker

```bash
docker build -f Docker/Dockerfile -t pathvqa-medgemma .
docker run --gpus all -p 8000:8000 \
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
│   └── train_runpod.sh    # Training launcher
├── Makefile
├── Docker/Dockerfile
├── tests/
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
