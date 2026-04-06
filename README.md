# 🔬 Path-VQA Med-GaMMa Fine-Tuning

**Fine-tune Med-GaMMa 4B on PathVQA Enhanced for pathology Visual Question Answering.**
Answers yes/no and open-ended clinical questions from H&E and other pathology images using LoRA fine-tuning on [moebouassida/path-vqa-enhanced](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced).

[![CI](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
PathVQA Enhanced (HF Hub)
        │
        ▼
data_processing.py  ← shuffle + conversation format
        │
        ▼
train.py  ← LoRA fine-tuning (SFTTrainer, 4-bit NF4)
        │
        ▼
outputs/final/
        ├── src/main.py        ← FastAPI inference API
        ├── src/inference.py   ← CLI inference
        ├── src/evaluate.py    ← eval + quality gates
        └── hf_spaces_app.py   ← Gradio demo
```

| Component | Details |
|-----------|---------|
| Base model | Med-GaMMa 4B (`google/medgemma-4b-it`) |
| Fine-tuning | LoRA r=32, alpha=64 · SFTTrainer (TRL) |
| Quantization | 4-bit NF4 + bfloat16 |
| Dataset | PathVQA Enhanced — LLM-enriched clinical answers |
| Tracking | W&B + MLflow |
| Serving | FastAPI + Gradio |

---

## Setup

```bash
git clone https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning.git
cd Path-VQA-Med-GaMMa-Fine-Tuning
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA GPU (RTX 4090 recommended), HF token for Med-GaMMa access.

Set environment variables:
```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...         # optional
export OPENROUTER_API_KEY=...    # optional
```

---

## Training

```bash
# Smoke test (5 steps, verify pipeline works)
python src/train.py --config config/config.yaml --smoke-test

# Full training
python src/train.py --config config/config.yaml

# Override epochs or LR
python src/train.py --config config/config.yaml --epochs 5 --lr 1e-4
```

Checkpoints save to `outputs/` and push to HF Hub (set `hub_model_id` in config).

---

## Evaluation

```bash
python src/evaluate.py --model outputs/final --config config/config.yaml
```

Quality gates (configurable in `config.yaml`):
- Yes/No exact match ≥ 0.55
- Open-ended BLEU ≥ 0.20

---

## Inference

**CLI:**
```bash
python src/inference.py --model outputs/final \
  --image path/to/slide.jpg \
  --question "Is there evidence of malignancy?"
```

**Gradio demo:**
```bash
python hf_spaces_app.py
# → http://localhost:7860
```

**FastAPI server:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Image URL + question → answer |
| `POST` | `/predict/upload` | Image file upload + question → answer |
| `GET` | `/docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://...", "question": "Is this benign?"}'
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
│   ├── train.py           # LoRA fine-tuning script
│   ├── inference.py       # Model loading + inference
│   ├── data_processing.py # Dataset loader + conversation format
│   ├── evaluate.py        # Evaluation + quality gates
│   ├── metrics.py         # Exact match, BLEU, VQA score
│   └── main.py            # FastAPI server
├── hf_spaces_app.py       # Gradio demo
├── config/config.yaml     # All hyperparameters
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

MIT — see [LICENSE](LICENSE).

> ⚠️ Research use only. Not a medical device. All outputs must be interpreted by qualified healthcare professionals.
