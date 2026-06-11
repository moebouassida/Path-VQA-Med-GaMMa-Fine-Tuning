"""
main.py — FastAPI production server for Med-GaMMa PathVQA.

Endpoints:
    GET  /health          — liveness + GPU stats
    GET  /info            — model metadata
    POST /predict         — image URL + question → answer
    POST /predict/upload  — multipart file upload + question → answer
    GET  /docs            — Swagger UI (auto-generated)
"""

import os
import sys
import time
from pathlib import Path
from io import BytesIO
from contextlib import asynccontextmanager

import yaml
import torch
from PIL import Image
import requests

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/final"))

cfg: dict = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

# ── Model (lazy-loaded on first request) ──────────────────────────────────────
_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run training first.",
            )
        from inference import load_model
        _model, _processor = load_model(str(MODEL_PATH))
    return _model, _processor


# ── GPU helpers ────────────────────────────────────────────────────────────────
def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "memory_total_gb": round(props.total_memory / 1e9, 2),
        "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
    }


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Path-VQA · Med-GaMMa API",
    description=(
        "Pathology Visual Question Answering using Med-GaMMa 4B "
        "fine-tuned on PathVQA Enhanced. "
        "Not a medical device — for research use only."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_url: str
    question: str
    max_tokens: int = 256
    temperature: float = 1.0
    do_sample: bool = False


class PredictResponse(BaseModel):
    question: str
    answer: str
    latency_s: float
    model: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check with GPU diagnostics."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
        "model_path_exists": MODEL_PATH.exists(),
        "gpu": _gpu_info(),
    }


@app.get("/info")
def info():
    """Model and task metadata."""
    return {
        "model": "Med-GaMMa 4B (google/medgemma-4b-it)",
        "adapter": "DoRA / LoRA fine-tuned",
        "dataset": "moebouassida/path-vqa-enhanced",
        "task": "Pathology Visual Question Answering",
        "supported_question_types": ["Yes/No", "Open-ended"],
        "hub_model_id": cfg.get("hub_model_id"),
        "version": "2.0.0",
        "disclaimer": "Research use only. Not a medical device.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict_from_url(req: PredictRequest):
    """Predict from an image URL + question."""
    model, processor = get_model()

    try:
        resp = requests.get(req.image_url, timeout=15)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    try:
        from inference import predict
        t0 = time.time()
        answer = predict(
            model, processor, image, req.question,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.do_sample,
        )
        latency = round(time.time() - t0, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        question=req.question,
        answer=answer,
        latency_s=latency,
        model=str(MODEL_PATH),
    )


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_from_upload(
    file: UploadFile = File(..., description="Pathology image (JPEG/PNG/TIFF)"),
    question: str = Form(..., description="Clinical question"),
    max_tokens: int = Form(256),
    temperature: float = Form(1.0),
    do_sample: bool = Form(False),
):
    """Predict from an uploaded image file + question form field."""
    model, processor = get_model()

    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        from inference import predict
        t0 = time.time()
        answer = predict(
            model, processor, image, question,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        latency = round(time.time() - t0, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        question=question,
        answer=answer,
        latency_s=latency,
        model=file.filename or str(MODEL_PATH),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
