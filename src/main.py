"""
main.py — FastAPI serving for Med-GaMMa PathVQA.

Endpoints:
    GET  /health          — liveness check
    POST /predict         — image URL + question -> answer
    POST /predict/upload  — image file upload + question -> answer
    GET  /docs            — Swagger UI
"""

import os
import sys
from pathlib import Path
from io import BytesIO

import yaml
from PIL import Image
import requests

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/final"))

cfg = yaml.safe_load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Path-VQA Med-GaMMa API",
    description="Visual Question Answering on pathology images using fine-tuned Med-GaMMa.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Train the model first.",
            )
        from inference import load_model

        _model, _processor = load_model(str(MODEL_PATH))
    return _model, _processor


# ── Request Schema ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_url: str
    question: str
    max_tokens: int = 256


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "device": "cuda" if _model is not None else "unknown",
    }


@app.post("/predict")
def predict_from_url(req: PredictRequest):
    """Upload image via URL + question → pathologically detailed answer."""
    model, processor = get_model()

    try:
        response = requests.get(req.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    try:
        from inference import predict

        answer = predict(model, processor, image, req.question, req.max_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "question": req.question,
        "answer": answer,
        "model": str(MODEL_PATH),
    }


@app.post("/predict/upload")
async def predict_from_upload(
    file: UploadFile = File(..., description="Pathology image file"),
    question: str = Form(..., description="Clinical question"),
    max_tokens: int = Form(256),
):
    """Upload image file + question → pathologically detailed answer."""
    model, processor = get_model()

    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        from inference import predict

        answer = predict(model, processor, image, question, max_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "question": question,
        "answer": answer,
        "filename": file.filename,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
