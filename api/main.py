"""FastAPI inference service.

Run from the repository root:
    uvicorn api.main:app --port 8000

POST /api/predict with a JPG/PNG file returns per-class probabilities and
which composition techniques cleared their stored thresholds. GET / serves
the demo page from app/.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import CompositionPredictor  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")

app = FastAPI(title="Photography Composition Analyzer")
predictor: CompositionPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    predictor = CompositionPredictor(MODEL_DIR)


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 20 MB")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in ("JPEG", "PNG"):
            raise HTTPException(status_code=415, detail="Only JPG and PNG are supported")
        rgb = np.asarray(image.convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="File is not a readable image")

    results = predictor.predict_array(rgb)
    return {
        "filename": file.filename,
        "predictions": results,
        "note": (
            "Composition labels are subjective; predictions reflect CADB "
            "annotator consensus, not ground truth."
        ),
    }
