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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import LABELS  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.inference import CompositionPredictor  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")

app = FastAPI(title="Photography Composition Analyzer")
predictor: CompositionPredictor | None = None
gradcam: GradCAM | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor, gradcam
    predictor = CompositionPredictor(MODEL_DIR)
    gradcam = GradCAM(model=predictor.model)


def _decode_upload(raw: bytes) -> np.ndarray:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 20 MB")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in ("JPEG", "PNG"):
            raise HTTPException(status_code=415, detail="Only JPG and PNG are supported")
        return np.asarray(image.convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="File is not a readable image")


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
    rgb = _decode_upload(await file.read())
    results = predictor.predict_array(rgb)
    return {
        "filename": file.filename,
        "predictions": results,
        "note": (
            "Composition labels are subjective; predictions reflect CADB "
            "annotator consensus, not ground truth."
        ),
    }


@app.post("/api/gradcam")
async def explain(file: UploadFile = File(...), label: str = Form(...)):
    """Grad-CAM overlay (PNG) showing which regions drove one class's score."""
    if gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if label not in LABELS:
        raise HTTPException(status_code=422, detail=f"Unknown label: {label}")
    rgb = _decode_upload(await file.read())

    overlay = gradcam.overlay(rgb, label)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
