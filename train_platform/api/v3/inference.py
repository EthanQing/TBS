from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, UploadFile
import requests
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.core.config import settings
from train_platform.schemas.v3.inference import (
    BatchInferenceCreate,
    BatchInferenceOut,
    BatchInferenceResultItem,
    InferenceRunCreate,
    InferenceRunOut,
    InferenceUploadOut,
    VideoInferenceCreate,
    VideoInferenceOut,
)
from train_platform.services.v3.inference_service import InferenceService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/inference-runs", tags=["inference"])

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
ALL_UPLOAD_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


@router.post("", response_model=InferenceRunOut, status_code=201)
def create_inference_run(payload: InferenceRunCreate, db: Session = Depends(get_db)):
    return InferenceService().run_inference(
        db,
        model_version_id=int(payload.model_version_id),
        deployment_id=int(payload.deployment_id) if payload.deployment_id is not None else None,
        input_path=payload.input_path,
        image_url=payload.image_url,
        input_meta=payload.input_meta,
        conf=float(payload.conf),
        iou=float(payload.iou),
    )


@router.post("/upload", response_model=InferenceUploadOut, status_code=201)
async def upload_inference_input(file: UploadFile = File(...)):
    """
    Upload a single image or video to temp storage for inference.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in ALL_UPLOAD_SUFFIXES:
        raise ValidationError(f"Unsupported format: {suffix}. Allowed: {', '.join(sorted(ALL_UPLOAD_SUFFIXES))}")

    settings.ensure_dirs()
    out_dir = settings.temp_dir / "inference_uploads"
    out_dir.mkdir(parents=True, exist_ok=True)

    token = f"inference_uploads/{uuid.uuid4().hex}{suffix}"
    out_path = (settings.temp_dir / token).resolve(strict=False)
    if settings.temp_dir.resolve() not in out_path.parents:
        raise ValidationError("Unsafe upload path")

    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return InferenceUploadOut(token=token, path=f"/static/temp/{token}")


@router.post("/batch", response_model=BatchInferenceOut, status_code=200)
def batch_inference(payload: BatchInferenceCreate, db: Session = Depends(get_db)):
    """
    Run inference on multiple uploaded images sequentially.
    """
    svc = InferenceService()
    results: List[BatchInferenceResultItem] = []
    total_start = time.time()
    success_count = 0

    for token in payload.input_tokens:
        t0 = time.time()
        item = BatchInferenceResultItem(
            token=token,
            filename=Path(token).name,
        )
        try:
            run = svc.run_inference(
                db,
                model_version_id=int(payload.model_version_id),
                input_path=token,
                conf=float(payload.conf),
                iou=float(payload.iou),
            )
            item.output = run.output
            item.error_message = run.error_message
            if run.output and not run.error_message:
                success_count += 1
        except Exception as e:
            item.error_message = f"{type(e).__name__}: {e}"
        item.inference_time_ms = round((time.time() - t0) * 1000, 1)
        results.append(item)

    total_time_ms = round((time.time() - total_start) * 1000, 1)
    return BatchInferenceOut(
        results=results,
        total=len(results),
        success_count=success_count,
        total_time_ms=total_time_ms,
    )


@router.post("/video", response_model=VideoInferenceOut, status_code=200)
def video_inference(payload: VideoInferenceCreate, db: Session = Depends(get_db)):
    """
    Run inference on a video file, extracting frames at the given interval.
    """
    svc = InferenceService()
    ctx = svc.resolve_model_context(db, model_version_id=int(payload.model_version_id))
    engine = str(ctx.get("engine") or "ultralytics-yolo").strip().lower()
    if engine == "paddle-det":
        worker_url = os.getenv("PADDLE_INFERENCE_WORKER_URL", "http://127.0.0.1:18003").rstrip("/")
        timeout = float(os.getenv("PADDLE_INFERENCE_WORKER_TIMEOUT", "240"))
    else:
        worker_url = os.getenv("INFERENCE_WORKER_URL", "http://127.0.0.1:18002").rstrip("/")
        timeout = float(os.getenv("INFERENCE_WORKER_TIMEOUT", "120"))

    headers = {}
    token = str(settings.internal_api_token or "").strip()
    if token:
        headers["X-Internal-Token"] = token
    worker_payload: Dict[str, Any] = {
        "weights_path": str(ctx.get("weights_path") or ""),
        "video_token": payload.video_token,
        "frame_interval": int(payload.frame_interval),
        "conf": float(payload.conf),
        "iou": float(payload.iou),
    }
    if engine == "paddle-det":
        worker_payload["config_path"] = str(ctx.get("config_path") or "")
    try:
        resp = requests.post(
            f"{worker_url}/internal/inference/video-frames",
            json=worker_payload,
            timeout=timeout,
            headers=headers,
        )
    except Exception as e:
        raise ValidationError(f"Failed to call inference worker: {type(e).__name__}: {e}") from e

    try:
        data = resp.json()
    except Exception as e:
        raise ValidationError(f"Inference worker returned non-JSON response: {type(e).__name__}: {e}") from e
    if resp.status_code != 200:
        detail = data.get("detail") if isinstance(data, dict) else None
        raise ValidationError(str(detail or f"Inference worker error {resp.status_code}: {resp.text}"))
    return VideoInferenceOut.model_validate(data)
