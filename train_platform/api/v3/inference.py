from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, UploadFile
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
    VideoFrameResult,
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

    # Resolve video path
    from train_platform.utils.path_utils import resolve_temp_path
    video_path = resolve_temp_path(payload.video_token)
    if not video_path.exists() or not video_path.is_file():
        raise ValidationError(f"Video file not found: {payload.video_token}")

    # Extract frames using OpenCV
    try:
        import cv2
    except ImportError:
        raise ValidationError("OpenCV (cv2) is not installed — required for video inference")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValidationError(f"Failed to open video: {payload.video_token}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(payload.frame_interval))

    # Create temp dir for extracted frames
    frames_dir = settings.temp_dir / "inference_video_frames" / uuid.uuid4().hex
    frames_dir.mkdir(parents=True, exist_ok=True)

    results: List[VideoFrameResult] = []
    total_start = time.time()
    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                # Save frame as image
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                result = VideoFrameResult(frame_index=frame_idx)
                try:
                    run = svc.run_inference(
                        db,
                        model_version_id=int(payload.model_version_id),
                        input_path=str(frame_path),
                        conf=float(payload.conf),
                        iou=float(payload.iou),
                    )
                    result.output = run.output
                    result.error_message = run.error_message
                except Exception as e:
                    result.error_message = f"{type(e).__name__}: {e}"

                results.append(result)
                processed += 1

            frame_idx += 1
    finally:
        cap.release()

    total_time_ms = round((time.time() - total_start) * 1000, 1)
    return VideoInferenceOut(
        results=results,
        total_frames=total_frames,
        processed_frames=processed,
        total_time_ms=total_time_ms,
    )
