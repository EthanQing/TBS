from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.domains.inference.input import save_uploaded_file
from train_platform.domains.inference.service import InferenceService
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


router = APIRouter(prefix="/inference-runs", tags=["inference"])

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
    token = save_uploaded_file(file.filename or "", file.file)
    return InferenceUploadOut(token=token, path=f"/static/temp/{token}")


@router.post("/batch", response_model=BatchInferenceOut, status_code=200)
def batch_inference(payload: BatchInferenceCreate, db: Session = Depends(get_db)):
    """
    Run inference on multiple uploaded images sequentially.
    """
    raw = InferenceService().run_batch_inference(
        db,
        model_version_id=int(payload.model_version_id),
        input_tokens=list(payload.input_tokens),
        conf=float(payload.conf),
        iou=float(payload.iou),
    )
    return BatchInferenceOut(
        results=[BatchInferenceResultItem.model_validate(item) for item in raw["results"]],
        total=int(raw["total"]),
        success_count=int(raw["success_count"]),
        total_time_ms=float(raw["total_time_ms"]),
    )


@router.post("/video", response_model=VideoInferenceOut, status_code=200)
def video_inference(payload: VideoInferenceCreate, db: Session = Depends(get_db)):
    """
    Run inference on a video file, extracting frames at the given interval.
    """
    data = InferenceService().run_video_inference(
        db,
        model_version_id=int(payload.model_version_id),
        video_token=payload.video_token,
        frame_interval=int(payload.frame_interval),
        conf=float(payload.conf),
        iou=float(payload.iou),
    )
    return VideoInferenceOut.model_validate(data)
