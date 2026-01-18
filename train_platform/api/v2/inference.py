from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.core.config import settings
from train_platform.schemas.v2.inference import InferenceRunCreate, InferenceRunOut, InferenceUploadOut
from train_platform.services.inference_service import InferenceService
from train_platform.utils.exceptions import ValidationError


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
    Upload a single image to BASE_TEMP_DIR so it can be referenced by `input_path`.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
        raise ValidationError("Unsupported image format")

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
