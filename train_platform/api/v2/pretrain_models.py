from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from train_platform.core.config import settings
from train_platform.schemas.v2.pretrain_models import PretrainUploadOut
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/pretrain-models", tags=["pretrain-models"])


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-._")
    return cleaned or "weights"


@router.post("/upload", response_model=PretrainUploadOut, status_code=201)
async def upload_pretrain_weights(file: UploadFile = File(...)):
    """
    Upload a single pretrain weights file to BASE_PRETRAIN_MODELS_DIR.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pt", ".pth", ".ckpt"):
        raise ValidationError("Unsupported weights format (.pt/.pth/.ckpt)")

    settings.ensure_dirs()
    out_dir = settings.pretrain_models_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize_name(Path(filename).stem)
    out_name = f"{stem}-{uuid.uuid4().hex}{suffix}"
    out_path = (out_dir / out_name).resolve(strict=False)
    if settings.pretrain_models_dir.resolve() not in out_path.parents:
        raise ValidationError("Unsafe upload path")

    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    token = out_name
    return PretrainUploadOut(token=token, path=f"/static/pretrain/{token}", filename=Path(filename).name or out_name)
