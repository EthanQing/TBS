from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.core.config import settings
from train_platform.domains.training.custom_models import (
    get_package,
    ingest_custom_model_package,
    list_packages,
    retire_custom_model_package,
)
from train_platform.schemas.v3.custom_models import (
    CustomModelPackageListResponse,
    CustomModelPackageOut,
)
from train_platform.utils.exceptions import ValidationError

router = APIRouter(prefix="/custom-models", tags=["custom-models"])


@router.get("", response_model=CustomModelPackageListResponse)
def list_custom_model_packages(
    include_retired: bool = Query(False, description="Include retired packages"),
    name: str | None = Query(None, description="Filter by package name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    items, total = list_packages(db, include_retired=include_retired, name=name, skip=skip, limit=limit)
    return CustomModelPackageListResponse(items=items, total=total)


@router.get("/{package_id}", response_model=CustomModelPackageOut)
def get_custom_model_package(
    package_id: int,
    db: Session = Depends(get_db),
):
    return get_package(db, package_id)


@router.post("/upload", response_model=CustomModelPackageOut, status_code=201)
async def upload_custom_model_package(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest a custom model source archive.
    
    Accepts .zip or tar archives containing source code and tbs-model.yaml.
    Must not contain trained weights (*.pt, *.pth, *.onnx, etc.).
    """
    filename = file.filename or ""
    suffix = "".join(Path(filename).suffixes).lower()
    if not (suffix.endswith(".zip") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz") or suffix.endswith(".tar")):
        raise ValidationError(f"Unsupported package archive format: {filename}. Expected .zip or .tar.gz archive.")

    settings.ensure_dirs()
    temp_upload_dir = settings.temp_dir / "custom_model_uploads"
    temp_upload_dir.mkdir(parents=True, exist_ok=True)

    temp_file = temp_upload_dir / f"upload_{uuid.uuid4().hex}_{Path(filename).name}"
    try:
        with open(temp_file, "wb") as f:
            shutil.copyfileobj(file.file, f)

        package = ingest_custom_model_package(db, archive_file_path=temp_file)
        return package
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


@router.post("/{package_id}/retire", response_model=CustomModelPackageOut)
def retire_package(
    package_id: int,
    db: Session = Depends(get_db),
):
    """Retire a package so it cannot be used for future architectures or training runs."""
    return retire_custom_model_package(db, package_id)


__all__ = ["router"]
