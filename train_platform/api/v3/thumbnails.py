from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.domains.datasets.thumbnails import detect_thumbnail_media_type, ensure_thumbnail
from train_platform.domains.datasets.illegal.service import IllegalDatasetService
from train_platform.domains.datasets.storage.mounted import resolve_dataset_file
from train_platform.domains.datasets.storage.paths import resolve_storage_token
from train_platform.domains.datasets.standard import StandardDatasetService
from train_platform.utils.exceptions import NotFoundError


router = APIRouter(prefix="/thumbnails", tags=["thumbnails"])


@router.get("/{dataset_kind}/{dataset_id}/{file_path:path}")
def get_thumbnail(
    dataset_kind: str,
    dataset_id: int,
    file_path: str,
    size: int = Query(200, ge=16, le=1024, description="Max edge length for the thumbnail"),
    version_id: int | None = Query(None, description="Optional illegal dataset version_id"),
    db: Session = Depends(get_db),
):
    kind = str(dataset_kind or "").strip().lower()

    if kind == "illegal":
        IllegalDatasetService().get_dataset(db, int(dataset_id))
        raise NotFoundError("Illegal dataset thumbnails are disabled")
    elif kind == "standard":
        ds = StandardDatasetService().get_dataset(db, int(dataset_id))
        dataset_root = resolve_storage_token(ds.storage_path)
        source_path = resolve_dataset_file(dataset_root, file_path)
    else:
        raise NotFoundError("Unknown dataset kind")

    thumb_path = ensure_thumbnail(
        dataset_id=int(dataset_id),
        dataset_namespace=kind,
        source_path=source_path,
        relative_path=file_path,
        size=int(size),
    )
    media_type = detect_thumbnail_media_type(thumb_path)
    return FileResponse(path=str(thumb_path), media_type=media_type)
