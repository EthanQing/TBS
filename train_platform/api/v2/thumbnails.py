from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.dataset import DatasetVersion
from train_platform.services.dataset_service import DatasetService
from train_platform.services.thumbnail_service import ThumbnailService
from train_platform.utils.exceptions import NotFoundError
from train_platform.utils.path_utils import resolve_dataset_path


router = APIRouter(prefix="/thumbnails", tags=["thumbnails"])


@router.get("/{dataset_id}/{file_path:path}")
def get_thumbnail(
    dataset_id: int,
    file_path: str,
    size: int = Query(200, ge=16, le=1024, description="Max edge length for the thumbnail"),
    version_id: int | None = Query(None, description="Optional dataset version_id (for snapshot browsing)"),
    db: Session = Depends(get_db),
):
    ds = DatasetService().get_dataset(db, int(dataset_id))
    root_token = ds.storage_path
    cache_prefix = None
    if version_id is not None:
        ver = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
            .first()
        )
        if not ver:
            raise NotFoundError("Dataset version not found")
        if ver.snapshot_path:
            root_token = ver.snapshot_path
            cache_prefix = f"v{int(ver.version_id)}"

    dataset_root = resolve_dataset_path(root_token).resolve(strict=False)

    svc = ThumbnailService()
    thumb_path = svc.ensure_thumbnail(
        dataset_id=int(dataset_id),
        dataset_root=dataset_root,
        file_rel_path=file_path,
        size=int(size),
        cache_prefix=cache_prefix,
    )
    media_type = svc.guess_media_type(thumb_path)
    return FileResponse(path=str(thumb_path), media_type=media_type)
