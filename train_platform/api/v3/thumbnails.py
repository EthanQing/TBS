from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.illegal_dataset import IllegalDatasetVersion
from train_platform.services.v3.illegal_dataset_cas import load_version_manifest, manifest_cas_file_path
from train_platform.services.v3.illegal_dataset_service import IllegalDatasetService
from train_platform.services.v3.standard_dataset_service import StandardDatasetService
from train_platform.services.v3.thumbnail_service import ThumbnailService
from train_platform.utils.exceptions import NotFoundError
from train_platform.utils.path_utils import resolve_dataset_path


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
    root_token = None
    cache_prefix = None
    cache_dataset_id = int(dataset_id)
    source_path = None

    if kind == "illegal":
        ds = IllegalDatasetService().get_dataset(db, int(dataset_id))
        root_token = ds.storage_path
        if version_id is not None:
            ver = (
                db.query(IllegalDatasetVersion)
                .filter(
                    IllegalDatasetVersion.version_id == int(version_id),
                    IllegalDatasetVersion.illegal_dataset_id == int(ds.illegal_dataset_id),
                )
                .first()
            )
            if not ver:
                raise NotFoundError("Illegal dataset version not found")
            manifest = load_version_manifest(ver)
            if manifest:
                source_path = manifest_cas_file_path(manifest, file_path, required=True)
                cache_prefix = f"v{int(ver.version_id)}"
            elif ver.snapshot_path:
                root_token = ver.snapshot_path
                cache_prefix = f"v{int(ver.version_id)}"
    elif kind == "standard":
        ds = StandardDatasetService().get_dataset(db, int(dataset_id))
        root_token = ds.storage_path
    else:
        raise NotFoundError("Unknown dataset kind")

    svc = ThumbnailService()
    if source_path is not None:
        thumb_path = svc.ensure_thumbnail_from_file(
            dataset_id=cache_dataset_id,
            dataset_namespace=kind,
            source_path=source_path,
            file_rel_path=file_path,
            size=int(size),
            cache_prefix=cache_prefix,
        )
    else:
        dataset_root = resolve_dataset_path(root_token).resolve(strict=False)
        thumb_path = svc.ensure_thumbnail(
            dataset_id=cache_dataset_id,
            dataset_namespace=kind,
            dataset_root=dataset_root,
            file_rel_path=file_path,
            size=int(size),
            cache_prefix=cache_prefix,
        )
    media_type = svc.guess_media_type(thumb_path)
    return FileResponse(path=str(thumb_path), media_type=media_type)
