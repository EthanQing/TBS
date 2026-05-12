from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.illegal_dataset import IllegalDataset, IllegalDatasetEvent, IllegalDatasetVersion
from train_platform.schemas.v3.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v3.illegal_datasets import (
    DatasetFileOut,
    DatasetImageAnnotationsOut,
    DatasetImageUploadOut,
    DatasetStatisticsOut,
    DatasetViewOut,
    IllegalDatasetCreate,
    IllegalDatasetDetailOut,
    IllegalDatasetEventOut,
    IllegalDatasetLabelMappingsOut,
    IllegalDatasetLabelMappingsUpdate,
    IllegalDatasetListOut,
    IllegalDatasetOut,
    IllegalDatasetPublishOut,
    IllegalDatasetPublishRequest,
    IllegalDatasetRawLabelsOut,
    IllegalDatasetUpdate,
    IllegalDatasetVersionOut,
)
from train_platform.services.v3.illegal_dataset_service import IllegalDatasetService


router = APIRouter(prefix="/illegal-datasets", tags=["illegal-datasets"])
svc = IllegalDatasetService()


@router.post("", response_model=IllegalDatasetOut, status_code=201)
def create_illegal_dataset(payload: IllegalDatasetCreate, db: Session = Depends(get_db)):
    return svc.create_dataset(db, obj=payload.model_dump())


@router.get("", response_model=Page[IllegalDatasetListOut])
def list_illegal_datasets(
    page: int = 1,
    page_size: int = 50,
    format: str | None = Query(None),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    q = db.query(IllegalDataset)
    if format:
        q = q.filter(IllegalDataset.format == str(format))
    total = q.count()
    items = svc.list_datasets(db, skip=skip, limit=page_size, format=format)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.get("/{illegal_dataset_id}", response_model=IllegalDatasetOut)
def get_illegal_dataset(illegal_dataset_id: int, db: Session = Depends(get_db)):
    return svc.get_dataset(db, illegal_dataset_id)


@router.get("/{illegal_dataset_id}/detail", response_model=IllegalDatasetDetailOut)
def get_illegal_dataset_detail(
    illegal_dataset_id: int,
    versions_limit: int = 20,
    events_limit: int = 20,
    db: Session = Depends(get_db),
):
    return svc.get_detail(db, int(illegal_dataset_id), versions_limit=int(versions_limit), events_limit=int(events_limit))


@router.patch("/{illegal_dataset_id}", response_model=IllegalDatasetOut)
def update_illegal_dataset(illegal_dataset_id: int, payload: IllegalDatasetUpdate, db: Session = Depends(get_db)):
    return svc.update_dataset(db, illegal_dataset_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{illegal_dataset_id}", response_model=DeleteResponse)
def delete_illegal_dataset(
    illegal_dataset_id: int,
    delete_files: bool = False,
    force: bool = Query(False),
    db: Session = Depends(get_db),
):
    svc.delete_dataset(db, illegal_dataset_id, delete_files=bool(delete_files), force=bool(force))
    return DeleteResponse(ok=True, message="Illegal dataset deleted")


@router.post("/{illegal_dataset_id}/upload", response_model=IllegalDatasetOut, status_code=201)
def upload_illegal_dataset_archive(
    illegal_dataset_id: int,
    file: UploadFile = File(...),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    db: Session = Depends(get_db),
):
    return svc.upload_archive(db, illegal_dataset_id, file, message=message, created_by=created_by, append=False)


@router.post("/{illegal_dataset_id}/append", response_model=IllegalDatasetOut, status_code=201)
def append_illegal_dataset_archive(
    illegal_dataset_id: int,
    file: UploadFile = File(...),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    db: Session = Depends(get_db),
):
    return svc.upload_archive(db, illegal_dataset_id, file, message=message, created_by=created_by, append=True)


@router.post("/{illegal_dataset_id}/uploads/images", response_model=DatasetImageUploadOut, status_code=201)
def upload_illegal_dataset_images(
    illegal_dataset_id: int,
    files: list[UploadFile] = File(...),
    relative_dir: str = Form("images/uploads"),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    db: Session = Depends(get_db),
):
    return svc.upload_images(
        db,
        illegal_dataset_id,
        files=files,
        relative_dir=relative_dir,
        message=message,
        created_by=created_by,
    )


@router.get("/{illegal_dataset_id}/versions", response_model=Page[IllegalDatasetVersionOut])
def list_illegal_dataset_versions(illegal_dataset_id: int, page: int = 1, page_size: int = 50, db: Session = Depends(get_db)):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    total = db.query(IllegalDatasetVersion).filter(IllegalDatasetVersion.illegal_dataset_id == int(illegal_dataset_id)).count()
    items = svc.list_versions(db, illegal_dataset_id, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("/{illegal_dataset_id}/versions/{version_id}/activate", response_model=IllegalDatasetOut)
def activate_illegal_dataset_version(illegal_dataset_id: int, version_id: int, db: Session = Depends(get_db)):
    return svc.activate_version(db, illegal_dataset_id, version_id)


@router.get("/{illegal_dataset_id}/versions/{version_id}/files/{file_path:path}")
def get_illegal_dataset_version_file(
    illegal_dataset_id: int,
    version_id: int,
    file_path: str,
    db: Session = Depends(get_db),
):
    path = svc.get_version_file_path(db, illegal_dataset_id, version_id, file_path)
    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return FileResponse(path=str(path), media_type=media_type)


@router.get("/{illegal_dataset_id}/events", response_model=Page[IllegalDatasetEventOut])
def list_illegal_dataset_events(
    illegal_dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    total = db.query(IllegalDatasetEvent).filter(IllegalDatasetEvent.illegal_dataset_id == int(illegal_dataset_id)).count()
    items = svc.list_events(db, illegal_dataset_id, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.get("/{illegal_dataset_id}/raw-labels", response_model=IllegalDatasetRawLabelsOut)
def get_illegal_dataset_raw_labels(illegal_dataset_id: int, db: Session = Depends(get_db)):
    return {"labels": svc.get_raw_labels(db, illegal_dataset_id)}


@router.get("/{illegal_dataset_id}/label-mappings", response_model=IllegalDatasetLabelMappingsOut)
def get_illegal_dataset_label_mappings(illegal_dataset_id: int, db: Session = Depends(get_db)):
    rows = svc.get_label_mappings(db, illegal_dataset_id)

    def _out(row):
        is_delete = (
            str(row.status or "").strip().lower() == "delete"
            or str(row.mapped_label or "").strip() == "__DISCARD__"
        )
        status = "delete" if is_delete else "keep"
        return {
            "raw_label": row.raw_label,
            "mapped_label": "" if status == "delete" else row.mapped_label,
            "status": status,
        }

    return {
        "items": [_out(row) for row in rows]
    }


@router.put("/{illegal_dataset_id}/label-mappings", response_model=IllegalDatasetOut)
def update_illegal_dataset_label_mappings(
    illegal_dataset_id: int,
    payload: IllegalDatasetLabelMappingsUpdate,
    db: Session = Depends(get_db),
):
    return svc.update_label_mappings(db, illegal_dataset_id, items=[item.model_dump() for item in payload.items])


@router.post("/{illegal_dataset_id}/publish", response_model=IllegalDatasetPublishOut, status_code=201)
def publish_standard_dataset(
    illegal_dataset_id: int,
    payload: IllegalDatasetPublishRequest,
    db: Session = Depends(get_db),
):
    return svc.publish_standard_dataset(db, illegal_dataset_id, obj=payload.model_dump())


@router.get("/{illegal_dataset_id}/view", response_model=DatasetViewOut)
def get_illegal_dataset_view(
    illegal_dataset_id: int,
    version_id: int | None = Query(None),
    class_id: int | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return svc.get_view(db, illegal_dataset_id, version_id=version_id, class_id=class_id, page=page, page_size=page_size)


@router.get("/{illegal_dataset_id}/image-annotations", response_model=DatasetImageAnnotationsOut)
def get_illegal_dataset_image_annotations(
    illegal_dataset_id: int,
    image_path: str = Query(...),
    version_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    return svc.get_image_annotations(db, illegal_dataset_id, image_path=image_path, version_id=version_id)


@router.get("/{illegal_dataset_id}/statistics", response_model=DatasetStatisticsOut)
def get_illegal_dataset_statistics(
    illegal_dataset_id: int,
    version_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    return svc.get_statistics(db, illegal_dataset_id, version_id=version_id)


@router.get("/{illegal_dataset_id}/files", response_model=Page[DatasetFileOut])
def list_illegal_dataset_files(
    illegal_dataset_id: int,
    version_id: int | None = Query(None),
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    items, total = svc.list_files(db, illegal_dataset_id, version_id=version_id, page=page, page_size=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}
