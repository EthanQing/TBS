from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetEvent
from train_platform.schemas.v3.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v3.standard_datasets import (
    DatasetFileOut,
    DatasetImageAnnotationsOut,
    DatasetStatisticsOut,
    DatasetViewOut,
    StandardDatasetCreate,
    StandardDatasetDetailOut,
    StandardDatasetEventOut,
    StandardDatasetOut,
    StandardDatasetUpdate,
)
from train_platform.services.v3.standard_dataset_service import StandardDatasetService


router = APIRouter(prefix="/standard-datasets", tags=["standard-datasets"])
svc = StandardDatasetService()


@router.post("", response_model=StandardDatasetOut, status_code=201)
def create_standard_dataset(payload: StandardDatasetCreate, db: Session = Depends(get_db)):
    return svc.create_dataset(db, obj=payload.model_dump())


@router.get("", response_model=Page[StandardDatasetOut])
def list_standard_datasets(
    page: int = 1,
    page_size: int = 50,
    format: str | None = Query(None),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    q = db.query(StandardDataset)
    if format:
        q = q.filter(StandardDataset.format == str(format))
    total = q.count()
    items = svc.list_datasets(db, skip=skip, limit=page_size, format=format)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.get("/{standard_dataset_id}", response_model=StandardDatasetOut)
def get_standard_dataset(standard_dataset_id: int, db: Session = Depends(get_db)):
    return svc.get_dataset(db, standard_dataset_id)


@router.get("/{standard_dataset_id}/detail", response_model=StandardDatasetDetailOut)
def get_standard_dataset_detail(standard_dataset_id: int, events_limit: int = 20, db: Session = Depends(get_db)):
    return svc.get_detail(db, standard_dataset_id, events_limit=events_limit)


@router.patch("/{standard_dataset_id}", response_model=StandardDatasetOut)
def update_standard_dataset(standard_dataset_id: int, payload: StandardDatasetUpdate, db: Session = Depends(get_db)):
    return svc.update_dataset(db, standard_dataset_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{standard_dataset_id}", response_model=DeleteResponse)
def delete_standard_dataset(
    standard_dataset_id: int,
    delete_files: bool = False,
    force: bool = Query(False),
    db: Session = Depends(get_db),
):
    svc.delete_dataset(db, standard_dataset_id, delete_files=bool(delete_files), force=bool(force))
    return DeleteResponse(ok=True, message="Standard dataset deleted")


@router.post("/{standard_dataset_id}/upload", response_model=StandardDatasetOut, status_code=201)
def upload_standard_dataset_archive(
    standard_dataset_id: int,
    file: UploadFile = File(...),
    created_by: str | None = Form(None),
    db: Session = Depends(get_db),
):
    return svc.upload_archive(db, standard_dataset_id, file, created_by=created_by)


@router.get("/{standard_dataset_id}/events", response_model=Page[StandardDatasetEventOut])
def list_standard_dataset_events(
    standard_dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    total = db.query(StandardDatasetEvent).filter(StandardDatasetEvent.standard_dataset_id == int(standard_dataset_id)).count()
    items = svc.list_events(db, standard_dataset_id, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.get("/{standard_dataset_id}/view", response_model=DatasetViewOut)
def get_standard_dataset_view(
    standard_dataset_id: int,
    class_id: int | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return svc.get_view(db, standard_dataset_id, class_id=class_id, page=page, page_size=page_size)


@router.get("/{standard_dataset_id}/image-annotations", response_model=DatasetImageAnnotationsOut)
def get_standard_dataset_image_annotations(
    standard_dataset_id: int,
    image_path: str = Query(...),
    db: Session = Depends(get_db),
):
    return svc.get_image_annotations(db, standard_dataset_id, image_path=image_path)


@router.get("/{standard_dataset_id}/statistics", response_model=DatasetStatisticsOut)
def get_standard_dataset_statistics(standard_dataset_id: int, db: Session = Depends(get_db)):
    return svc.get_statistics(db, standard_dataset_id)


@router.get("/{standard_dataset_id}/files", response_model=Page[DatasetFileOut])
def list_standard_dataset_files(
    standard_dataset_id: int,
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    items, total = svc.list_files(db, standard_dataset_id, page=page, page_size=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}
