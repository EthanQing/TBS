from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.dataset_event import DatasetEvent
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.datasets import (
    DatasetCreate,
    DatasetDetailOut,
    DatasetEventOut,
    DatasetFileOut,
    DatasetImageUploadOut,
    DatasetOut,
    DatasetSplitRequest,
    DatasetSplitResultOut,
    DatasetSplitSummary,
    DatasetStatisticsOut,
    DatasetUpdate,
    DatasetVersionCreate,
    DatasetVersionDiffOut,
    DatasetVersionOut,
)
from train_platform.services.dataset_service import DatasetService
from train_platform.services.file_service import FileService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=Page[DatasetOut])
def list_datasets(page: int = 1, page_size: int = 50, db: Session = Depends(get_db)):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    total = db.query(Dataset).count()
    items = DatasetService().list_datasets(db, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    return DatasetService().create_dataset(db, obj=payload.model_dump())


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    return DatasetService().get_dataset(db, dataset_id)


@router.get("/{dataset_id}/detail", response_model=DatasetDetailOut)
def get_dataset_detail(
    dataset_id: int,
    versions_limit: int = 20,
    events_limit: int = 20,
    db: Session = Depends(get_db),
):
    versions_limit = max(0, min(int(versions_limit), 200))
    events_limit = max(0, min(int(events_limit), 200))
    return DatasetService().get_detail(db, int(dataset_id), versions_limit=versions_limit, events_limit=events_limit)


@router.patch("/{dataset_id}", response_model=DatasetOut)
def update_dataset(dataset_id: int, payload: DatasetUpdate, db: Session = Depends(get_db)):
    return DatasetService().update_dataset(db, dataset_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{dataset_id}", response_model=DeleteResponse)
def delete_dataset(
    dataset_id: int,
    delete_files: bool = False,
    force: bool = Query(False, description="Delete dataset and all related projects/training runs/model versions"),
    db: Session = Depends(get_db),
):
    DatasetService().delete_dataset(db, dataset_id, delete_files=bool(delete_files), force=bool(force))
    return DeleteResponse(ok=True, message="Dataset deleted")


@router.post("/{dataset_id}/upload", response_model=DatasetOut, status_code=201)
async def upload_dataset_archive(
    dataset_id: int,
    file: UploadFile = File(...),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    create_version: bool = Form(True),
    activate: bool = Form(True),
    db: Session = Depends(get_db),
):
    ds, _ver = DatasetService().upload_dataset_archive(
        db,
        int(dataset_id),
        file=file,
        message=message,
        created_by=created_by,
        create_version=bool(create_version),
        activate=bool(activate),
    )
    return ds


@router.post("/import", response_model=DatasetOut, status_code=201)
@router.post("/upload", response_model=DatasetOut, status_code=201)
async def import_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    dataset_type: str = Form(...),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
):
    try:
        dt = DatasetType(str(dataset_type))
    except Exception as e:
        raise ValidationError("Invalid dataset_type")
    FileService().upload_dataset(file, name, dt)
    return DatasetService().create_dataset(
        db,
        obj={
            "name": name,
            "dataset_type": dt,
            "storage_path": name,  # upload_dataset extracts to BASE_DATASETS_DIR/<name>
            "description": description,
        },
    )


@router.post("/{dataset_id}/uploads/images", response_model=DatasetImageUploadOut, status_code=201)
async def upload_dataset_images(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    relative_dir: str = Form("images"),
    labels: list[UploadFile] | None = File(None),
    labels_relative_dir: str | None = Form(None),
    require_labels: bool = Form(True),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    create_version: bool = Form(True),
    create_snapshot: bool = Form(False),
    activate: bool = Form(True),
    db: Session = Depends(get_db),
):
    return DatasetService().upload_images(
        db,
        int(dataset_id),
        files=files,
        relative_dir=relative_dir,
        labels=labels,
        labels_relative_dir=labels_relative_dir,
        require_labels=bool(require_labels),
        message=message,
        created_by=created_by,
        create_version=bool(create_version),
        create_snapshot=bool(create_snapshot),
        activate=bool(activate),
    )


@router.get("/{dataset_id}/events", response_model=Page[DatasetEventOut])
def list_dataset_events(
    dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    event_type: str | None = None,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    q = db.query(DatasetEvent).filter(DatasetEvent.dataset_id == int(dataset_id))
    if event_type:
        q = q.filter(DatasetEvent.event_type == str(event_type))
    total = q.count()

    items = DatasetService().list_events(db, int(dataset_id), skip=skip, limit=page_size, event_type=event_type)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.get("/{dataset_id}/versions", response_model=Page[DatasetVersionOut])
def list_dataset_versions(dataset_id: int, page: int = 1, page_size: int = 50, db: Session = Depends(get_db)):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    total = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == int(dataset_id)).count()
    items = DatasetService().list_versions(db, dataset_id, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("/{dataset_id}/versions", response_model=DatasetVersionOut, status_code=201)
def create_dataset_version(dataset_id: int, payload: DatasetVersionCreate, db: Session = Depends(get_db)):
    return DatasetService().create_version(
        db,
        dataset_id,
        message=payload.message,
        created_by=payload.created_by,
        create_snapshot=bool(payload.create_snapshot),
    )


@router.post("/{dataset_id}/versions/{version_id}/activate", response_model=DatasetOut)
def activate_dataset_version(dataset_id: int, version_id: int, db: Session = Depends(get_db)):
    return DatasetService().activate_version(db, dataset_id, version_id)


@router.get("/{dataset_id}/statistics", response_model=DatasetStatisticsOut)
def get_dataset_statistics(dataset_id: int, version_id: int | None = None, db: Session = Depends(get_db)):
    return DatasetService().get_statistics(db, dataset_id, version_id=version_id)


@router.get("/{dataset_id}/versions/{version_id}/diff", response_model=DatasetVersionDiffOut)
def diff_dataset_versions(
    dataset_id: int,
    version_id: int,
    base_version_id: int | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    return DatasetService().diff_versions(db, dataset_id, version_id, base_version_id=base_version_id, limit=limit)


@router.get("/{dataset_id}/files", response_model=Page[DatasetFileOut])
def list_dataset_files(
    dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    version_id: int | None = None,
    kind: str = "image",
    prefix: str | None = None,
    q: str | None = None,
    include_missing: bool = False,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    items, total = DatasetService().list_files(
        db,
        int(dataset_id),
        version_id=version_id,
        kind=kind,
        prefix=prefix,
        q=q,
        skip=skip,
        limit=page_size,
        include_missing=bool(include_missing),
    )
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("/{dataset_id}/split", response_model=DatasetSplitSummary)
def split_dataset(dataset_id: int, payload: DatasetSplitRequest, db: Session = Depends(get_db)):
    data = payload.model_dump()
    return DatasetService().split_dataset(db, int(dataset_id), **data)


@router.get("/{dataset_id}/split", response_model=DatasetSplitResultOut)
def get_dataset_split(
    dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    version_id: int | None = None,
    split: str | None = None,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    items, summary, total = DatasetService().get_split_result(
        db,
        int(dataset_id),
        version_id=version_id,
        split=split,
        skip=skip,
        limit=page_size,
    )

    return {"summary": summary, "items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}
