from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.core.config import settings
from train_platform.models.dataset_event import DatasetEvent
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.datasets import (
    DatasetCreate,
    DatasetDetailOut,
    DatasetEventOut,
    DatasetFileOut,
    DatasetImageUploadOut,
    DatasetListOut,
    DatasetOut,
    DatasetIllegalConvertRequest,
    DatasetIllegalConvertOut,
    DatasetIllegalLabelsOut,
    DatasetIllegalLabelsUpdate,
    DatasetRenameClassesRequest,
    DatasetRenameClassesOut,
    DatasetSplitRequest,
    DatasetSplitResultOut,
    DatasetSplitSummary,
    DatasetStatisticsOut,
    DatasetUpdate,
    DatasetVersionCreate,
    DatasetVersionDiffOut,
    DatasetVersionOut,
    DatasetViewOut,
)
from train_platform.services.dataset_service import DatasetService
from train_platform.utils.exceptions import ValidationError

# NOTE: v2's root router (train_platform.api.v2) is included under `/api/v2` in the
# FastAPI app. Every sub-router must therefore have a non-empty prefix, otherwise
# path operations like `@router.get("")` would have both an empty prefix and path,
# which FastAPI rejects at startup.
router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    return DatasetService().create_dataset(db, obj=payload.model_dump())


@router.get("", response_model=Page[DatasetListOut])
def list_datasets(
    page: int = 1,
    page_size: int = 50,
    format: str | None = Query(None, description="Filter by format (yolo, coco)"),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    # Count total
    q = db.query(Dataset)
    if format:
        q = q.filter(Dataset.format == str(format))
    total = q.count()

    items = DatasetService().list_datasets_with_stats(db, skip=skip, limit=page_size, format=format)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


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


@router.get("/{dataset_id}/view", response_model=DatasetViewOut)
def get_dataset_view(
    dataset_id: int,
    version_id: int | None = Query(None, description="Version ID (default: active version)"),
    class_id: int | None = Query(None, description="Filter images by class ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Get dataset view with category statistics and paginated image list.
    
    This endpoint returns category information with image counts for the sidebar,
    and a paginated list of images for the main grid. Supports filtering by class_id.
    """
    return DatasetService().get_view(
        db,
        int(dataset_id),
        version_id=version_id,
        class_id=class_id,
        page=page,
        page_size=page_size,
    )


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
    split_enabled: bool = Form(False),
    split_train_ratio: float | None = Form(None),
    split_val_ratio: float | None = Form(None),
    split_test_ratio: float | None = Form(None),
    split_seed: int | None = Form(None),
    split_shuffle: bool | None = Form(None),
    split_overwrite: bool | None = Form(None),
    db: Session = Depends(get_db),
):
    ds, _ver = await DatasetService().upload_dataset_archive_async(
        db,
        int(dataset_id),
        file=file,
        message=message,
        created_by=created_by,
        create_version=bool(create_version),
        activate=bool(activate),
        split_enabled=bool(split_enabled),
        split_train_ratio=split_train_ratio,
        split_val_ratio=split_val_ratio,
        split_test_ratio=split_test_ratio,
        split_seed=split_seed,
        split_shuffle=split_shuffle,
        split_overwrite=split_overwrite,
    )
    return ds


@router.post("/import", response_model=DatasetOut, status_code=201)
@router.post("/upload", response_model=DatasetOut, status_code=201)
async def import_dataset(
    file: UploadFile = File(...),
    dataset_id: int | None = Form(None),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    split_enabled: bool = Form(False),
    split_train_ratio: float | None = Form(None),
    split_val_ratio: float | None = Form(None),
    split_test_ratio: float | None = Form(None),
    split_seed: int | None = Form(None),
    split_shuffle: bool | None = Form(None),
    split_overwrite: bool | None = Form(None),
    db: Session = Depends(get_db),
):
    if dataset_id is None:
        raise ValidationError("dataset_id is required; create the dataset first and upload to /datasets/{id}/upload")

    ds, _ver = await DatasetService().upload_dataset_archive_async(
        db,
        int(dataset_id),
        file=file,
        message=message,
        created_by=created_by,
        create_version=True,
        activate=True,
        split_enabled=bool(split_enabled),
        split_train_ratio=split_train_ratio,
        split_val_ratio=split_val_ratio,
        split_test_ratio=split_test_ratio,
        split_seed=split_seed,
        split_shuffle=split_shuffle,
        split_overwrite=split_overwrite,
    )
    return ds


@router.post("/{dataset_id}/append", response_model=DatasetOut, status_code=201)
async def append_dataset_archive(
    dataset_id: int,
    file: UploadFile = File(...),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    create_version: bool = Form(True),
    activate: bool = Form(True),
    db: Session = Depends(get_db),
):
    """Append ZIP archive contents to an existing (possibly non-empty) dataset."""
    if settings.disable_append_upload:
        raise ValidationError("Append upload disabled")
    ds, _ver = await DatasetService().append_dataset_archive_async(
        db,
        int(dataset_id),
        file=file,
        message=message,
        created_by=created_by,
        create_version=bool(create_version),
        activate=bool(activate),
    )
    return ds


@router.post("/{dataset_id}/convert", response_model=DatasetIllegalConvertOut, status_code=202)
def convert_illegal_dataset(
    dataset_id: int,
    payload: DatasetIllegalConvertRequest,
    db: Session = Depends(get_db),
):
    data = payload.model_dump()
    return DatasetService().convert_illegal_dataset(
        db,
        int(dataset_id),
        label_strategy=data.get("label_strategy"),
        label_level=data.get("label_level"),
        label_separator=data.get("label_separator"),
        label_mapping=data.get("label_mapping"),
        split_enabled=data.get("split_enabled"),
        split_train_ratio=data.get("split_train_ratio"),
        split_val_ratio=data.get("split_val_ratio"),
        split_test_ratio=data.get("split_test_ratio"),
        split_seed=data.get("split_seed"),
        split_shuffle=data.get("split_shuffle"),
        split_overwrite=data.get("split_overwrite"),
    )


@router.get("/{dataset_id}/illegal-labels", response_model=DatasetIllegalLabelsOut)
def get_illegal_dataset_labels(dataset_id: int, db: Session = Depends(get_db)):
    labels = DatasetService().get_illegal_labels(db, dataset_id)
    return {"labels": labels}


@router.put("/{dataset_id}/illegal-labels", response_model=DatasetOut)
def update_illegal_dataset_labels(
    dataset_id: int, payload: DatasetIllegalLabelsUpdate, db: Session = Depends(get_db)
):
    return DatasetService().update_illegal_labels(db, dataset_id, payload.label_mapping)


@router.post("/{dataset_id}/uploads/images", response_model=DatasetImageUploadOut, status_code=201)
async def upload_dataset_images(
    dataset_id: int,
    files: list[UploadFile] | UploadFile | None = File(None),
    images: list[UploadFile] | UploadFile | None = File(None),
    relative_dir: str = Form("images"),
    labels: list[UploadFile] | UploadFile | None = File(None),
    annotations: list[UploadFile] | UploadFile | None = File(None),
    labels_relative_dir: str | None = Form(None),
    require_labels: bool = Form(True),
    message: str | None = Form(None),
    created_by: str | None = Form(None),
    create_version: bool = Form(True),
    create_snapshot: bool = Form(False),
    activate: bool = Form(True),
    db: Session = Depends(get_db),
):
    if settings.disable_append_upload:
        raise ValidationError("Append upload disabled")
    def _as_upload_list(value: list[UploadFile] | UploadFile | None) -> list[UploadFile]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    image_files: list[UploadFile] = []
    image_files.extend(_as_upload_list(files))
    image_files.extend(_as_upload_list(images))

    label_files: list[UploadFile] = []
    label_files.extend(_as_upload_list(labels))
    label_files.extend(_as_upload_list(annotations))

    if not image_files:
        raise ValidationError("images are required")

    return DatasetService().upload_images(
        db,
        int(dataset_id),
        files=image_files,
        relative_dir=relative_dir,
        labels=label_files,
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


@router.put("/{dataset_id}/classes", response_model=DatasetRenameClassesOut)
def rename_dataset_classes(
    dataset_id: int,
    payload: DatasetRenameClassesRequest,
    db: Session = Depends(get_db),
):
    """Rename class labels in a converted YOLO dataset.

    Only updates classes.txt and data.yaml; label files are unchanged
    because class_id (line index) stays the same.
    """
    return DatasetService().rename_classes(db, int(dataset_id), payload.rename_map)


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

# this router will return dataset YAML data
# @router.get("/{dataset_id}/yaml", response_model=DatasetDataYAMLOut)
# def get_dataset_data_yaml(dataset_id: int, db: Session = Depends(get_db)):
#     return DatasetService().get_dataset_data_yaml(db, dataset_id)
