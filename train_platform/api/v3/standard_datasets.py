from __future__ import annotations

import mimetypes

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetEvent
from train_platform.schemas.v3.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v3.standard_datasets import (
    DatasetFileOut,
    DatasetImageAnnotationsOut,
    DatasetSplitRequest,
    DatasetSplitResultOut,
    DatasetSplitSummary,
    DatasetStatisticsOut,
    DatasetViewOut,
    StandardDatasetCreate,
    StandardDatasetDetailOut,
    StandardDatasetEventOut,
    StandardDatasetListOut,
    StandardDatasetOut,
    StandardDatasetUpdate,
)
from train_platform.schemas.v3.dataset_uploads import (
    DatasetImportFromPathRequest,
    DatasetUploadCompleteOut,
    DatasetUploadPartOut,
    DatasetUploadSessionCreate,
    DatasetUploadSessionOut,
)
from train_platform.services.v3.dataset_upload_service import DatasetUploadService
from train_platform.services.v3.standard_dataset_service import StandardDatasetService


router = APIRouter(prefix="/standard-datasets", tags=["standard-datasets"])
svc = StandardDatasetService()
upload_svc = DatasetUploadService()


@router.post("", response_model=StandardDatasetOut, status_code=201)
def create_standard_dataset(payload: StandardDatasetCreate, db: Session = Depends(get_db)):
    return svc.create_dataset(db, obj=payload.model_dump())


@router.get("", response_model=Page[StandardDatasetListOut])
def list_standard_datasets(
    page: int = 1,
    page_size: int = 50,
    format: str | None = Query(None),
    include_statistics: bool = Query(
        True,
        description="Set false for lightweight reference lists that do not need statistics or preview images.",
    ),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    q = db.query(StandardDataset)
    if format:
        q = q.filter(StandardDataset.format == str(format))
    total = q.count()
    items = svc.list_datasets(
        db,
        skip=skip,
        limit=page_size,
        format=format,
        include_statistics=bool(include_statistics),
    )
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


@router.post("/{standard_dataset_id}/upload-sessions", response_model=DatasetUploadSessionOut, status_code=201)
def create_standard_upload_session(
    standard_dataset_id: int,
    payload: DatasetUploadSessionCreate,
    db: Session = Depends(get_db),
):
    return upload_svc.create_session(
        db,
        "standard",
        standard_dataset_id,
        filename=payload.filename,
        total_size=payload.total_size,
        chunk_size=payload.chunk_size,
        mode=payload.mode,
        created_by=payload.created_by,
    )


@router.put("/{standard_dataset_id}/upload-sessions/{session_id}/parts/{part_no}", response_model=DatasetUploadPartOut)
def upload_standard_session_part(
    standard_dataset_id: int,
    session_id: str,
    part_no: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return upload_svc.save_part(db, "standard", standard_dataset_id, session_id, part_no, file)


@router.get("/{standard_dataset_id}/upload-sessions/{session_id}", response_model=DatasetUploadSessionOut)
def get_standard_upload_session(standard_dataset_id: int, session_id: str, db: Session = Depends(get_db)):
    return upload_svc.get_session(db, "standard", standard_dataset_id, session_id)


@router.post("/{standard_dataset_id}/upload-sessions/{session_id}/complete", response_model=DatasetUploadCompleteOut, status_code=202)
def complete_standard_upload_session(
    standard_dataset_id: int,
    session_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    task = upload_svc.complete_session(db, "standard", standard_dataset_id, session_id)
    background_tasks.add_task(upload_svc.run_task, task.task_id)
    return {"task_id": task.task_id, "session_id": session_id, "status": task.status}


@router.delete("/{standard_dataset_id}/upload-sessions/{session_id}", response_model=DatasetUploadSessionOut)
def cancel_standard_upload_session(standard_dataset_id: int, session_id: str, db: Session = Depends(get_db)):
    return upload_svc.cancel_session(db, "standard", standard_dataset_id, session_id)


@router.post("/{standard_dataset_id}/import-from-path", response_model=DatasetUploadCompleteOut, status_code=202)
def import_standard_dataset_from_path(
    standard_dataset_id: int,
    payload: DatasetImportFromPathRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    task = upload_svc.create_import_task(
        db,
        "standard",
        standard_dataset_id,
        root_id=payload.root_id,
        rel_path=payload.path,
        mode=payload.mode,
        storage_strategy=payload.storage_strategy,
        created_by=payload.created_by,
        message=payload.message,
    )
    background_tasks.add_task(upload_svc.run_task, task.task_id)
    return {"task_id": task.task_id, "session_id": None, "status": task.status}


@router.post("/{standard_dataset_id}/split", response_model=DatasetSplitSummary)
def split_standard_dataset(
    standard_dataset_id: int,
    payload: DatasetSplitRequest,
    db: Session = Depends(get_db),
):
    return svc.split_dataset(db, standard_dataset_id, **payload.model_dump())


@router.get("/{standard_dataset_id}/split", response_model=DatasetSplitResultOut)
def get_standard_dataset_split(
    standard_dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    split: str | None = Query(None),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size
    items, summary, total = svc.get_split_result(
        db,
        standard_dataset_id,
        split=split,
        skip=skip,
        limit=page_size,
    )
    return {"summary": summary, "items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


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


@router.get("/{standard_dataset_id}/file/{file_path:path}")
def get_standard_dataset_file(
    standard_dataset_id: int,
    file_path: str,
    db: Session = Depends(get_db),
):
    path = svc.get_file_path(db, standard_dataset_id, file_path)
    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return FileResponse(path=str(path), media_type=media_type)


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
