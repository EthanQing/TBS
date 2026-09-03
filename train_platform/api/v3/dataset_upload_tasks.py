from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.schemas.v3.dataset_uploads import DatasetUploadTaskOut
from train_platform.domains.datasets.uploads import DatasetUploadTaskService


router = APIRouter(prefix="/dataset-upload-tasks", tags=["dataset-upload-tasks"])
svc = DatasetUploadTaskService()


@router.get("/{task_id}", response_model=DatasetUploadTaskOut)
def get_dataset_upload_task(task_id: str, db: Session = Depends(get_db)):
    return svc.get_task(db, task_id)
