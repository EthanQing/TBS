from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.enums import TaskType
from train_platform.schemas.v3.architectures import ArchitectureCreate, ArchitectureOut
from train_platform.services.v3.architecture_service import ArchitectureService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/architectures", tags=["architectures"])


@router.get("", response_model=list[ArchitectureOut])
def list_architectures(
    family: str | None = Query(None),
    task_type: str | None = Query(None, description="detection/segmentation/classification"),
    q: str | None = Query(None),
    db: Session = Depends(get_db),
):
    tt = None
    if task_type:
        try:
            tt = TaskType(str(task_type))
        except Exception:
            raise ValidationError("Invalid task_type")
    return ArchitectureService().list_architectures(db, family=family, task_type=tt, q=q)


@router.post("", response_model=ArchitectureOut, status_code=201)
def create_architecture(payload: ArchitectureCreate, db: Session = Depends(get_db)):
    return ArchitectureService().create_architecture(db, obj=payload.model_dump())

