from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.qualified_model import QualifiedModel
from train_platform.schemas.v3.common import Page, PageMeta
from train_platform.schemas.v3.qualified_models import (
    QualifiedModelCreate,
    QualifiedModelMarkResponse,
    QualifiedModelOut,
)
from train_platform.services.v3.qualified_model_service import QualifiedModelService


router = APIRouter(prefix="/qualified-models", tags=["qualified-models"])


@router.get("", response_model=Page[QualifiedModelOut])
def list_qualified_models(
    page: int = 1,
    page_size: int = 50,
    project_id: int | None = Query(None),
    standard_dataset_id: int | None = Query(None),
    run_id: str | None = Query(None),
    model_version_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    q = db.query(QualifiedModel)
    if project_id is not None:
        q = q.filter(QualifiedModel.project_id == int(project_id))
    if standard_dataset_id is not None:
        q = q.filter(QualifiedModel.standard_dataset_id == int(standard_dataset_id))
    if run_id:
        q = q.filter(QualifiedModel.run_id == str(run_id))
    if model_version_id is not None:
        q = q.filter(QualifiedModel.model_version_id == int(model_version_id))
    total = q.count()

    items = QualifiedModelService().list_qualified_models(
        db,
        project_id=project_id,
        standard_dataset_id=standard_dataset_id,
        run_id=run_id,
        model_version_id=model_version_id,
        skip=skip,
        limit=page_size,
    )
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=QualifiedModelMarkResponse)
def mark_model_qualified(payload: QualifiedModelCreate, db: Session = Depends(get_db)):
    item, created = QualifiedModelService().mark_model_qualified(
        db,
        model_version_id=payload.model_version_id,
        qualified_by=payload.qualified_by,
        note=payload.note,
    )
    return {
        "created": created,
        "message": "模型已标记为合格" if created else "该模型已标记为合格，无需重复操作",
        "item": item,
    }


@router.get("/{qualified_model_id}", response_model=QualifiedModelOut)
def get_qualified_model(qualified_model_id: int, db: Session = Depends(get_db)):
    return QualifiedModelService().get_qualified_model(db, qualified_model_id)
