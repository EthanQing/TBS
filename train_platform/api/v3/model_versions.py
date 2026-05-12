from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.enums import ModelStage
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.schemas.v3.common import Page, PageMeta
from train_platform.schemas.v3.model_versions import ModelVersionCreate, ModelVersionOut, ModelVersionUpdate
from train_platform.services.v3.model_version_service import ModelVersionService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/model-versions", tags=["model-versions"])


@router.get("", response_model=Page[ModelVersionOut])
def list_model_versions(
    page: int = 1,
    page_size: int = 50,
    project_id: int | None = Query(None),
    run_id: str | None = Query(None),
    stage: str | None = Query(None, description="development/testing/production/deprecated"),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    st = None
    if stage:
        try:
            st = ModelStage(str(stage))
        except Exception:
            raise ValidationError("Invalid stage")

    q = db.query(ModelVersion)
    if project_id is not None:
        q = q.filter(ModelVersion.project_id == int(project_id))
    if run_id:
        q = q.filter(ModelVersion.run_id == str(run_id))
    if st is not None:
        q = q.filter(ModelVersion.stage == st)
    total = q.count()

    items = ModelVersionService().list_model_versions(db, project_id=project_id, run_id=run_id, stage=st, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=ModelVersionOut, status_code=201)
def register_model_version(payload: ModelVersionCreate, db: Session = Depends(get_db)):
    return ModelVersionService().register_from_run(
        db,
        run_id=payload.run_id,
        version=payload.version,
        stage=payload.stage,
        description=payload.description,
    )


@router.get("/{model_version_id}", response_model=ModelVersionOut)
def get_model_version(model_version_id: int, db: Session = Depends(get_db)):
    return ModelVersionService().get_model_version(db, model_version_id)


@router.patch("/{model_version_id}", response_model=ModelVersionOut)
def update_model_version(model_version_id: int, payload: ModelVersionUpdate, db: Session = Depends(get_db)):
    return ModelVersionService().update_model_version(db, model_version_id, patch=payload.model_dump(exclude_unset=True))

