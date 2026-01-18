from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.deployment import Deployment, DeploymentLog
from train_platform.models.enums import DeploymentStatus
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.deployments import DeploymentCreate, DeploymentLogCreate, DeploymentLogOut, DeploymentOut, DeploymentUpdate
from train_platform.services.deployment_service import DeploymentService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/deployments", tags=["deployments"])


@router.get("", response_model=Page[DeploymentOut])
def list_deployments(
    page: int = 1,
    page_size: int = 50,
    model_version_id: int | None = Query(None),
    status: str | None = Query(None, description="pending/deploying/active/inactive/failed/deleting"),
    is_active: bool | None = Query(None),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    st = None
    if status:
        try:
            st = DeploymentStatus(str(status))
        except Exception:
            raise ValidationError("Invalid status")

    q = db.query(Deployment)
    if model_version_id is not None:
        q = q.filter(Deployment.model_version_id == int(model_version_id))
    if st is not None:
        q = q.filter(Deployment.status == st)
    if is_active is not None:
        q = q.filter(Deployment.is_active == bool(is_active))
    total = q.count()

    items = DeploymentService().list_deployments(db, model_version_id=model_version_id, status=st, is_active=is_active, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=DeploymentOut, status_code=201)
def create_deployment(payload: DeploymentCreate, db: Session = Depends(get_db)):
    return DeploymentService().create_deployment(db, obj=payload.model_dump())


@router.get("/{deployment_id}", response_model=DeploymentOut)
def get_deployment(deployment_id: int, db: Session = Depends(get_db)):
    return DeploymentService().get_deployment(db, deployment_id)


@router.patch("/{deployment_id}", response_model=DeploymentOut)
def update_deployment(deployment_id: int, payload: DeploymentUpdate, db: Session = Depends(get_db)):
    return DeploymentService().update_deployment(db, deployment_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{deployment_id}", response_model=DeleteResponse)
def delete_deployment(deployment_id: int, db: Session = Depends(get_db)):
    DeploymentService().delete_deployment(db, deployment_id)
    return DeleteResponse(ok=True, message="Deployment deleted")


@router.get("/{deployment_id}/logs", response_model=list[DeploymentLogOut])
def list_deployment_logs(
    deployment_id: int,
    limit: int = Query(200, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    # Query directly; service returns deployment with logs too, but list is common.
    DeploymentService().get_deployment(db, deployment_id)
    return (
        db.query(DeploymentLog)
        .filter(DeploymentLog.deployment_id == int(deployment_id))
        .order_by(DeploymentLog.created_at.desc())
        .limit(int(limit))
        .all()
    )


@router.post("/{deployment_id}/logs", response_model=DeploymentLogOut, status_code=201)
def add_deployment_log(deployment_id: int, payload: DeploymentLogCreate, db: Session = Depends(get_db)):
    return DeploymentService().add_log(db, deployment_id, level=payload.level, message=payload.message, data=payload.data)

