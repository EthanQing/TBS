from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session, joinedload

from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.enums import DeploymentStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.repositories.v3.base import BaseRepository


class DeploymentRepository(BaseRepository[Deployment]):
    def __init__(self) -> None:
        super().__init__(Deployment)

    def get(self, db: Session, deployment_id: int) -> Optional[Deployment]:
        return (
            db.query(Deployment)
            .options(joinedload(Deployment.logs))
            .filter(Deployment.deployment_id == int(deployment_id))
            .first()
        )

    def list(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        status: Optional[DeploymentStatus] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Deployment]:
        q = db.query(Deployment)
        if project_id is not None:
            q = q.join(ModelVersion, ModelVersion.model_version_id == Deployment.model_version_id)
            q = q.filter(ModelVersion.project_id == int(project_id))
        if model_version_id is not None:
            q = q.filter(Deployment.model_version_id == int(model_version_id))
        if status is not None:
            q = q.filter(Deployment.status == status)
        if is_active is not None:
            q = q.filter(Deployment.is_active == bool(is_active))
        return q.order_by(Deployment.updated_at.desc()).offset(skip).limit(limit).all()
