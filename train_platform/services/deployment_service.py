from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.deployment import Deployment, DeploymentLog
from train_platform.models.enums import DeploymentStatus, LogLevel, ModelStage
from train_platform.models.model_registry import ModelVersion
from train_platform.repositories.deployment_repo import DeploymentRepository
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class DeploymentService:
    def __init__(self) -> None:
        self.repo = DeploymentRepository()

    def list_deployments(
        self,
        db: Session,
        *,
        model_version_id: Optional[int] = None,
        status: Optional[DeploymentStatus] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Deployment]:
        return self.repo.list(db, model_version_id=model_version_id, status=status, is_active=is_active, skip=skip, limit=limit)

    def get_deployment(self, db: Session, deployment_id: int) -> Deployment:
        d = self.repo.get(db, int(deployment_id))
        if not d:
            raise NotFoundError("Deployment not found")
        return d

    def create_deployment(self, db: Session, *, obj: dict) -> Deployment:
        model_version_id = int(obj["model_version_id"])
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == model_version_id).first()
        if not mv:
            raise NotFoundError("Model version not found")
        if mv.stage == ModelStage.DEPRECATED:
            raise ConflictError("Cannot deploy a deprecated model version")

        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")

        d = self.repo.create(
            db,
            obj_in={
                "model_version_id": model_version_id,
                "name": name,
                "platform": obj["platform"],
                "status": DeploymentStatus.PENDING,
                "config": obj.get("config"),
                "health_check_url": obj.get("health_check_url"),
                "is_active": True,
            },
        )
        db.add(DeploymentLog(deployment_id=d.deployment_id, level=LogLevel.INFO, message="Deployment created"))
        db.commit()
        db.refresh(d)
        return d

    def update_deployment(self, db: Session, deployment_id: int, *, patch: dict) -> Deployment:
        d = self.get_deployment(db, deployment_id)

        if "name" in patch and patch["name"] is not None:
            d.name = str(patch["name"]).strip()
        if "status" in patch and patch["status"] is not None:
            d.status = patch["status"]
        if "endpoint_url" in patch:
            d.endpoint_url = patch["endpoint_url"]
        if "health_check_url" in patch:
            d.health_check_url = patch["health_check_url"]
        if "config" in patch:
            d.config = patch["config"]
        if "is_active" in patch and patch["is_active"] is not None:
            d.is_active = bool(patch["is_active"])

        db.commit()
        db.refresh(d)
        return d

    def delete_deployment(self, db: Session, deployment_id: int) -> None:
        d = self.get_deployment(db, deployment_id)
        db.delete(d)
        db.commit()

    def add_log(self, db: Session, deployment_id: int, *, level: LogLevel, message: str, data: Optional[dict] = None) -> DeploymentLog:
        self.get_deployment(db, deployment_id)
        row = DeploymentLog(deployment_id=int(deployment_id), level=level, message=str(message), data=data)
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

