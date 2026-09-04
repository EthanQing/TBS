from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from train_platform.core.license import assert_valid_license
from train_platform.domains.deployment import activation
from train_platform.domains.deployment.logs import (
    append_deployment_log,
    list_deployment_logs,
    map_rollback_log,
    rollback_log_data,
)
from train_platform.models.v3.deployment import Deployment, DeploymentLog
from train_platform.models.v3.enums import DeploymentStatus, LogLevel, ModelStage
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class DeploymentService:
    """Application operations for the Deployment aggregate."""

    def _deployment_query(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        status: Optional[DeploymentStatus] = None,
        is_active: Optional[bool] = None,
    ):
        query = db.query(Deployment)
        if project_id is not None:
            query = query.join(ModelVersion, ModelVersion.model_version_id == Deployment.model_version_id)
            query = query.filter(ModelVersion.project_id == int(project_id))
        if model_version_id is not None:
            query = query.filter(Deployment.model_version_id == int(model_version_id))
        if status is not None:
            query = query.filter(Deployment.status == status)
        if is_active is not None:
            query = query.filter(Deployment.is_active == bool(is_active))
        return query

    def list_deployments_page(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        status: Optional[DeploymentStatus] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[Deployment], int]:
        query = self._deployment_query(
            db,
            project_id=project_id,
            model_version_id=model_version_id,
            status=status,
            is_active=is_active,
        )
        total = int(query.count())
        items = (
            query
            .order_by(Deployment.updated_at.desc())
            .offset(max(0, int(skip)))
            .limit(max(0, int(limit)))
            .all()
        )
        return items, total

    def get_deployment(self, db: Session, deployment_id: int) -> Deployment:
        row = (
            db.query(Deployment)
            .options(joinedload(Deployment.logs))
            .filter(Deployment.deployment_id == int(deployment_id))
            .first()
        )
        if not row:
            raise NotFoundError("Deployment not found")
        return row

    def create_deployment(self, db: Session, *, obj: dict) -> Deployment:
        assert_valid_license()
        model_version_id = int(obj["model_version_id"])
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == model_version_id)
            .first()
        )
        if not model_version:
            raise NotFoundError("Model version not found")
        if model_version.stage == ModelStage.DEPRECATED:
            raise ConflictError("Cannot deploy a deprecated model version")

        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        row = Deployment(
            model_version_id=model_version_id,
            name=name,
            platform=obj["platform"],
            status=DeploymentStatus.PENDING,
            config=obj.get("config"),
            health_check_url=obj.get("health_check_url"),
            is_active=False,
        )
        db.add(row)
        db.flush()
        db.refresh(row)
        append_deployment_log(
            db,
            int(row.deployment_id),
            level=LogLevel.INFO,
            message="Deployment created (pending execution)",
        )
        db.commit()
        db.refresh(row)
        return row

    def update_deployment(self, db: Session, deployment_id: int, *, patch: dict) -> Deployment:
        row = self.get_deployment(db, deployment_id)

        if "name" in patch and patch["name"] is not None:
            name = str(patch["name"]).strip()
            if not name:
                raise ValidationError("name cannot be empty")
            row.name = name
        if "endpoint_url" in patch:
            row.endpoint_url = patch["endpoint_url"]
        if "health_check_url" in patch:
            row.health_check_url = patch["health_check_url"]
        if "config" in patch:
            row.config = patch["config"]

        lifecycle_fields = {"status", "is_active"} & set(patch)
        if lifecycle_fields:
            activation.patch_lifecycle(
                db,
                deployment_id=int(row.deployment_id),
                status=patch.get("status"),
                is_active=patch.get("is_active"),
                status_provided="status" in patch,
                is_active_provided="is_active" in patch,
            )

        db.commit()
        db.refresh(row)
        return row

    def delete_deployment(self, db: Session, deployment_id: int) -> None:
        row = self.get_deployment(db, deployment_id)
        db.delete(row)
        db.commit()

    def add_log(
        self,
        db: Session,
        deployment_id: int,
        *,
        level: LogLevel,
        message: str,
        data: Optional[dict] = None,
    ) -> DeploymentLog:
        row = append_deployment_log(
            db,
            int(deployment_id),
            level=level,
            message=message,
            data=data,
        )
        db.commit()
        db.refresh(row)
        return row

    def get_logs(self, db: Session, deployment_id: int, *, limit: int = 200) -> list[DeploymentLog]:
        return list_deployment_logs(db, deployment_id, limit=limit)

    def get_rollback_candidates(self, db: Session, deployment_id: int) -> dict:
        deployment = self.get_deployment(db, deployment_id)
        project_id = self._project_id_of_deployment(db, deployment)
        current_model_version_id = int(deployment.model_version_id)
        candidate_ids = self._candidate_model_version_ids(
            db,
            project_id,
            deployment_id=int(deployment.deployment_id),
        )
        candidate_ids.discard(current_model_version_id)
        if not candidate_ids:
            return {
                "deployment": deployment,
                "current_model_version_id": current_model_version_id,
                "candidates": [],
            }
        candidates = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.project_id == int(project_id),
                ModelVersion.model_version_id.in_(list(candidate_ids)),
            )
            .order_by(ModelVersion.updated_at.desc(), ModelVersion.model_version_id.desc())
            .all()
        )
        return {
            "deployment": deployment,
            "current_model_version_id": current_model_version_id,
            "candidates": candidates,
        }

    def rollback_deployment(
        self,
        db: Session,
        deployment_id: int,
        *,
        target_model_version_id: int,
        reason: str,
        operator: str,
    ) -> dict:
        deployment = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).first()
        if not deployment:
            raise NotFoundError("Deployment not found")
        current_model = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(deployment.model_version_id)).first()
        if not current_model:
            raise NotFoundError("Model version not found for deployment")

        clean_reason = str(reason or "").strip()
        if not clean_reason:
            raise ValidationError("reason is required")
        clean_operator = str(operator or "").strip() or "admin"
        target_id = int(target_model_version_id)
        if int(current_model.model_version_id) == target_id:
            raise ConflictError("Target model version is already deployed")

        target = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == target_id)
            .first()
        )
        if not target:
            raise NotFoundError("Target model version not found")
        if int(target.project_id) != int(current_model.project_id):
            raise ConflictError("Target model version does not belong to this deployment's project")
        allowed = self._candidate_model_version_ids(
            db,
            int(current_model.project_id),
            deployment_id=int(deployment.deployment_id),
        )
        if target_id not in allowed:
            raise ConflictError("Target model version has not been successfully deployed in this project")

        activation_result = activation.activate_deployment(
            db,
            deployment_id=int(deployment.deployment_id),
            model_version_id=target_id,
        )
        activated = activation_result.deployment
        actual_from_id = int(activation_result.previous_model_version_id)
        if actual_from_id == target_id:
            raise ConflictError("Target model version is already deployed")
        actual_current = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == actual_from_id)
            .first()
        )
        if not actual_current:
            raise NotFoundError("Current model version not found")
        target = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == target_id)
            .first()
        )
        if not target:
            raise NotFoundError("Target model version not found")
        log_row = append_deployment_log(
            db,
            int(activated.deployment_id),
            level=LogLevel.INFO,
            message=f"Rollback deployment to model version {target.version}",
            data=rollback_log_data(
                project_id=int(actual_current.project_id),
                from_model_version_id=actual_from_id,
                to_model_version_id=int(target.model_version_id),
                from_version=str(actual_current.version),
                to_version=str(target.version),
                reason=clean_reason,
                operator=clean_operator,
                at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()
        db.refresh(activated)
        db.refresh(log_row)
        return {"deployment": activated, "event": map_rollback_log(log_row)}

    def list_rollback_history(self, db: Session, deployment_id: int, *, limit: int = 200) -> list[dict]:
        self.get_deployment(db, deployment_id)
        rows = (
            db.query(DeploymentLog)
            .filter(DeploymentLog.deployment_id == int(deployment_id))
            .order_by(DeploymentLog.created_at.desc(), DeploymentLog.log_id.desc())
            .limit(max(1, int(limit)))
            .all()
        )
        return [item for row in rows if (item := map_rollback_log(row)) is not None]

    def _project_id_of_deployment(self, db: Session, deployment: Deployment) -> int:
        model = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == int(deployment.model_version_id))
            .first()
        )
        if not model:
            raise NotFoundError("Model version not found for deployment")
        return int(model.project_id)

    def _candidate_model_version_ids(self, db: Session, project_id: int, *, deployment_id: int) -> set[int]:
        status_hits = (
            db.query(Deployment.model_version_id)
            .join(ModelVersion, ModelVersion.model_version_id == Deployment.model_version_id)
            .filter(
                ModelVersion.project_id == int(project_id),
                Deployment.status.in_([DeploymentStatus.ACTIVE, DeploymentStatus.INACTIVE]),
            )
            .distinct()
            .all()
        )
        ids = {int(row[0]) for row in status_hits if row and row[0] is not None}

        project_logs = (
            db.query(DeploymentLog)
            .join(Deployment, Deployment.deployment_id == DeploymentLog.deployment_id)
            .join(ModelVersion, ModelVersion.model_version_id == Deployment.model_version_id)
            .filter(ModelVersion.project_id == int(project_id))
            .all()
        )
        for log_row in project_logs:
            data = log_row.data if isinstance(log_row.data, dict) else {}
            if str(data.get("action") or "").strip().lower() != "rollback":
                continue
            for key in ("from_model_version_id", "to_model_version_id"):
                try:
                    value = int(data.get(key))
                except (TypeError, ValueError):
                    value = None
                if value and value > 0:
                    ids.add(value)

        current = self.get_deployment(db, deployment_id)
        ids.discard(int(current.model_version_id))
        if not ids:
            return set()
        valid = (
            db.query(ModelVersion.model_version_id)
            .filter(
                ModelVersion.project_id == int(project_id),
                ModelVersion.model_version_id.in_(list(ids)),
            )
            .all()
        )
        return {int(row[0]) for row in valid if row and row[0] is not None}


def get_serving_deployment(db: Session, deployment_id: int) -> Deployment:
    row = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).first()
    if not row:
        raise NotFoundError("Deployment not found")
    return row


def assert_serving_ready(deployment: Deployment) -> None:
    if not bool(deployment.is_active) or deployment.status != DeploymentStatus.ACTIVE:
        raise ConflictError("Deployment is not active")
    if not str(deployment.api_key_hash or "").strip():
        raise ConflictError("Deployment API key is not configured")


__all__ = ["DeploymentService", "assert_serving_ready", "get_serving_deployment"]
