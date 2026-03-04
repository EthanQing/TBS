from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import or_
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
        project_id: Optional[int] = None,
        model_version_id: Optional[int] = None,
        status: Optional[DeploymentStatus] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Deployment]:
        return self.repo.list(
            db,
            project_id=project_id,
            model_version_id=model_version_id,
            status=status,
            is_active=is_active,
            skip=skip,
            limit=limit,
        )

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
                "is_active": False,
            },
        )
        db.add(
            DeploymentLog(
                deployment_id=d.deployment_id,
                level=LogLevel.INFO,
                message="Deployment created (pending execution)",
            )
        )
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
            if d.is_active:
                project_id = self._project_id_of_deployment(db, d)
                self._deactivate_other_deployments_in_project(db, project_id, keep_deployment_id=int(d.deployment_id))
        if d.is_active and d.status == DeploymentStatus.INACTIVE:
            d.status = DeploymentStatus.ACTIVE

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

    def get_rollback_candidates(self, db: Session, deployment_id: int) -> dict:
        d = self.get_deployment(db, deployment_id)
        project_id = self._project_id_of_deployment(db, d)
        current_model_version_id = int(d.model_version_id)

        candidate_ids = self._candidate_model_version_ids(db, project_id, deployment_id=int(d.deployment_id))
        candidate_ids.discard(current_model_version_id)
        if not candidate_ids:
            return {
                "deployment": d,
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
            "deployment": d,
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
        d = self.get_deployment(db, deployment_id)
        project_id = self._project_id_of_deployment(db, d)

        target_model_version_id = int(target_model_version_id)
        if int(d.model_version_id) == target_model_version_id:
            raise ConflictError("Target model version is already deployed")

        target_mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == target_model_version_id).first()
        if not target_mv:
            raise NotFoundError("Target model version not found")
        if int(target_mv.project_id) != int(project_id):
            raise ConflictError("Target model version does not belong to this deployment's project")

        allowed = self._candidate_model_version_ids(db, int(project_id), deployment_id=int(d.deployment_id))
        if target_model_version_id not in allowed:
            raise ConflictError("Target model version has not been successfully deployed in this project")

        current_mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(d.model_version_id)).first()
        if not current_mv:
            raise NotFoundError("Current model version not found")

        clean_reason = str(reason or "").strip()
        if not clean_reason:
            raise ValidationError("reason is required")
        clean_operator = str(operator or "").strip() or "admin"

        self._deactivate_other_deployments_in_project(db, int(project_id), keep_deployment_id=int(d.deployment_id))

        d.model_version_id = target_model_version_id
        d.status = DeploymentStatus.ACTIVE
        d.is_active = True
        d.deployed_at = datetime.now(timezone.utc)

        self._sync_model_stage_after_rollback(db, int(project_id), target_model_version_id)

        log_data = self._build_rollback_log_data(
            project_id=int(project_id),
            from_model_version_id=int(current_mv.model_version_id),
            to_model_version_id=int(target_mv.model_version_id),
            from_version=str(current_mv.version),
            to_version=str(target_mv.version),
            reason=clean_reason,
            operator=clean_operator,
            stage_sync=True,
        )
        log_row = DeploymentLog(
            deployment_id=int(d.deployment_id),
            level=LogLevel.INFO,
            message=f"Rollback deployment to model version {target_mv.version}",
            data=log_data,
        )
        db.add(log_row)
        db.commit()
        db.refresh(d)
        db.refresh(log_row)

        return {
            "deployment": d,
            "event": self._map_rollback_history_row(log_row),
        }

    def list_rollback_history(self, db: Session, deployment_id: int, *, limit: int = 200) -> list[dict]:
        self.get_deployment(db, deployment_id)
        rows = (
            db.query(DeploymentLog)
            .filter(DeploymentLog.deployment_id == int(deployment_id))
            .order_by(DeploymentLog.created_at.desc(), DeploymentLog.log_id.desc())
            .limit(max(1, int(limit)))
            .all()
        )

        out: list[dict] = []
        for row in rows:
            item = self._map_rollback_history_row(row)
            if item is not None:
                out.append(item)
        return out

    def _project_id_of_deployment(self, db: Session, deployment: Deployment) -> int:
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(deployment.model_version_id)).first()
        if not mv:
            raise NotFoundError("Model version not found for deployment")
        return int(mv.project_id)

    def _deactivate_other_deployments_in_project(self, db: Session, project_id: int, *, keep_deployment_id: int) -> None:
        rows = (
            db.query(Deployment)
            .join(ModelVersion, ModelVersion.model_version_id == Deployment.model_version_id)
            .filter(
                ModelVersion.project_id == int(project_id),
                Deployment.deployment_id != int(keep_deployment_id),
                or_(Deployment.is_active == True, Deployment.status == DeploymentStatus.ACTIVE),  # noqa: E712
            )
            .all()
        )
        for row in rows:
            row.is_active = False
            if row.status == DeploymentStatus.ACTIVE:
                row.status = DeploymentStatus.INACTIVE

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
            for k in ("from_model_version_id", "to_model_version_id"):
                try:
                    value = int(data.get(k))
                except Exception:
                    value = None
                if value and value > 0:
                    ids.add(value)

        current_dep = self.get_deployment(db, deployment_id)
        ids.discard(int(current_dep.model_version_id))
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

    def _sync_model_stage_after_rollback(self, db: Session, project_id: int, target_model_version_id: int) -> None:
        target_mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(target_model_version_id)).first()
        if not target_mv:
            raise NotFoundError("Target model version not found")

        db.query(ModelVersion).filter(
            ModelVersion.project_id == int(project_id),
            ModelVersion.model_version_id != int(target_model_version_id),
            ModelVersion.stage == ModelStage.PRODUCTION,
        ).update({ModelVersion.stage: ModelStage.TESTING}, synchronize_session=False)
        target_mv.stage = ModelStage.PRODUCTION

    def _build_rollback_log_data(
        self,
        *,
        project_id: int,
        from_model_version_id: int,
        to_model_version_id: int,
        from_version: str,
        to_version: str,
        reason: str,
        operator: str,
        stage_sync: bool,
    ) -> dict:
        return {
            "action": "rollback",
            "project_id": int(project_id),
            "from_model_version_id": int(from_model_version_id),
            "to_model_version_id": int(to_model_version_id),
            "from_version": str(from_version),
            "to_version": str(to_version),
            "reason": str(reason),
            "operator": str(operator),
            "stage_sync": bool(stage_sync),
            "at": datetime.now(timezone.utc).isoformat(),
        }

    def _map_rollback_history_row(self, row: DeploymentLog) -> Optional[dict]:
        data = row.data if isinstance(row.data, dict) else {}
        if str(data.get("action") or "").strip().lower() != "rollback":
            return None
        return {
            "log_id": int(row.log_id),
            "deployment_id": int(row.deployment_id),
            "created_at": row.created_at,
            "operator": data.get("operator"),
            "reason": data.get("reason"),
            "from_model_version_id": data.get("from_model_version_id"),
            "to_model_version_id": data.get("to_model_version_id"),
            "from_version": data.get("from_version"),
            "to_version": data.get("to_version"),
        }
