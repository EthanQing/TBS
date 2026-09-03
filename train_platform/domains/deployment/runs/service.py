from __future__ import annotations

import threading
import uuid
from typing import Any

from sqlalchemy.orm import Session

from train_platform.domains.deployment import activation
from train_platform.domains.deployment.credentials import generate_api_key
from train_platform.domains.deployment.logs import append_run_log, list_run_logs
from train_platform.domains.deployment.runs import lifecycle
from train_platform.domains.deployment.runs.pipeline import execute_pipeline
from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.models.v3.enums import DeploymentRunStatus, LogLevel
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.project import Project
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


_THREADS_LOCK = threading.Lock()
_RUN_THREADS: dict[str, threading.Thread] = {}


class DeploymentRunService:
    """Public deployment-run operations and process-local execution dispatch."""

    ACTIVE_STATUSES = {DeploymentRunStatus.QUEUED, DeploymentRunStatus.RUNNING}
    TERMINAL_STATUSES = {
        DeploymentRunStatus.COMPLETED,
        DeploymentRunStatus.FAILED,
        DeploymentRunStatus.CANCELLED,
    }
    _THREADS_LOCK = _THREADS_LOCK
    _RUN_THREADS = _RUN_THREADS

    def get_run(self, db: Session, run_id: str) -> DeploymentRun:
        rid = str(run_id or "").strip()
        if not rid:
            raise ValidationError("run_id is required")
        row = db.query(DeploymentRun).filter(DeploymentRun.run_id == rid).first()
        if not row:
            raise NotFoundError("Deployment run not found")
        return row

    def execute_deployment(self, db: Session, deployment_id: int, *, payload: dict[str, Any]) -> dict[str, Any]:
        deployment = (
            db.query(Deployment)
            .filter(Deployment.deployment_id == int(deployment_id))
            .first()
        )
        if not deployment:
            raise NotFoundError("Deployment not found")
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == int(deployment.model_version_id))
            .first()
        )
        if not model_version:
            raise NotFoundError("Model version not found")

        # The project row is the admission lock.  DeploymentRun status queries
        # are made only after taking it, so concurrent requests serialize.
        project = (
            db.query(Project)
            .filter(Project.project_id == int(model_version.project_id))
            .with_for_update()
            .first()
        )
        if not project:
            raise NotFoundError("Project not found")
        active_run = (
            db.query(DeploymentRun)
            .filter(
                DeploymentRun.project_id == int(project.project_id),
                DeploymentRun.status.in_(list(self.ACTIVE_STATUSES)),
            )
            .order_by(DeploymentRun.created_at.desc())
            .first()
        )
        if active_run:
            raise ConflictError(
                f"Another deployment run is active (run_id={active_run.run_id}, status={active_run.status.value})"
            )

        rotate_api_key = bool(payload.get("rotate_api_key", True))
        issued_key = None
        pending_hash = None
        api_key_hint = str(deployment.api_key_hint or "").strip() or None
        if rotate_api_key or not str(deployment.api_key_hash or "").strip():
            issued_key, pending_hash, api_key_hint = generate_api_key()

        run_id = str(uuid.uuid4())
        snapshot = {
            "steps": lifecycle.snapshot_steps(),
            "operator": str(payload.get("operator") or "admin"),
            "reason": str(payload.get("reason") or "").strip() or None,
            "defaults": {
                "conf": float(payload.get("conf", 0.25)),
                "iou": float(payload.get("iou", 0.45)),
            },
            "api_key_hint": api_key_hint,
            "pending_api_key_hash": pending_hash,
            "last_seq": 0,
        }
        activation.prepare_deployment_for_run(db, deployment_id=int(deployment.deployment_id))
        row = lifecycle.new_queued_run(
            run_id=run_id,
            deployment_id=int(deployment.deployment_id),
            project_id=int(project.project_id),
            model_version_id=int(deployment.model_version_id),
            snapshot=snapshot,
        )
        db.add(row)
        db.flush()
        append_run_log(
            db,
            run_id,
            level=LogLevel.INFO,
            message="Deployment run queued",
            action="queued",
            detail={"deployment_id": int(deployment.deployment_id)},
        )
        db.commit()
        db.refresh(row)
        self._start_pipeline_thread(run_id)
        return {"run": row, "issued_api_key": issued_key, "api_key_hint": api_key_hint}

    def retry_run(self, db: Session, run_id: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        previous = self.get_run(db, run_id)
        if previous.status not in {DeploymentRunStatus.FAILED, DeploymentRunStatus.CANCELLED}:
            raise ConflictError("Only failed/cancelled deployment runs can be retried")
        return self.execute_deployment(db, int(previous.deployment_id), payload=payload)

    def cancel_run(self, db: Session, run_id: str) -> DeploymentRun:
        run = lifecycle.request_cancel(db, run_id)
        db.commit()
        db.refresh(run)
        return run

    def list_logs_since(
        self,
        db: Session,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return list_run_logs(db, run_id, after_seq=after_seq, limit=limit)

    def _start_pipeline_thread(self, run_id: str) -> None:
        def target() -> None:
            try:
                execute_pipeline(run_id)
            finally:
                with _THREADS_LOCK:
                    _RUN_THREADS.pop(run_id, None)

        with _THREADS_LOCK:
            existing = _RUN_THREADS.get(run_id)
            if existing and existing.is_alive():
                return
            thread = threading.Thread(
                target=target,
                name=f"deployment-run-{str(run_id)[:8]}",
                daemon=True,
            )
            _RUN_THREADS[run_id] = thread
            thread.start()

    def recover_orphaned_runs(self) -> dict[str, int]:
        """Repair runs left by process interruption and reschedule queued work."""
        queued_ids: list[str] = []
        failed_count = 0
        with self._recovery_session() as db:
            running_ids = [
                str(row.run_id)
                for row in db.query(DeploymentRun.run_id)
                .filter(DeploymentRun.status == DeploymentRunStatus.RUNNING)
                .all()
            ]
            for run_id in running_ids:
                lifecycle.mark_failed(
                    db,
                    run_id,
                    error="Deployment run interrupted by process restart",
                )
                failed_count += 1
            queued_ids = [
                str(row.run_id)
                for row in db.query(DeploymentRun.run_id)
                .filter(DeploymentRun.status == DeploymentRunStatus.QUEUED)
                .all()
            ]

        for run_id in queued_ids:
            self._start_pipeline_thread(run_id)
        return {"failed": failed_count, "rescheduled": len(queued_ids)}

    @staticmethod
    def _recovery_session():
        from train_platform.db.session import session_scope

        return session_scope()


__all__ = ["DeploymentRunService"]
