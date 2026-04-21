from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.v3.deployment import Deployment, DeploymentLog
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.models.v3.enums import (
    DeploymentRunPhase,
    DeploymentRunStatus,
    DeploymentStatus,
    DeploymentTriggerType,
    LogLevel,
    ModelStage,
)
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.services.v3.deployment_adapters import DeploymentAdapterContext, get_deployment_adapter
from train_platform.services.v3.inference_service import InferenceService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _snapshot_steps() -> list[dict[str, Any]]:
    return [
        {"key": "validate_artifacts", "name": "Validate Artifacts", "status": "pending"},
        {"key": "materialize_runtime", "name": "Materialize Runtime", "status": "pending"},
        {"key": "smoke_test", "name": "Smoke Test", "status": "pending"},
        {"key": "activate", "name": "Activate", "status": "pending"},
    ]


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(str(raw_key).encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, key_hash: str) -> bool:
    candidate = hash_api_key(raw_key)
    return hmac.compare_digest(candidate, str(key_hash or ""))


def generate_api_key() -> tuple[str, str, str]:
    key = f"dpk_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(key)
    hint = f"{key[:6]}...{key[-4:]}" if len(key) >= 12 else key
    return key, key_hash, hint


class DeploymentRuntimeService:
    ACTIVE_STATUSES = {DeploymentRunStatus.QUEUED, DeploymentRunStatus.RUNNING}
    TERMINAL_STATUSES = {
        DeploymentRunStatus.COMPLETED,
        DeploymentRunStatus.FAILED,
        DeploymentRunStatus.CANCELLED,
    }
    _THREADS_LOCK = threading.Lock()
    _RUN_THREADS: dict[str, threading.Thread] = {}

    def __init__(self) -> None:
        self._infer = InferenceService()

    def get_run(self, db: Session, run_id: str) -> DeploymentRun:
        rid = str(run_id or "").strip()
        if not rid:
            raise ValidationError("run_id is required")
        row = db.query(DeploymentRun).filter(DeploymentRun.run_id == rid).first()
        if not row:
            raise NotFoundError("Deployment run not found")
        return row

    def execute_deployment(self, db: Session, deployment_id: int, *, payload: dict[str, Any]) -> dict[str, Any]:
        deployment = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).first()
        if not deployment:
            raise NotFoundError("Deployment not found")

        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == int(deployment.model_version_id))
            .first()
        )
        if not model_version:
            raise NotFoundError("Model version not found")

        project_id = int(model_version.project_id)
        self._ensure_no_active_project_run(db, project_id=project_id)

        rotate_api_key = bool(payload.get("rotate_api_key", True))
        issued_key = None
        pending_hash = None
        api_key_hint = None
        if rotate_api_key or not str(deployment.api_key_hash or "").strip():
            issued_key, pending_hash, api_key_hint = generate_api_key()
        else:
            api_key_hint = str(deployment.api_key_hint or "").strip() or None

        run_id = str(uuid.uuid4())
        snapshot = {
            "steps": _snapshot_steps(),
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
        row = DeploymentRun(
            run_id=run_id,
            deployment_id=int(deployment.deployment_id),
            project_id=int(project_id),
            model_version_id=int(deployment.model_version_id),
            trigger_type=DeploymentTriggerType.MANUAL,
            status=DeploymentRunStatus.QUEUED,
            phase=DeploymentRunPhase.PREPARING,
            current_step=None,
            progress=0,
            cancel_requested=False,
            snapshot=snapshot,
        )
        deployment.status = DeploymentStatus.DEPLOYING
        db.add(row)
        self._append_log(
            db,
            run=row,
            level=LogLevel.INFO,
            message="Deployment run queued",
            step_key=None,
            action="queued",
            detail={"deployment_id": int(deployment.deployment_id)},
        )
        db.commit()
        db.refresh(row)

        self._start_pipeline_thread(run_id)
        return {"run": row, "issued_api_key": issued_key, "api_key_hint": api_key_hint}

    def retry_run(self, db: Session, run_id: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        prev = self.get_run(db, run_id)
        if prev.status not in {DeploymentRunStatus.FAILED, DeploymentRunStatus.CANCELLED}:
            raise ConflictError("Only failed/cancelled deployment runs can be retried")
        return self.execute_deployment(db, int(prev.deployment_id), payload=payload)

    def cancel_run(self, db: Session, run_id: str) -> DeploymentRun:
        run = self.get_run(db, run_id)
        if run.status in self.TERMINAL_STATUSES:
            return run

        run.cancel_requested = True
        if run.status == DeploymentRunStatus.QUEUED:
            run.status = DeploymentRunStatus.CANCELLED
            run.phase = DeploymentRunPhase.CANCELLED
            run.finished_at = _utcnow()
            run.error_message = "Cancelled before execution"
            run.progress = 0
            self._set_step_status(run, key=str(run.current_step or ""), status="cancelled", detail="cancel requested")

        self._append_log(
            db,
            run=run,
            level=LogLevel.WARNING,
            message="Cancellation requested",
            step_key=run.current_step,
            action="cancel_requested",
            detail=None,
        )
        db.commit()
        db.refresh(run)
        return run

    def list_logs_since(self, db: Session, run_id: str, *, after_seq: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        run = self.get_run(db, run_id)
        rows = (
            db.query(DeploymentLog)
            .filter(DeploymentLog.deployment_id == int(run.deployment_id))
            .order_by(DeploymentLog.log_id.asc())
            .limit(max(1, min(int(limit), 5000)))
            .all()
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            data = row.data if isinstance(row.data, dict) else {}
            if str(data.get("run_id") or "") != str(run.run_id):
                continue
            seq = int(data.get("seq") or 0)
            if seq <= int(after_seq):
                continue
            out.append(
                {
                    "seq": seq,
                    "log_id": int(row.log_id),
                    "level": str(row.level.value if hasattr(row.level, "value") else row.level),
                    "message": row.message,
                    "created_at": row.created_at,
                    "step_key": data.get("step_key"),
                    "action": data.get("action"),
                    "detail": data.get("detail"),
                }
            )
        return out

    def _start_pipeline_thread(self, run_id: str) -> None:
        def _target() -> None:
            try:
                self._run_pipeline(run_id)
            finally:
                with self._THREADS_LOCK:
                    self._RUN_THREADS.pop(run_id, None)

        with self._THREADS_LOCK:
            existing = self._RUN_THREADS.get(run_id)
            if existing and existing.is_alive():
                return
            t = threading.Thread(target=_target, name=f"deployment-run-{run_id[:8]}", daemon=True)
            self._RUN_THREADS[run_id] = t
            t.start()

    def _run_pipeline(self, run_id: str) -> None:
        with SessionLocal() as db:
            run = db.query(DeploymentRun).filter(DeploymentRun.run_id == str(run_id)).first()
            if not run:
                return
            if run.status not in {DeploymentRunStatus.QUEUED, DeploymentRunStatus.RUNNING}:
                return

            deployment = db.query(Deployment).filter(Deployment.deployment_id == int(run.deployment_id)).first()
            if not deployment:
                run.status = DeploymentRunStatus.FAILED
                run.error_message = "Deployment not found"
                run.finished_at = _utcnow()
                db.commit()
                return

            run.status = DeploymentRunStatus.RUNNING
            run.phase = DeploymentRunPhase.PREPARING
            run.started_at = run.started_at or _utcnow()
            run.current_step = None
            run.progress = 1
            deployment.status = DeploymentStatus.DEPLOYING
            db.commit()

            try:
                model_context = self._step_validate_artifacts(db, run, deployment)
                if self._check_cancelled(db, run):
                    return

                adapter = get_deployment_adapter(deployment.platform)
                ctx = DeploymentAdapterContext(
                    deployment=deployment,
                    run_id=str(run.run_id),
                    model_context=model_context,
                    defaults=self._get_defaults(run),
                )
                self._step_materialize_runtime(db, run, deployment, ctx, adapter_output=adapter.prepare(ctx))
                if self._check_cancelled(db, run):
                    return

                self._step_smoke_test(db, run, deployment, model_context)
                if self._check_cancelled(db, run):
                    return

                self._step_activate(db, run, deployment, ctx, adapter_output=adapter.activate(ctx))
            except Exception as e:
                run = db.query(DeploymentRun).filter(DeploymentRun.run_id == str(run_id)).first()
                if not run:
                    return
                deployment = db.query(Deployment).filter(Deployment.deployment_id == int(run.deployment_id)).first()
                run.status = DeploymentRunStatus.FAILED
                run.error_message = f"{type(e).__name__}: {e}"
                run.finished_at = _utcnow()
                self._set_step_status(run, key=str(run.current_step or ""), status="failed", detail=str(e))
                if deployment and deployment.status == DeploymentStatus.DEPLOYING:
                    deployment.status = DeploymentStatus.FAILED
                self._append_log(
                    db,
                    run=run,
                    level=LogLevel.ERROR,
                    message=f"Deployment run failed: {type(e).__name__}: {e}",
                    step_key=run.current_step,
                    action="failed",
                    detail={"error": str(e)},
                )
                db.commit()

    def _ensure_no_active_project_run(self, db: Session, *, project_id: int) -> None:
        row = (
            db.query(DeploymentRun)
            .filter(
                DeploymentRun.project_id == int(project_id),
                DeploymentRun.status.in_(list(self.ACTIVE_STATUSES)),
            )
            .order_by(DeploymentRun.created_at.desc())
            .first()
        )
        if row:
            raise ConflictError(f"Another deployment run is active (run_id={row.run_id}, status={row.status.value})")

    def _get_defaults(self, run: DeploymentRun) -> dict[str, Any]:
        snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
        defaults = snapshot.get("defaults") if isinstance(snapshot.get("defaults"), dict) else {}
        return {
            "conf": float(defaults.get("conf", 0.25)),
            "iou": float(defaults.get("iou", 0.45)),
        }

    def _step_validate_artifacts(self, db: Session, run: DeploymentRun, deployment: Deployment) -> Dict[str, Any]:
        run.phase = DeploymentRunPhase.VALIDATE_ARTIFACTS
        run.current_step = "validate_artifacts"
        run.progress = 10
        self._set_step_status(run, key="validate_artifacts", status="running")
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Validating model artifacts",
            step_key="validate_artifacts",
            action="start",
            detail=None,
        )
        db.commit()

        model_context = self._infer.resolve_model_context(db, model_version_id=int(run.model_version_id))
        self._set_step_status(run, key="validate_artifacts", status="completed")
        run.progress = 25
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Artifacts validated",
            step_key="validate_artifacts",
            action="completed",
            detail={"engine": model_context.get("engine"), "weights_path": model_context.get("weights_path")},
        )
        db.commit()
        return model_context

    def _step_materialize_runtime(
        self,
        db: Session,
        run: DeploymentRun,
        deployment: Deployment,
        ctx: DeploymentAdapterContext,
        *,
        adapter_output: dict[str, Any],
    ) -> None:
        run.phase = DeploymentRunPhase.MATERIALIZE_RUNTIME
        run.current_step = "materialize_runtime"
        run.progress = 35
        self._set_step_status(run, key="materialize_runtime", status="running")
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Materializing runtime",
            step_key="materialize_runtime",
            action="start",
            detail=None,
        )

        endpoint = str(adapter_output.get("endpoint_url") or "").strip()
        health = str(adapter_output.get("health_check_url") or "").strip()
        if endpoint:
            deployment.endpoint_url = endpoint
        if health:
            deployment.health_check_url = health

        cfg = dict(deployment.config or {})
        cfg.setdefault("serving_defaults", {})
        cfg["serving_defaults"]["conf"] = float(ctx.defaults.get("conf", 0.25))
        cfg["serving_defaults"]["iou"] = float(ctx.defaults.get("iou", 0.45))
        cfg["last_deployment_run_id"] = str(run.run_id)
        cfg["last_materialized_at"] = _utcnow().isoformat()
        deployment.config = cfg

        snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
        pending_hash = str(snapshot.get("pending_api_key_hash") or "").strip()
        if pending_hash:
            deployment.api_key_hash = pending_hash
            deployment.api_key_hint = str(snapshot.get("api_key_hint") or "") or None

        self._set_step_status(run, key="materialize_runtime", status="completed")
        run.progress = 55
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Runtime materialized",
            step_key="materialize_runtime",
            action="completed",
            detail={"endpoint_url": deployment.endpoint_url, "health_check_url": deployment.health_check_url},
        )
        db.commit()

    def _step_smoke_test(self, db: Session, run: DeploymentRun, deployment: Deployment, model_context: dict[str, Any]) -> None:
        run.phase = DeploymentRunPhase.SMOKE_TEST
        run.current_step = "smoke_test"
        run.progress = 65
        self._set_step_status(run, key="smoke_test", status="running")
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Running smoke test inference",
            step_key="smoke_test",
            action="start",
            detail=None,
        )
        db.commit()

        smoke_dir = (settings.temp_dir / "deployment_smoke").resolve()
        smoke_dir.mkdir(parents=True, exist_ok=True)
        smoke_image = smoke_dir / f"{run.run_id}.jpg"
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(smoke_image)

        defaults = self._get_defaults(run)
        out = self._infer.run_inference_output(
            db,
            model_version_id=int(run.model_version_id),
            input_path=str(smoke_image),
            conf=float(defaults["conf"]),
            iou=float(defaults["iou"]),
        )
        err = str(out.get("error_message") or "").strip()
        if err:
            raise RuntimeError(f"Smoke test failed: {err}")

        output = out.get("output") if isinstance(out.get("output"), dict) else {}
        preds = output.get("predictions") if isinstance(output, dict) else None
        pred_count = len(preds) if isinstance(preds, list) else 0
        self._set_step_status(run, key="smoke_test", status="completed")
        run.progress = 80
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Smoke test passed",
            step_key="smoke_test",
            action="completed",
            detail={"detections": pred_count, "inference_time_ms": out.get("inference_time_ms")},
        )
        db.commit()

    def _step_activate(
        self,
        db: Session,
        run: DeploymentRun,
        deployment: Deployment,
        ctx: DeploymentAdapterContext,
        *,
        adapter_output: dict[str, Any],
    ) -> None:
        run.phase = DeploymentRunPhase.ACTIVATE
        run.current_step = "activate"
        run.progress = 90
        self._set_step_status(run, key="activate", status="running")
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Activating deployment",
            step_key="activate",
            action="start",
            detail=None,
        )

        _ = adapter_output
        self._deactivate_other_deployments(db, project_id=int(run.project_id), keep_deployment_id=int(deployment.deployment_id))
        deployment.status = DeploymentStatus.ACTIVE
        deployment.is_active = True
        deployment.deployed_at = _utcnow()
        self._sync_model_stage(db, project_id=int(run.project_id), model_version_id=int(run.model_version_id))

        self._set_step_status(run, key="activate", status="completed")
        run.status = DeploymentRunStatus.COMPLETED
        run.phase = DeploymentRunPhase.DONE
        run.progress = 100
        run.finished_at = _utcnow()
        run.error_message = None
        self._append_log(
            db,
            run=run,
            level=LogLevel.INFO,
            message="Deployment activated",
            step_key="activate",
            action="completed",
            detail={
                "endpoint_url": deployment.endpoint_url,
                "health_check_url": deployment.health_check_url,
                "api_key_hint": deployment.api_key_hint,
            },
        )
        db.commit()

    def _sync_model_stage(self, db: Session, *, project_id: int, model_version_id: int) -> None:
        db.query(ModelVersion).filter(
            ModelVersion.project_id == int(project_id),
            ModelVersion.model_version_id != int(model_version_id),
            ModelVersion.stage == ModelStage.PRODUCTION,
        ).update({ModelVersion.stage: ModelStage.TESTING}, synchronize_session=False)
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
        if mv:
            mv.stage = ModelStage.PRODUCTION

    def _deactivate_other_deployments(self, db: Session, *, project_id: int, keep_deployment_id: int) -> None:
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

    def _append_log(
        self,
        db: Session,
        *,
        run: DeploymentRun,
        level: LogLevel,
        message: str,
        step_key: Optional[str],
        action: str,
        detail: Optional[dict[str, Any]],
    ) -> None:
        snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
        seq = int(snapshot.get("last_seq") or 0) + 1
        snapshot["last_seq"] = seq
        run.snapshot = snapshot

        data = {
            "run_id": str(run.run_id),
            "seq": int(seq),
            "step_key": str(step_key or ""),
            "action": str(action),
            "detail": detail or {},
        }
        db.add(
            DeploymentLog(
                deployment_id=int(run.deployment_id),
                level=level,
                message=str(message),
                data=data,
            )
        )

    def _set_step_status(self, run: DeploymentRun, *, key: str, status: str, detail: Optional[str] = None) -> None:
        if not key:
            return
        snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
        steps = snapshot.get("steps")
        if not isinstance(steps, list):
            steps = _snapshot_steps()
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("key") or "") != str(key):
                continue
            step["status"] = str(status)
            if detail:
                step["detail"] = str(detail)
            step["updated_at"] = _utcnow().isoformat()
        snapshot["steps"] = steps
        run.snapshot = snapshot

    def _check_cancelled(self, db: Session, run: DeploymentRun) -> bool:
        db.refresh(run)
        if not bool(run.cancel_requested):
            return False
        run.status = DeploymentRunStatus.CANCELLED
        run.phase = DeploymentRunPhase.CANCELLED
        run.finished_at = _utcnow()
        self._set_step_status(run, key=str(run.current_step or ""), status="cancelled", detail="cancel requested")
        self._append_log(
            db,
            run=run,
            level=LogLevel.WARNING,
            message="Deployment run cancelled",
            step_key=run.current_step,
            action="cancelled",
            detail=None,
        )
        dep = db.query(Deployment).filter(Deployment.deployment_id == int(run.deployment_id)).first()
        if dep and dep.status == DeploymentStatus.DEPLOYING and not dep.is_active:
            dep.status = DeploymentStatus.PENDING
        db.commit()
        return True
