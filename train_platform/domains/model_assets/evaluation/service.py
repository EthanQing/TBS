from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import session_scope
from train_platform.models.v3.enums import ModelStage, TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.platform.jobs import JobNotFoundError, JobStatus, JobStore, JobStoreError, is_active_status
from train_platform.schemas.v3.model_evaluations import ModelEvaluationCreate, ModelEvaluationOut
from train_platform.services.v3.inference_service import InferenceService
from train_platform.services.v3.model_version_service import ModelVersionService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

from .preparation import PreparedEvaluation, prepare_evaluation
from .runner import run_evaluation


_ACTIVE_CREATE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: Optional[datetime] = None) -> str:
    return (dt or _utcnow()).isoformat()


def _parse_time(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None


class _JobExecutionObserver:
    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id

    def is_cancel_requested(self) -> bool:
        try:
            return bool(self._store.read_status(self._job_id).get("cancel_requested"))
        except JobNotFoundError as exc:
            raise ValidationError("Evaluation job not found") from exc
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation status: {exc}") from exc

    def update_phase(
        self,
        phase: str,
        *,
        progress: int,
        total: int | None = None,
        processed: int | None = None,
    ) -> None:
        patch: dict[str, Any] = {
            "status": JobStatus.RUNNING,
            "phase": str(phase),
            "progress": int(progress),
        }
        if total is not None:
            patch["total"] = int(total)
        if processed is not None:
            patch["processed"] = int(processed)
        self._store.update(self._job_id, patch, bump_seq=True)

    def update_progress(self, *, processed: int, progress: int) -> None:
        self._store.update(
            self._job_id,
            {"processed": int(processed), "progress": int(progress)},
            bump_seq=True,
        )

    def emit_item(self, item: Mapping[str, Any]) -> None:
        self._store.append_result(self._job_id, item)

    def mark_cancelled(self) -> None:
        self._store.update(
            self._job_id,
            {"status": JobStatus.CANCELLED, "phase": "cancelled"},
            bump_seq=True,
        )


class EvaluationService:
    ACTIVE_STALE_AFTER = timedelta(hours=4)

    def __init__(self) -> None:
        self._infer = InferenceService()
        self._mv_svc = ModelVersionService()

    def jobs_root(self) -> Path:
        root = settings.temp_dir / "model_evaluations"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _store(self) -> JobStore:
        return JobStore(self.jobs_root())

    def job_dir(self, job_id: str) -> Path:
        return self._store().job_dir(job_id, create=True)

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex

    def _has_active_job(self) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        try:
            statuses = self._store().list_statuses()
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation jobs: {exc}") from exc
        for data in statuses:
            if not is_active_status(data.get("status")):
                continue
            updated = _parse_time(data.get("updated_at"))
            if updated and (now - updated) > self.ACTIVE_STALE_AFTER:
                continue
            return data
        return None

    def get_active_job(self, *, include_items: bool = False) -> Optional[ModelEvaluationOut]:
        active = self._has_active_job()
        if not active:
            return None
        job_id = str(active.get("job_id") or "").strip()
        if not job_id:
            return None
        return self.get_job(job_id, include_items=include_items)

    def _ensure_model_version_for_payload(
        self,
        db: Session,
        *,
        model_version_id: int | None,
        run_id: str | None,
    ) -> int:
        if model_version_id is not None:
            row = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
            if not row:
                raise NotFoundError("Model version not found")
            return int(row.model_version_id)

        rid = str(run_id or "").strip()
        if not rid:
            raise ValidationError("Missing model_version_id/run_id")

        existing = (
            db.query(ModelVersion)
            .filter(ModelVersion.run_id == rid)
            .order_by(ModelVersion.created_at.desc(), ModelVersion.model_version_id.desc())
            .first()
        )
        if existing:
            return int(existing.model_version_id)

        run = db.query(TrainingRun).filter(TrainingRun.run_id == rid).first()
        if not run:
            raise NotFoundError("Training run not found")
        if run.status != TrainingRunStatus.COMPLETED:
            raise ConflictError("Only completed runs can be evaluated")

        base = f"run-{rid[:8]}"
        for index in range(1, 200):
            version = base if index == 1 else f"{base}-{index}"
            try:
                model_version = self._mv_svc.register_from_run(
                    db,
                    run_id=rid,
                    version=version,
                    stage=ModelStage.DEVELOPMENT,
                    description="Auto-created for model evaluation jobs",
                )
                return int(model_version.model_version_id)
            except ConflictError:
                continue
        raise ConflictError("Failed to auto-register model version for evaluation")

    def create_job(self, db: Session, payload: ModelEvaluationCreate) -> ModelEvaluationOut:
        with _ACTIVE_CREATE_LOCK:
            active = self._has_active_job()
            if active:
                job_id = active.get("job_id") or "unknown"
                status = active.get("status") or JobStatus.RUNNING.value
                raise ConflictError(f"Another evaluation job is active (job_id={job_id}, status={status})")

            model_version_id = self._ensure_model_version_for_payload(
                db,
                model_version_id=payload.model_version_id,
                run_id=payload.run_id,
            )
            prepared = prepare_evaluation(
                db,
                standard_dataset_id=payload.standard_dataset_id,
                scope=payload.scope,
                model_version_id=model_version_id,
                conf=float(payload.conf),
                iou=float(payload.iou),
                inference_service=self._infer,
            )
            context = prepared.model_context
            job_id = self._new_job_id()
            self.job_dir(job_id)
            status: Dict[str, Any] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "phase": "preparing",
                "progress": 0,
                "processed": 0,
                "total": int(prepared.total_images),
                "seq": 1,
                "last_result_id": 0,
                "model_version_id": int(model_version_id),
                "run_id": str(context.get("run_id") or payload.run_id or ""),
                "standard_dataset_id": int(prepared.standard_dataset_id),
                "dataset_name": prepared.dataset_name,
                "scope": prepared.scope,
                "conf": float(prepared.conf),
                "iou": float(prepared.iou),
                "engine": str(context.get("engine") or ""),
                "family": context.get("family"),
                "variant": context.get("variant"),
                "cancel_requested": False,
                "skipped_images": int(prepared.skipped_images),
                "result": {"metrics": None},
                "error_message": None,
                "created_at": _to_iso(),
                "updated_at": _to_iso(),
            }
            self._store().create(job_id, status)

        thread = threading.Thread(target=self._run_job_thread, args=(job_id,), daemon=True)
        thread.start()
        return self.get_job(job_id, include_items=False)

    def read_results_since(self, job_id: str, after_result_id: int = 0) -> List[Dict[str, Any]]:
        try:
            return self._store().read_results_since(job_id, after_result_id=after_result_id)
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation job results: {exc}") from exc

    def _read_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            return self._store().read_status(job_id)
        except JobNotFoundError as exc:
            raise ValidationError("Evaluation job not found") from exc
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation status: {exc}") from exc

    def cancel_job(self, job_id: str) -> ModelEvaluationOut:
        try:
            self._store().cancel(
                job_id,
                terminal_if=(JobStatus.QUEUED, JobStatus.RUNNING),
                terminal_patch={"phase": "cancelled", "error_message": None},
            )
        except JobStoreError as exc:
            raise ValidationError(f"Failed to cancel evaluation job: {exc}") from exc
        return self.get_job(job_id, include_items=False)

    def get_job(self, job_id: str, *, include_items: bool = True) -> ModelEvaluationOut:
        status = self._read_job_status(job_id)
        result = status.get("result") if isinstance(status.get("result"), dict) else {"metrics": None}
        if include_items:
            result = dict(result or {})
            result["items"] = self.read_results_since(job_id, after_result_id=0)
        payload = dict(status)
        payload["result"] = result
        return ModelEvaluationOut.model_validate(payload)

    def _run_job_thread(self, job_id: str) -> None:
        try:
            self._run_job(job_id)
        except Exception as exc:
            try:
                self._store().update(
                    job_id,
                    {
                        "status": JobStatus.FAILED,
                        "phase": "failed",
                        "progress": 100,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    },
                    bump_seq=True,
                )
            except Exception:
                pass

    def _prepare_job_snapshot(self, job_id: str) -> tuple[Dict[str, Any], PreparedEvaluation]:
        status = self._read_job_status(job_id)
        with session_scope() as db:
            prepared = prepare_evaluation(
                db,
                standard_dataset_id=int(status["standard_dataset_id"]),
                scope=str(status.get("scope") or "all"),
                model_version_id=int(status["model_version_id"]),
                conf=float(status.get("conf") or 0.25),
                iou=float(status.get("iou") or 0.5),
                inference_service=self._infer,
            )
        return status, prepared

    def _run_job(self, job_id: str) -> None:
        _, prepared = self._prepare_job_snapshot(job_id)
        observer = _JobExecutionObserver(self._store(), job_id)
        result = run_evaluation(
            prepared,
            job_dir=self.job_dir(job_id),
            inference_service=self._infer,
            observer=observer,
        )
        if result.cancelled:
            return

        self._store().update(
            job_id,
            {
                "status": JobStatus.COMPLETED,
                "phase": "done",
                "progress": 100,
                "processed": int(result.processed),
                "total": int(prepared.total_images),
                "result": {"metrics": result.metrics},
                "error_message": None,
            },
            bump_seq=True,
        )


__all__ = ["EvaluationService"]
