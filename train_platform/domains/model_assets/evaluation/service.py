from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import session_scope
from train_platform.platform.jobs import JobNotFoundError, JobStatus, JobStore, JobStoreError, is_active_status
from train_platform.platform.runtime import ModelWorkerClient
from train_platform.schemas.v3.model_evaluations import ModelEvaluationCreate, ModelEvaluationOut
from train_platform.utils.exceptions import ConflictError, ValidationError

from ..runtime import resolve_model_runtime
from ..versions.service import ModelVersionService
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

    def __init__(self, worker_client: ModelWorkerClient | None = None) -> None:
        self._worker = worker_client or ModelWorkerClient()
        self._versions = ModelVersionService()

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

    def create_job(self, db: Session, payload: ModelEvaluationCreate) -> ModelEvaluationOut:
        with _ACTIVE_CREATE_LOCK:
            active = self._has_active_job()
            if active:
                job_id = active.get("job_id") or "unknown"
                status = active.get("status") or JobStatus.RUNNING.value
                raise ConflictError(f"Another evaluation job is active (job_id={job_id}, status={status})")

            model_version = self._versions.resolve_for_execution(
                db,
                model_version_id=payload.model_version_id,
                run_id=payload.run_id,
                description="Auto-created for model evaluation jobs",
            )
            runtime = resolve_model_runtime(db, model_version=model_version)
            prepared = prepare_evaluation(
                db,
                standard_dataset_id=payload.standard_dataset_id,
                scope=payload.scope,
                model=runtime,
                conf=float(payload.conf),
                iou=float(payload.iou),
            )
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
                "model_version_id": int(runtime.model_version_id),
                "run_id": str(runtime.run_id or payload.run_id or ""),
                "standard_dataset_id": int(prepared.standard_dataset_id),
                "dataset_name": prepared.dataset_name,
                "scope": prepared.scope,
                "conf": float(prepared.conf),
                "iou": float(prepared.iou),
                "engine": str(runtime.engine or ""),
                "family": runtime.family,
                "variant": runtime.variant,
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
            runtime = resolve_model_runtime(db, model_version_id=int(status["model_version_id"]))
            prepared = prepare_evaluation(
                db,
                standard_dataset_id=int(status["standard_dataset_id"]),
                scope=str(status.get("scope") or "all"),
                model=runtime,
                conf=float(status.get("conf") or 0.25),
                iou=float(status.get("iou") or 0.5),
            )
        return status, prepared

    def _run_job(self, job_id: str) -> None:
        _, prepared = self._prepare_job_snapshot(job_id)
        observer = _JobExecutionObserver(self._store(), job_id)
        result = run_evaluation(
            prepared,
            job_dir=self.job_dir(job_id),
            worker_client=self._worker,
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
