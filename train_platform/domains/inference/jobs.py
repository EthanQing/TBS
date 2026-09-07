from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.core.license import assert_valid_license
from train_platform.domains.model_assets.runtime import ModelRuntimeSpec, resolve_model_runtime
from train_platform.domains.model_assets.versions.service import ModelVersionService
from train_platform.platform.jobs import JobNotFoundError, JobStatus, JobStore, JobStoreError, is_active_status
from train_platform.platform.runtime import ModelWorkerClient
from train_platform.schemas.v3.inference_jobs import InferenceJobCreate, InferenceJobOut
from train_platform.utils.exceptions import ConflictError, ValidationError


_CREATE_JOB_PROCESS_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso() -> str:
    return _utcnow().isoformat()


def _parse_time(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


class InferenceJobService:
    ACTIVE_STALE_AFTER = timedelta(hours=2)
    CREATE_JOB_LOCK_TIMEOUT_SEC = float(os.getenv("INFERENCE_JOB_CREATE_LOCK_TIMEOUT_SEC", "5"))

    def __init__(self, worker_client: ModelWorkerClient | None = None) -> None:
        self._versions = ModelVersionService()
        self._worker = worker_client or ModelWorkerClient()

    def jobs_root(self) -> Path:
        root = settings.temp_dir / "inference_jobs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _store(self) -> JobStore:
        return JobStore(self.jobs_root())

    def _create_job_lock_path(self) -> Path:
        return self.jobs_root() / ".create_job.lock"

    def _acquire_create_job_lock(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        lock_path = self._create_job_lock_path()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n{time.time()}\n")
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ConflictError("Another inference job request is being processed, please retry")
                try:
                    if lock_path.exists():
                        age = time.time() - float(lock_path.stat().st_mtime)
                        if age > max(30.0, float(timeout_sec) * 3.0):
                            lock_path.unlink(missing_ok=True)
                            continue
                except Exception:
                    pass
                time.sleep(0.05)

    def _release_create_job_lock(self) -> None:
        try:
            self._create_job_lock_path().unlink(missing_ok=True)
        except OSError:
            pass

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex

    def _has_active_job(self) -> Optional[Dict[str, Any]]:
        try:
            statuses = self._store().list_statuses()
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read inference jobs: {exc}") from exc
        now = _utcnow()
        for data in statuses:
            if not is_active_status(data.get("status")):
                continue
            updated = _parse_time(data.get("updated_at"))
            if updated and (now - updated) > self.ACTIVE_STALE_AFTER:
                continue
            return data
        return None

    def _normalize_inputs(self, payload: InferenceJobCreate) -> tuple[list[str], Optional[str]]:
        if payload.mode == "video":
            return [], str(payload.video_token or "").strip()
        tokens = [str(item).strip() for item in payload.input_tokens if str(item).strip()]
        if not tokens:
            raise ValidationError("No input tokens provided")
        if payload.mode == "image":
            tokens = tokens[:1]
        return tokens, None

    def _dispatch_job_to_worker(
        self,
        job_id: str,
        *,
        status: Dict[str, Any],
        model: ModelRuntimeSpec,
    ) -> None:
        self._worker.dispatch_inference_job(
            engine=str(model.engine or status.get("engine") or "ultralytics-yolo"),
            job_id=job_id,
            mode=str(status.get("mode") or "image"),
            weights_path=model.weights_path,
            input_tokens=list(status.get("input_tokens") or []),
            video_token=status.get("video_token"),
            conf=float(status.get("conf") or 0.5),
            iou=float(status.get("iou") or 0.45),
            show_labels=bool(status.get("show_labels", True)),
            show_confidence=bool(status.get("show_confidence", True)),
            config_path=model.config_path,
        )

    def create_job(self, db: Session, payload: InferenceJobCreate) -> InferenceJobOut:
        assert_valid_license()
        with _CREATE_JOB_PROCESS_LOCK:
            self._acquire_create_job_lock(self.CREATE_JOB_LOCK_TIMEOUT_SEC)
            try:
                active = self._has_active_job()
                if active:
                    job_id = active.get("job_id") or "unknown"
                    status = active.get("status") or JobStatus.RUNNING.value
                    raise ConflictError(f"Another inference job is active (job_id={job_id}, status={status})")

                model_version = self._versions.resolve_for_execution(
                    db,
                    model_version_id=payload.model_version_id,
                    run_id=payload.run_id,
                    description="Auto-created for inference jobs",
                )
                model = resolve_model_runtime(db, model_version=model_version)
                tokens, video_token = self._normalize_inputs(payload)
                mode = str(payload.mode)
                total = 0 if mode == "video" else len(tokens)
                job_id = self._new_job_id()
                status: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": JobStatus.QUEUED,
                    "phase": "preparing",
                    "mode": mode,
                    "progress": 0,
                    "processed": 0,
                    "total": int(total),
                    "seq": 1,
                    "last_result_id": 0,
                    "model_version_id": int(model.model_version_id),
                    "run_id": str(model.run_id or payload.run_id or ""),
                    "engine": str(model.engine or ""),
                    "family": model.family,
                    "variant": model.variant,
                    "conf": float(payload.conf),
                    "iou": float(payload.iou),
                    "show_labels": bool(payload.show_labels),
                    "show_confidence": bool(payload.show_confidence),
                    "input_tokens": tokens,
                    "video_token": video_token,
                    "cancel_requested": False,
                    "result": {"mode": mode},
                    "error_message": None,
                    "created_at": _to_iso(),
                    "updated_at": _to_iso(),
                }
                status = self._store().create(job_id, status)
            finally:
                self._release_create_job_lock()

        try:
            self._dispatch_job_to_worker(job_id, status=status, model=model)
        except Exception as exc:
            self._store().update(
                job_id,
                {
                    "status": JobStatus.FAILED,
                    "phase": "failed",
                    "progress": 100,
                    "error_message": f"Failed to dispatch inference job to worker: {type(exc).__name__}: {exc}",
                },
                bump_seq=True,
            )
        return self.get_job(job_id, include_items=False)

    def read_results_since(self, job_id: str, after_result_id: int = 0) -> List[Dict[str, Any]]:
        try:
            return self._store().read_results_since(job_id, after_result_id=after_result_id)
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read inference job results: {exc}") from exc

    def _read_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            return self._store().read_status(job_id)
        except JobNotFoundError as exc:
            raise ValidationError("Job not found") from exc
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read inference job status: {exc}") from exc

    def cancel_job(self, job_id: str) -> InferenceJobOut:
        try:
            self._store().cancel(
                job_id,
                terminal_if=(JobStatus.QUEUED,),
                terminal_patch={"phase": "cancelled", "progress": 0},
            )
        except JobStoreError as exc:
            raise ValidationError(f"Failed to cancel inference job: {exc}") from exc
        return self.get_job(job_id, include_items=False)

    def get_job(self, job_id: str, *, include_items: bool = True) -> InferenceJobOut:
        status = self._read_job_status(job_id)
        result = status.get("result") if isinstance(status.get("result"), dict) else {"mode": status.get("mode")}
        if include_items and str(status.get("mode")) in {"image", "batch"}:
            result = dict(result or {})
            result.setdefault("mode", status.get("mode"))
            result["items"] = self.read_results_since(job_id, after_result_id=0)
        payload = dict(status)
        payload["result"] = result
        return InferenceJobOut.model_validate(payload)

    def list_jobs_for_debug(self) -> List[Dict[str, Any]]:
        try:
            return self._store().list_statuses()
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read inference jobs: {exc}") from exc


__all__ = ["InferenceJobService"]
