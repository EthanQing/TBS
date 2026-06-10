from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.v3.illegal_dataset import IllegalDatasetPublishJob
from train_platform.schemas.v3.illegal_datasets import IllegalDatasetPublishJobOut, IllegalDatasetPublishRequest
from train_platform.services.v3.illegal_dataset_service import IllegalDatasetService
from train_platform.utils.exceptions import NotFoundError


TERMINAL_RETRYABLE_STATUSES = {"failed", "cancelled"}
ACTIVE_OR_DONE_STATUSES = {"queued", "running", "completed"}


class IllegalDatasetPublishJobService:
    def __init__(self) -> None:
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._svc = IllegalDatasetService()

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def jobs_root(self, illegal_dataset_id: int) -> Path:
        root = settings.temp_dir / "illegal_dataset_publish_jobs" / str(int(illegal_dataset_id))
        root.mkdir(parents=True, exist_ok=True)
        return root

    def job_dir(self, illegal_dataset_id: int, job_id: str) -> Path:
        path = self.jobs_root(illegal_dataset_id) / str(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def status_path(self, illegal_dataset_id: int, job_id: str) -> Path:
        return self.job_dir(illegal_dataset_id, job_id) / "status.json"

    def request_path(self, illegal_dataset_id: int, job_id: str) -> Path:
        return self.job_dir(illegal_dataset_id, job_id) / "request.json"

    def _lock(self, illegal_dataset_id: int, job_id: str) -> threading.RLock:
        key = f"{int(illegal_dataset_id)}:{str(job_id)}"
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        text = json.dumps(payload, ensure_ascii=False, default=str)
        last_exc: Exception | None = None
        for attempt in range(8):
            try:
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(path)
                return
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.03 * (attempt + 1))
            finally:
                if tmp.exists() and attempt >= 7:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
        if last_exc:
            raise last_exc

    def _read_json_retry(self, path: Path, *, missing_message: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(8):
            try:
                if not path.exists():
                    raise NotFoundError(missing_message)
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise NotFoundError(missing_message)
                return data
            except NotFoundError:
                raise
            except (PermissionError, OSError, json.JSONDecodeError) as exc:
                last_exc = exc
                time.sleep(0.03 * (attempt + 1))
        if last_exc:
            raise last_exc
        raise NotFoundError(missing_message)

    def _write_status_file(self, illegal_dataset_id: int, job_id: str, payload: dict[str, Any]) -> None:
        data = dict(payload or {})
        data["updated_at"] = self._utcnow().isoformat()
        with self._lock(illegal_dataset_id, job_id):
            self._write_json_atomic(self.status_path(illegal_dataset_id, job_id), data)

    def _write_request_file(self, illegal_dataset_id: int, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock(illegal_dataset_id, job_id):
            self._write_json_atomic(self.request_path(illegal_dataset_id, job_id), payload)

    def _read_request_file(self, illegal_dataset_id: int, job_id: str) -> dict[str, Any]:
        with self._lock(illegal_dataset_id, job_id):
            return self._read_json_retry(
                self.request_path(illegal_dataset_id, job_id),
                missing_message="Illegal dataset publish request not found",
            )

    def _append_log(self, payload: dict[str, Any], message: str) -> None:
        msg = str(message or "").strip()
        if not msg:
            return
        logs = payload.get("logs")
        if not isinstance(logs, list):
            logs = []
        if logs and str(logs[-1]).strip() == msg:
            payload["logs"] = logs
            return
        logs.append(msg)
        if len(logs) > 200:
            logs = logs[-200:]
        payload["logs"] = logs

    def _request_summary(self, payload: dict[str, Any], *, version_id: int | None = None) -> dict[str, Any]:
        overrides = payload.get("label_mapping_overrides")
        filters = payload.get("label_filters")
        return {
            "name": str(payload.get("name") or "").strip(),
            "version_id": int(version_id) if version_id is not None else (
                int(payload["version_id"]) if payload.get("version_id") is not None else None
            ),
            "label_mapping_overrides_count": len(overrides) if isinstance(overrides, dict) else 0,
            "label_filters_count": len(filters) if isinstance(filters, list) else 0,
        }

    def _phase_progress(self, phase: str, *, completed: int = 0, total: int = 0) -> int:
        phase_key = str(phase or "").strip().lower()
        if phase_key == "queued":
            return 0
        if phase_key == "preparing":
            return 5
        if phase_key == "materializing":
            return 15
        if phase_key == "converting":
            if total > 0:
                base = 20
                span = 70
                pct = max(0.0, min(1.0, float(completed) / float(total)))
                return min(90, max(base, base + int(round(span * pct))))
            return 20
        if phase_key == "publishing":
            return 95
        if phase_key in {"done", "completed", "failed", "cancelled"}:
            return 100
        return 0

    def _canonical_payload_for_hash(
        self,
        *,
        illegal_dataset_id: int,
        source_version_id: int,
        payload: dict[str, Any],
        mapping_snapshot: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "source_illegal_dataset_id": int(illegal_dataset_id),
            "source_illegal_version_id": int(source_version_id),
            "label_filters": payload.get("label_filters") or [],
            "label_mappings": mapping_snapshot,
            "split": payload.get("split") or {},
            "publish_config": payload.get("publish_config") or {},
        }

    def _idempotency_key(
        self,
        *,
        illegal_dataset_id: int,
        source_version_id: int,
        payload: dict[str, Any],
        mapping_snapshot: dict[str, str],
    ) -> str:
        canonical = self._canonical_payload_for_hash(
            illegal_dataset_id=int(illegal_dataset_id),
            source_version_id=int(source_version_id),
            payload=payload,
            mapping_snapshot=mapping_snapshot,
        )
        raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _effective_mapping_snapshot(self, db: Session, illegal_dataset_id: int, payload: dict[str, Any]) -> dict[str, str]:
        mapping_rows = self._svc.get_label_mappings(db, int(illegal_dataset_id))
        mapping_snapshot = {
            raw_label: mapped_label
            for item in mapping_rows
            for raw_label, mapped_label in [
                (str(item.raw_label or "").strip(), self._svc._effective_label_mapping_value(item))
            ]
            if raw_label and mapped_label
        }
        overrides = payload.get("label_mapping_overrides") or {}
        if isinstance(overrides, dict):
            for raw_label, raw_value in overrides.items():
                raw_label_s = str(raw_label or "").strip()
                mapped_label = self._svc._normalize_label_mapping_override_value(raw_value)
                if raw_label_s and mapped_label:
                    mapping_snapshot[raw_label_s] = mapped_label
        return mapping_snapshot

    def _job_to_payload(self, job: IllegalDatasetPublishJob, *, reused: bool = False) -> dict[str, Any]:
        result = job.result if isinstance(job.result, dict) else None
        payload = {
            "job_id": str(job.job_id),
            "illegal_dataset_id": int(job.illegal_dataset_id),
            "status": str(job.status or "queued"),
            "phase": str(job.phase or job.status or "queued"),
            "progress": int(job.progress or 0),
            "processed": int(job.processed or 0),
            "total": int(job.total or 0),
            "seq": int(job.seq or 0),
            "request": job.request_summary if isinstance(job.request_summary, dict) else None,
            "result": result,
            "logs": list(job.logs) if isinstance(job.logs, list) else [],
            "error_message": job.error_message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "idempotency_key": str(job.idempotency_key or ""),
            "reused": bool(reused),
        }
        if result is None and job.standard_dataset_id is not None:
            payload["result"] = {
                "standard_dataset_id": int(job.standard_dataset_id),
                "name": "",
                "source_illegal_dataset_id": int(job.illegal_dataset_id),
                "source_illegal_version_id": int(job.source_illegal_version_id),
                "publish_config": {},
            }
        return payload

    def _job_out(self, job: IllegalDatasetPublishJob, *, reused: bool = False) -> IllegalDatasetPublishJobOut:
        return IllegalDatasetPublishJobOut.model_validate(self._job_to_payload(job, reused=reused))

    def _sync_status_file(self, job: IllegalDatasetPublishJob, *, reused: bool = False) -> None:
        try:
            self._write_status_file(int(job.illegal_dataset_id), str(job.job_id), self._job_to_payload(job, reused=reused))
        except Exception:
            pass

    def _get_job_row(self, db: Session, illegal_dataset_id: int, job_id: str) -> IllegalDatasetPublishJob:
        row = (
            db.query(IllegalDatasetPublishJob)
            .filter(
                IllegalDatasetPublishJob.illegal_dataset_id == int(illegal_dataset_id),
                IllegalDatasetPublishJob.job_id == str(job_id),
            )
            .first()
        )
        if not row:
            raise NotFoundError("Illegal dataset publish job not found")
        return row

    def _get_job_by_key(self, db: Session, key: str) -> IllegalDatasetPublishJob | None:
        return db.query(IllegalDatasetPublishJob).filter(IllegalDatasetPublishJob.idempotency_key == str(key)).first()

    def _reset_retryable_job(
        self,
        db: Session,
        job: IllegalDatasetPublishJob,
        *,
        request_payload: dict[str, Any],
        request_summary: dict[str, Any],
    ) -> IllegalDatasetPublishJob:
        job.status = "queued"
        job.phase = "queued"
        job.progress = 0
        job.processed = 0
        job.total = 0
        job.seq = int(job.seq or 0) + 1
        job.request_payload = request_payload
        job.request_summary = request_summary
        job.result = None
        job.standard_dataset_id = None
        job.logs = ["上次转换未完成，已重新加入后台执行队列"]
        job.error_message = None
        job.started_at = None
        job.finished_at = None
        job.updated_at = self._utcnow()
        db.flush()
        return job

    def _build_new_job(
        self,
        *,
        illegal_dataset_id: int,
        source_version_id: int,
        idempotency_key: str,
        request_payload: dict[str, Any],
        request_summary: dict[str, Any],
    ) -> IllegalDatasetPublishJob:
        return IllegalDatasetPublishJob(
            job_id=uuid.uuid4().hex,
            illegal_dataset_id=int(illegal_dataset_id),
            source_illegal_version_id=int(source_version_id),
            idempotency_key=str(idempotency_key),
            status="queued",
            phase="queued",
            progress=0,
            processed=0,
            total=0,
            seq=1,
            request_payload=request_payload,
            request_summary=request_summary,
            result=None,
            logs=["转换任务已创建，等待后台执行"],
            error_message=None,
        )

    def create_job(self, db: Session, illegal_dataset_id: int, payload: IllegalDatasetPublishRequest) -> IllegalDatasetPublishJobOut:
        dataset = self._svc.get_dataset(db, int(illegal_dataset_id))
        version = self._svc._selected_version(db, dataset, version_id=payload.version_id)
        request_payload = payload.model_dump(mode="json")
        request_payload["version_id"] = int(version.version_id)
        mapping_snapshot = self._effective_mapping_snapshot(db, int(illegal_dataset_id), request_payload)
        key = self._idempotency_key(
            illegal_dataset_id=int(illegal_dataset_id),
            source_version_id=int(version.version_id),
            payload=request_payload,
            mapping_snapshot=mapping_snapshot,
        )
        summary = self._request_summary(request_payload, version_id=int(version.version_id))

        existing = self._get_job_by_key(db, key)
        reused = False
        if existing is not None:
            if str(existing.status or "").lower() in TERMINAL_RETRYABLE_STATUSES:
                job = self._reset_retryable_job(db, existing, request_payload=request_payload, request_summary=summary)
            else:
                job = existing
                reused = str(job.status or "").lower() in ACTIVE_OR_DONE_STATUSES
        else:
            job = self._build_new_job(
                illegal_dataset_id=int(illegal_dataset_id),
                source_version_id=int(version.version_id),
                idempotency_key=key,
                request_payload=request_payload,
                request_summary=summary,
            )
            db.add(job)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                job = self._get_job_by_key(db, key)
                if job is None:
                    raise
                reused = True

        db.commit()
        db.refresh(job)
        self._write_request_file(int(illegal_dataset_id), str(job.job_id), request_payload)
        self._sync_status_file(job, reused=reused)
        return self._job_out(job, reused=reused)

    def get_job(self, illegal_dataset_id: int, job_id: str) -> IllegalDatasetPublishJobOut:
        db = SessionLocal()
        try:
            job = self._get_job_row(db, int(illegal_dataset_id), str(job_id))
            return self._job_out(job)
        except NotFoundError:
            with self._lock(illegal_dataset_id, job_id):
                return IllegalDatasetPublishJobOut.model_validate(
                    self._read_json_retry(
                        self.status_path(illegal_dataset_id, job_id),
                        missing_message="Illegal dataset publish job not found",
                    )
                )
        finally:
            db.close()

    def start_job(self, illegal_dataset_id: int, job_id: str) -> None:
        thread = threading.Thread(
            target=self._run_job,
            args=(int(illegal_dataset_id), str(job_id)),
            daemon=True,
        )
        thread.start()

    def _claim_job(self, db: Session, illegal_dataset_id: int, job_id: str) -> IllegalDatasetPublishJob | None:
        now = self._utcnow()
        claimed = (
            db.query(IllegalDatasetPublishJob)
            .filter(
                IllegalDatasetPublishJob.illegal_dataset_id == int(illegal_dataset_id),
                IllegalDatasetPublishJob.job_id == str(job_id),
                IllegalDatasetPublishJob.status == "queued",
            )
            .update(
                {
                    IllegalDatasetPublishJob.status: "running",
                    IllegalDatasetPublishJob.phase: "preparing",
                    IllegalDatasetPublishJob.progress: self._phase_progress("preparing"),
                    IllegalDatasetPublishJob.error_message: None,
                    IllegalDatasetPublishJob.started_at: now,
                    IllegalDatasetPublishJob.finished_at: None,
                    IllegalDatasetPublishJob.updated_at: now,
                    IllegalDatasetPublishJob.seq: IllegalDatasetPublishJob.seq + 1,
                },
                synchronize_session=False,
            )
        )
        if int(claimed or 0) <= 0:
            db.rollback()
            return None
        db.commit()
        job = self._get_job_row(db, int(illegal_dataset_id), str(job_id))
        payload = self._job_to_payload(job)
        self._append_log(payload, "开始执行转换任务")
        job.logs = payload["logs"]
        job.seq = int(job.seq or 0) + 1
        job.updated_at = self._utcnow()
        db.commit()
        db.refresh(job)
        self._sync_status_file(job)
        return job

    def _update_status(
        self,
        illegal_dataset_id: int,
        job_id: str,
        patch: dict[str, Any],
        *,
        message: str | None = None,
    ) -> dict[str, Any]:
        db = SessionLocal()
        try:
            job = self._get_job_row(db, int(illegal_dataset_id), str(job_id))
            payload = self._job_to_payload(job)
            payload.update(dict(patch or {}))
            payload["seq"] = int(payload.get("seq") or 0) + 1
            if message:
                self._append_log(payload, message)

            job.status = str(payload.get("status") or job.status or "queued")
            job.phase = str(payload.get("phase") or job.phase or job.status)
            job.progress = int(payload.get("progress") or 0)
            job.processed = int(payload.get("processed") or 0)
            job.total = int(payload.get("total") or 0)
            job.seq = int(payload.get("seq") or 0)
            job.logs = payload.get("logs") if isinstance(payload.get("logs"), list) else []
            job.error_message = payload.get("error_message")
            if isinstance(payload.get("result"), dict):
                job.result = payload["result"]
                standard_dataset_id = payload["result"].get("standard_dataset_id")
                job.standard_dataset_id = int(standard_dataset_id) if standard_dataset_id is not None else None
            if job.status in {"completed", "failed", "cancelled"}:
                job.finished_at = self._utcnow()
            job.updated_at = self._utcnow()
            db.commit()
            db.refresh(job)
            synced = self._job_to_payload(job)
            self._sync_status_file(job)
            return synced
        finally:
            db.close()

    def _run_job(self, illegal_dataset_id: int, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = self._claim_job(db, int(illegal_dataset_id), str(job_id))
            if job is None:
                return
            request_payload = job.request_payload if isinstance(job.request_payload, dict) else None
            if request_payload is None:
                request_payload = self._read_request_file(int(illegal_dataset_id), str(job_id))
        except Exception as exc:
            db.close()
            msg = f"{type(exc).__name__}: {exc}"
            self._update_status(
                int(illegal_dataset_id),
                str(job_id),
                {"status": "failed", "phase": "failed", "progress": 100, "error_message": msg},
                message=msg,
            )
            return

        def progress_callback(phase: str, info: dict[str, Any]) -> None:
            payload = info if isinstance(info, dict) else {}
            total = max(0, int(payload.get("total") or 0))
            completed = max(
                0,
                int(
                    payload.get("completed")
                    if payload.get("completed") is not None
                    else payload.get("processed") or 0
                ),
            )
            patch: dict[str, Any] = {
                "status": "running",
                "phase": str(phase or "").strip().lower() or "running",
                "progress": self._phase_progress(phase, completed=completed, total=total),
            }
            if total > 0:
                patch["total"] = total
                patch["processed"] = completed
            self._update_status(
                int(illegal_dataset_id),
                str(job_id),
                patch,
                message=str(payload.get("message") or "").strip() or None,
            )

        try:
            result = self._svc.publish_standard_dataset(
                db,
                int(illegal_dataset_id),
                obj=request_payload,
                progress_callback=progress_callback,
                publish_job_id=str(job_id),
                idempotency_key=str(job.idempotency_key),
            )
            total = int((((result.get("publish_config") or {}).get("conversion_result", {})).get("pairs_total", 0)) or 0)
            processed = int((((result.get("publish_config") or {}).get("conversion_result", {})).get("pairs_processed", 0)) or 0)
            self._update_status(
                int(illegal_dataset_id),
                str(job_id),
                {
                    "status": "completed",
                    "phase": "done",
                    "progress": 100,
                    "processed": processed,
                    "total": total,
                    "result": result,
                    "error_message": None,
                },
                message=f"转换完成，已生成标准数据集 #{int(result.get('standard_dataset_id') or 0)}",
            )
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            msg = f"{type(exc).__name__}: {exc}"
            self._update_status(
                int(illegal_dataset_id),
                str(job_id),
                {"status": "failed", "phase": "failed", "progress": 100, "error_message": msg},
                message=msg,
            )
        finally:
            db.close()
