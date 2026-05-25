from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.schemas.v3.illegal_datasets import IllegalDatasetPublishJobOut, IllegalDatasetPublishRequest
from train_platform.services.v3.illegal_dataset_service import IllegalDatasetService
from train_platform.utils.exceptions import NotFoundError


class IllegalDatasetPublishJobService:
    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
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

    def _lock(self, illegal_dataset_id: int, job_id: str) -> threading.Lock:
        key = f"{int(illegal_dataset_id)}:{str(job_id)}"
        self._locks.setdefault(key, threading.Lock())
        return self._locks[key]

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)

    def _write_status(self, illegal_dataset_id: int, job_id: str, payload: dict[str, Any]) -> None:
        data = dict(payload or {})
        data["updated_at"] = self._utcnow().isoformat()
        self._write_json_atomic(self.status_path(illegal_dataset_id, job_id), data)

    def _read_status(self, illegal_dataset_id: int, job_id: str) -> dict[str, Any]:
        path = self.status_path(illegal_dataset_id, job_id)
        if not path.exists():
            raise NotFoundError("Illegal dataset publish job not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise NotFoundError("Illegal dataset publish job not found")
        return data

    def _read_request(self, illegal_dataset_id: int, job_id: str) -> dict[str, Any]:
        path = self.request_path(illegal_dataset_id, job_id)
        if not path.exists():
            raise NotFoundError("Illegal dataset publish request not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise NotFoundError("Illegal dataset publish request not found")
        return data

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

    def _request_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        overrides = payload.get("label_mapping_overrides")
        filters = payload.get("label_filters")
        return {
            "name": str(payload.get("name") or "").strip(),
            "version_id": int(payload["version_id"]) if payload.get("version_id") is not None else None,
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

    def _update_status(
        self,
        illegal_dataset_id: int,
        job_id: str,
        patch: dict[str, Any],
        *,
        message: str | None = None,
    ) -> dict[str, Any]:
        with self._lock(illegal_dataset_id, job_id):
            payload = self._read_status(illegal_dataset_id, job_id)
            payload.update(dict(patch or {}))
            payload["seq"] = int(payload.get("seq") or 0) + 1
            if message:
                self._append_log(payload, message)
            self._write_status(illegal_dataset_id, job_id, payload)
            return payload

    def create_job(self, db: Session, illegal_dataset_id: int, payload: IllegalDatasetPublishRequest) -> IllegalDatasetPublishJobOut:
        self._svc.get_dataset(db, int(illegal_dataset_id))
        job_id = uuid.uuid4().hex
        request_payload = payload.model_dump(mode="json")
        created_at = self._utcnow().isoformat()
        status = {
            "job_id": job_id,
            "illegal_dataset_id": int(illegal_dataset_id),
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "processed": 0,
            "total": 0,
            "seq": 1,
            "request": self._request_summary(request_payload),
            "result": None,
            "logs": ["转换任务已创建，等待后台执行"],
            "error_message": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        self._write_json_atomic(self.request_path(int(illegal_dataset_id), job_id), request_payload)
        self._write_status(int(illegal_dataset_id), job_id, status)
        return IllegalDatasetPublishJobOut.model_validate(status)

    def get_job(self, illegal_dataset_id: int, job_id: str) -> IllegalDatasetPublishJobOut:
        return IllegalDatasetPublishJobOut.model_validate(self._read_status(int(illegal_dataset_id), str(job_id)))

    def start_job(self, illegal_dataset_id: int, job_id: str) -> None:
        thread = threading.Thread(
            target=self._run_job,
            args=(int(illegal_dataset_id), str(job_id)),
            daemon=True,
        )
        thread.start()

    def _run_job(self, illegal_dataset_id: int, job_id: str) -> None:
        try:
            request_payload = self._read_request(int(illegal_dataset_id), str(job_id))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._update_status(
                int(illegal_dataset_id),
                str(job_id),
                {
                    "status": "failed",
                    "phase": "failed",
                    "progress": 100,
                    "error_message": msg,
                },
                message=msg,
            )
            return

        self._update_status(
            int(illegal_dataset_id),
            str(job_id),
            {
                "status": "running",
                "phase": "preparing",
                "progress": self._phase_progress("preparing"),
                "error_message": None,
            },
            message="开始执行转换任务",
        )

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

        db = SessionLocal()
        try:
            result = self._svc.publish_standard_dataset(
                db,
                int(illegal_dataset_id),
                obj=request_payload,
                progress_callback=progress_callback,
            )
            total = int(
                (
                    (result.get("publish_config") or {})
                    .get("conversion_result", {})
                    .get("pairs_total", 0)
                )
                or 0
            )
            processed = int(
                (
                    (result.get("publish_config") or {})
                    .get("conversion_result", {})
                    .get("pairs_processed", 0)
                )
                or 0
            )
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
                {
                    "status": "failed",
                    "phase": "failed",
                    "progress": 100,
                    "error_message": msg,
                },
                message=msg,
            )
        finally:
            db.close()
