from __future__ import annotations

import json
import math
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from fastapi import UploadFile
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.services.dataset_service import DatasetService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class DatasetUploadService:
    _lock = threading.Lock()
    _thread_by_job: dict[str, threading.Thread] = {}

    @property
    def _sessions_root(self) -> Path:
        p = (settings.temp_dir / "upload_sessions").resolve(strict=False)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def _jobs_root(self) -> Path:
        p = (settings.temp_dir / "dataset_import_jobs").resolve(strict=False)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _session_dir(self, session_id: str) -> Path:
        return (self._sessions_root / str(session_id)).resolve(strict=False)

    def _session_meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _parts_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "parts"

    def _archive_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "archive"

    def _job_meta_path(self, job_id: str) -> Path:
        return (self._jobs_root / f"{job_id}.json").resolve(strict=False)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _normalize_chunk_size(self, chunk_size: int | None) -> int:
        default_size = int(settings.upload_chunk_size_mb) * 1024 * 1024
        size = int(chunk_size) if chunk_size else default_size
        min_size = 1 * 1024 * 1024
        max_size = 256 * 1024 * 1024
        return max(min_size, min(max_size, size))

    def _cleanup_expired_sessions(self) -> None:
        now = _now_utc()
        for session_dir in self._sessions_root.iterdir():
            if not session_dir.is_dir():
                continue
            meta = self._read_json(session_dir / "session.json")
            if not meta:
                continue
            expires_at = _from_iso(str(meta.get("expires_at") or ""))
            status = str(meta.get("status") or "").strip().lower()
            if status in {"completed", "failed", "cancelled", "expired"}:
                continue
            if expires_at is not None and expires_at < now:
                meta["status"] = "expired"
                meta["updated_at"] = _to_iso(now)
                self._write_json(session_dir / "session.json", meta)

    def _active_session_for_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        now = _now_utc()
        for session_dir in self._sessions_root.iterdir():
            if not session_dir.is_dir():
                continue
            meta = self._read_json(session_dir / "session.json")
            if not meta:
                continue
            if int(meta.get("dataset_id") or -1) != int(dataset_id):
                continue
            status = str(meta.get("status") or "").strip().lower()
            expires_at = _from_iso(str(meta.get("expires_at") or ""))
            if status in {"completed", "failed", "cancelled", "expired"}:
                continue
            if expires_at is not None and expires_at < now:
                continue
            return meta
        return None

    def _coerce_uploaded_parts(self, meta: dict[str, Any]) -> list[int]:
        total_parts = int(meta.get("total_parts") or 0)
        out: list[int] = []
        for raw in list(meta.get("uploaded_parts") or []):
            try:
                n = int(raw)
            except Exception:
                continue
            if 1 <= n <= total_parts:
                out.append(n)
        return sorted(set(out))

    def create_session(self, db: Session, dataset_id: int, *, filename: str, total_size: int, chunk_size: int | None) -> dict[str, Any]:
        DatasetService().get_dataset(db, int(dataset_id))

        file_name = Path(str(filename or "").strip()).name
        if not file_name:
            raise ValidationError("filename is required")
        total_size = int(total_size or 0)
        if total_size <= 0:
            raise ValidationError("total_size must be > 0")
        normalized_chunk = self._normalize_chunk_size(chunk_size)
        total_parts = int(math.ceil(float(total_size) / float(normalized_chunk)))
        if total_parts <= 0:
            raise ValidationError("invalid total_parts")

        with self._lock:
            self._cleanup_expired_sessions()
            active = self._active_session_for_dataset(int(dataset_id))
            if active:
                same_file = (
                    str(active.get("filename") or "") == str(file_name)
                    and int(active.get("total_size") or -1) == int(total_size)
                )
                if same_file:
                    return active
                raise ConflictError("Another upload session is already active for this dataset")

            session_id = uuid.uuid4().hex
            now = _now_utc()
            expires = now + timedelta(hours=int(settings.upload_session_ttl_hours))
            payload = {
                "session_id": session_id,
                "dataset_id": int(dataset_id),
                "filename": file_name,
                "total_size": int(total_size),
                "chunk_size": int(normalized_chunk),
                "total_parts": int(total_parts),
                "uploaded_parts": [],
                "status": "active",
                "job_id": None,
                "created_at": _to_iso(now),
                "updated_at": _to_iso(now),
                "expires_at": _to_iso(expires),
            }
            self._parts_dir(session_id).mkdir(parents=True, exist_ok=True)
            self._write_json(self._session_meta_path(session_id), payload)
            return payload

    def get_session_status(self, db: Session, dataset_id: int, session_id: str) -> dict[str, Any]:
        DatasetService().get_dataset(db, int(dataset_id))
        meta = self._read_json(self._session_meta_path(session_id))
        if not meta:
            raise NotFoundError("Upload session not found")
        if int(meta.get("dataset_id") or -1) != int(dataset_id):
            raise NotFoundError("Upload session not found")

        uploaded_parts = self._coerce_uploaded_parts(meta)
        meta["uploaded_parts"] = uploaded_parts
        return {
            "session_id": str(meta.get("session_id") or session_id),
            "dataset_id": int(meta.get("dataset_id") or dataset_id),
            "filename": str(meta.get("filename") or ""),
            "total_size": int(meta.get("total_size") or 0),
            "chunk_size": int(meta.get("chunk_size") or 0),
            "total_parts": int(meta.get("total_parts") or 0),
            "uploaded_parts": int(len(uploaded_parts)),
            "status": str(meta.get("status") or "unknown"),
            "expires_at": _from_iso(str(meta.get("expires_at") or "")) or _now_utc(),
            "job_id": str(meta.get("job_id") or "") or None,
        }

    def upload_part(self, db: Session, dataset_id: int, session_id: str, *, part_no: int, upload: UploadFile) -> dict[str, Any]:
        DatasetService().get_dataset(db, int(dataset_id))
        with self._lock:
            meta = self._read_json(self._session_meta_path(session_id))
            if not meta:
                raise NotFoundError("Upload session not found")
            if int(meta.get("dataset_id") or -1) != int(dataset_id):
                raise NotFoundError("Upload session not found")

            expires = _from_iso(str(meta.get("expires_at") or ""))
            if expires is not None and expires < _now_utc():
                meta["status"] = "expired"
                meta["updated_at"] = _to_iso(_now_utc())
                self._write_json(self._session_meta_path(session_id), meta)
                raise ConflictError("Upload session expired")

            total_parts = int(meta.get("total_parts") or 0)
            pno = int(part_no or 0)
            if pno < 1 or pno > total_parts:
                raise ValidationError("part_no out of range")
            status = str(meta.get("status") or "").strip().lower()
            if status not in {"active", "uploading"}:
                raise ConflictError(f"Upload session status is {status}; cannot upload parts")

            part_path = (self._parts_dir(session_id) / f"{pno:08d}.part").resolve(strict=False)
            part_path.parent.mkdir(parents=True, exist_ok=True)
            with open(part_path, "wb") as f:
                shutil.copyfileobj(upload.file, f, length=1024 * 1024)

            uploaded_parts = self._coerce_uploaded_parts(meta)
            uploaded_parts.append(pno)
            uploaded_parts = sorted(set(uploaded_parts))
            meta["uploaded_parts"] = uploaded_parts
            meta["status"] = "uploading"
            meta["updated_at"] = _to_iso(_now_utc())
            self._write_json(self._session_meta_path(session_id), meta)

            return {
                "session_id": str(meta.get("session_id") or session_id),
                "part_no": pno,
                "uploaded_parts": int(len(uploaded_parts)),
                "total_parts": int(total_parts),
                "status": str(meta.get("status") or "uploading"),
            }

    def complete_session(self, db: Session, dataset_id: int, session_id: str) -> dict[str, Any]:
        DatasetService().get_dataset(db, int(dataset_id))
        with self._lock:
            meta = self._read_json(self._session_meta_path(session_id))
            if not meta:
                raise NotFoundError("Upload session not found")
            if int(meta.get("dataset_id") or -1) != int(dataset_id):
                raise NotFoundError("Upload session not found")

            status = str(meta.get("status") or "").strip().lower()
            if status in {"completed", "completing"} and meta.get("job_id"):
                return {
                    "session_id": str(meta.get("session_id") or session_id),
                    "job_id": str(meta.get("job_id")),
                    "status": "queued",
                }
            if status in {"failed", "expired", "cancelled"}:
                raise ConflictError(f"Upload session status is {status}; cannot complete")

            total_parts = int(meta.get("total_parts") or 0)
            uploaded_parts = self._coerce_uploaded_parts(meta)
            if len(uploaded_parts) != total_parts:
                raise ValidationError("PART_MISSING: not all parts were uploaded")

            job_id = uuid.uuid4().hex
            now = _now_utc()
            job_payload = {
                "job_id": job_id,
                "dataset_id": int(dataset_id),
                "session_id": str(session_id),
                "status": "queued",
                "phase": "queued",
                "progress": 0,
                "seq": 1,
                "updated_at": _to_iso(now),
                "output_version_id": None,
                "error_code": None,
                "error_message": None,
                "error_hint": None,
            }
            self._write_json(self._job_meta_path(job_id), job_payload)

            meta["status"] = "completing"
            meta["job_id"] = job_id
            meta["updated_at"] = _to_iso(now)
            self._write_json(self._session_meta_path(session_id), meta)

            worker = threading.Thread(target=self._run_import_job, args=(job_id,), daemon=True, name=f"upload-job-{job_id[:8]}")
            self._thread_by_job[job_id] = worker
            worker.start()

            return {"session_id": str(session_id), "job_id": job_id, "status": "queued"}

    def cancel_session(self, db: Session, dataset_id: int, session_id: str) -> None:
        DatasetService().get_dataset(db, int(dataset_id))
        with self._lock:
            meta = self._read_json(self._session_meta_path(session_id))
            if not meta:
                raise NotFoundError("Upload session not found")
            if int(meta.get("dataset_id") or -1) != int(dataset_id):
                raise NotFoundError("Upload session not found")
            status = str(meta.get("status") or "").strip().lower()
            if status in {"completed", "failed"}:
                raise ConflictError(f"Upload session status is {status}; cannot cancel")
            meta["status"] = "cancelled"
            meta["updated_at"] = _to_iso(_now_utc())
            self._write_json(self._session_meta_path(session_id), meta)
            self._cleanup_session_files(session_id, remove_meta=False)

    def get_import_job(self, db: Session, dataset_id: int, job_id: str) -> dict[str, Any]:
        DatasetService().get_dataset(db, int(dataset_id))
        payload = self._read_json(self._job_meta_path(job_id))
        if not payload:
            raise NotFoundError("Import job not found")
        if int(payload.get("dataset_id") or -1) != int(dataset_id):
            raise NotFoundError("Import job not found")
        return {
            "job_id": str(payload.get("job_id") or job_id),
            "dataset_id": int(payload.get("dataset_id") or dataset_id),
            "session_id": str(payload.get("session_id") or ""),
            "status": str(payload.get("status") or "unknown"),
            "phase": str(payload.get("phase") or "unknown"),
            "progress": int(payload.get("progress") or 0),
            "seq": int(payload.get("seq") or 0),
            "updated_at": _from_iso(str(payload.get("updated_at") or "")) or _now_utc(),
            "output_version_id": int(payload["output_version_id"]) if payload.get("output_version_id") is not None else None,
            "error_code": str(payload.get("error_code") or "") or None,
            "error_message": str(payload.get("error_message") or "") or None,
            "error_hint": str(payload.get("error_hint") or "") or None,
        }

    def _update_job(self, job_id: str, **patch: Any) -> dict[str, Any]:
        with self._lock:
            payload = self._read_json(self._job_meta_path(job_id))
            if payload is None:
                payload = {"job_id": job_id, "seq": 0}
            payload.update(patch)
            payload["seq"] = int(payload.get("seq") or 0) + 1
            payload["updated_at"] = _to_iso(_now_utc())
            self._write_json(self._job_meta_path(job_id), payload)
            return payload

    def _map_error(self, message: str) -> tuple[str, str]:
        text = str(message or "").strip()
        low = text.lower()
        if "part_missing" in low or "not all parts were uploaded" in low:
            return "PART_MISSING", "上传分片不完整，请继续上传缺失分片后重试。"
        if "unsupported file format" in low:
            return "ARCHIVE_CORRUPTED", "压缩包格式不支持或已损坏，请重新打包后上传。"
        if "is not a zip file" in low or "badzipfile" in low:
            return "ARCHIVE_CORRUPTED", "压缩包已损坏，请重新打包后上传。"
        if "no image files found" in low:
            return "NO_IMAGES", "未检测到图片文件，请确认压缩包中包含图片。"
        if "no label files found" in low:
            return "NO_LABELS", "未检测到标注文件，请确认包含 YOLO txt 或 LabelMe JSON。"
        if "coco json" in low or "invalid json" in low:
            return "COCO_JSON_INVALID", "标注 JSON 无法解析，请检查 JSON 文件格式与编码。"
        if "class" in low and "compatible" in low:
            return "CLASS_MISMATCH", "类别定义不兼容，请检查 classnames/data.yaml 与现有数据集。"
        if "timeout" in low:
            return "SERVER_TIMEOUT", "服务器处理超时，请重试或拆分更小数据包。"
        return "PARSE_FAILED", "服务器解析失败，请检查压缩包结构和标注内容。"

    def _cleanup_session_files(self, session_id: str, *, remove_meta: bool = False) -> None:
        session_dir = self._session_dir(session_id)
        for name in ("parts", "archive"):
            p = session_dir / name
            if p.exists():
                try:
                    shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
        if remove_meta:
            try:
                meta_path = self._session_meta_path(session_id)
                if meta_path.exists():
                    meta_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _run_import_job(self, job_id: str) -> None:
        payload = self._read_json(self._job_meta_path(job_id))
        if not payload:
            return
        session_id = str(payload.get("session_id") or "").strip()
        dataset_id = int(payload.get("dataset_id") or 0)
        if not session_id or dataset_id <= 0:
            self._update_job(job_id, status="failed", phase="failed", progress=100, error_code="PARSE_FAILED", error_message="Invalid job payload", error_hint="任务数据异常，请重新发起上传。")
            return

        try:
            self._update_job(job_id, status="running", phase="merging", progress=5)
            session_meta = self._read_json(self._session_meta_path(session_id))
            if not session_meta:
                raise NotFoundError("Upload session not found")

            total_parts = int(session_meta.get("total_parts") or 0)
            part_dir = self._parts_dir(session_id)
            archive_dir = self._archive_dir(session_id)
            archive_dir.mkdir(parents=True, exist_ok=True)
            merged_name = Path(str(session_meta.get("filename") or "upload.zip")).name
            merged_path = (archive_dir / merged_name).resolve(strict=False)

            with open(merged_path, "wb") as out:
                for i in range(1, total_parts + 1):
                    part_path = (part_dir / f"{i:08d}.part").resolve(strict=False)
                    if not part_path.exists():
                        raise ValidationError(f"PART_MISSING: missing part {i}")
                    with open(part_path, "rb") as src:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
                    if i == max(1, total_parts // 2):
                        self._update_job(job_id, phase="merging", progress=25)

            self._update_job(job_id, phase="extracting", progress=40)
            with open(merged_path, "rb") as f:
                upload_obj = SimpleNamespace(file=f, filename=merged_name)
                db = SessionLocal()
                try:
                    _ds, ver = DatasetService().append_dataset_archive(
                        db,
                        int(dataset_id),
                        file=upload_obj,
                        message="upload_session",
                        created_by="upload_session",
                        create_version=True,
                        activate=True,
                    )
                    out_version_id = int(ver.version_id) if ver is not None else None
                finally:
                    db.close()

            self._update_job(job_id, status="completed", phase="done", progress=100, output_version_id=out_version_id, error_code=None, error_message=None, error_hint=None)
            with self._lock:
                session_meta = self._read_json(self._session_meta_path(session_id)) or {}
                session_meta["status"] = "completed"
                session_meta["updated_at"] = _to_iso(_now_utc())
                self._write_json(self._session_meta_path(session_id), session_meta)
                self._cleanup_session_files(session_id, remove_meta=False)
        except Exception as e:
            code, hint = self._map_error(str(e))
            self._update_job(
                job_id,
                status="failed",
                phase="failed",
                progress=100,
                error_code=code,
                error_message=str(e),
                error_hint=hint,
            )
            with self._lock:
                session_meta = self._read_json(self._session_meta_path(session_id)) or {}
                session_meta["status"] = "failed"
                session_meta["updated_at"] = _to_iso(_now_utc())
                self._write_json(self._session_meta_path(session_id), session_meta)
        finally:
            with self._lock:
                self._thread_by_job.pop(job_id, None)
