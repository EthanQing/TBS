from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Mapping

from train_platform.core.config import settings
from train_platform.platform.jobs import (
    JobNotFoundError,
    JobStatus,
    JobStore,
    JobStoreError,
)
from train_platform.utils.exceptions import ValidationError


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_MAX_LOGS = 400


def _jobs_root() -> Path:
    root = settings.temp_dir / "model_conversions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _store() -> JobStore:
    return JobStore(_jobs_root())


def _validate_job_id(job_id: str) -> str:
    value = str(job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(value):
        raise ValidationError("Invalid conversion job id")
    return value


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso() -> str:
    return _utcnow().isoformat()


def _append_log_value(logs: Any, message: str) -> list[str]:
    values = list(logs) if isinstance(logs, list) else []
    values.append(str(message))
    return values[-_MAX_LOGS:]


def _validate_request(source_format: Any, target_format: Any, filename: str) -> tuple[str, str, str]:
    source = str(source_format or "").strip().lower()
    target = str(target_format or "").strip().lower()
    if source not in {"pt", "pth"}:
        raise ValidationError("Only pt/pth is supported for now (YOLOv8)")
    if target != "onnx":
        raise ValidationError("Only onnx is supported for now (YOLOv8)")

    original_name = str(filename or "model.pt")
    suffix = Path(original_name).suffix.lower() or ".pt"
    if suffix not in {".pt", ".pth"}:
        raise ValidationError("Unsupported source model file type")
    return source, target, original_name


def create_job(
    source: BinaryIO,
    *,
    filename: str | None,
    source_format: str = "pt",
    target_format: str = "onnx",
    opset: int | None = None,
    dynamic: bool = True,
) -> Dict[str, Any]:
    """Persist an uploaded model and enqueue a conversion job."""

    source_name, target, original_name = _validate_request(
        source_format,
        target_format,
        filename or "model.pt",
    )
    try:
        normalized_opset = int(opset) if opset is not None else None
    except (TypeError, ValueError) as exc:
        raise ValidationError("Invalid opset") from exc
    job_id = uuid.uuid4().hex
    store = _store()
    job_root = store.job_dir(job_id, create=True)
    input_path = job_root / "input.pt"
    temporary_path = job_root / ".input.pt.tmp"

    try:
        with temporary_path.open("wb") as handle:
            shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, input_path)

        created_at = _to_iso()
        status: Dict[str, Any] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "progress": 0,
            "logs": _append_log_value(
                [f"已接收文件: {original_name}", f"source_format={source_name} target_format={target}"],
                "已加入 YOLO worker 转换队列",
            ),
            "source_format": source_name,
            "target_format": target,
            "opset": normalized_opset,
            "dynamic": bool(dynamic),
            "worker_id": None,
            "output_filename": None,
            "performance": None,
            "error_message": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        return store.create(job_id, status)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
            input_path.unlink(missing_ok=True)
            job_root.rmdir()
        except OSError:
            pass
        raise


def read_job(job_id: str) -> Dict[str, Any]:
    """Read a conversion status and translate storage errors for API callers."""

    normalized_id = _validate_job_id(job_id)
    try:
        return _store().read_status(normalized_id)
    except JobNotFoundError as exc:
        raise ValidationError("Job not found") from exc
    except JobStoreError as exc:
        raise ValidationError(f"Failed to read job status: {exc}") from exc


def update_job(job_id: str, patch: Mapping[str, Any], *, log: str | None = None) -> Dict[str, Any]:
    """Apply a status patch through the shared atomic JobStore."""

    normalized_id = _validate_job_id(job_id)
    update = dict(patch)
    if log is not None:
        current = read_job(normalized_id)
        update["logs"] = _append_log_value(current.get("logs"), log)
    try:
        return _store().update(normalized_id, update, bump_seq=True)
    except JobNotFoundError as exc:
        raise ValidationError("Job not found") from exc
    except JobStoreError as exc:
        raise ValidationError(f"Failed to update job status: {exc}") from exc


def input_path(job_id: str) -> Path:
    normalized_id = _validate_job_id(job_id)
    path = _store().job_dir(normalized_id) / "input.pt"
    _ensure_under_job_root(path, normalized_id)
    return path


def resolve_download_path(job_id: str) -> tuple[Path, str]:
    """Resolve the completed output artifact without storing route knowledge."""

    normalized_id = _validate_job_id(job_id)
    data = read_job(normalized_id)
    if str(data.get("status") or "").strip().lower() != JobStatus.COMPLETED.value:
        raise ValidationError("Conversion is not completed")
    filename = str(data.get("output_filename") or "output.onnx").strip() or "output.onnx"
    path = _store().job_dir(normalized_id) / filename
    _ensure_under_job_root(path, normalized_id)
    if not path.exists() or not path.is_file():
        raise ValidationError("Conversion output file not found")
    return path, filename


def _ensure_under_job_root(path: Path, job_id: str) -> None:
    root = _store().job_dir(_validate_job_id(job_id)).resolve(strict=False)
    candidate = path.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValidationError("Unsafe conversion artifact path") from exc


def enumerate_queued_jobs() -> Iterable[tuple[str, Dict[str, Any]]]:
    """Yield queued jobs while isolating malformed status files per job."""

    root = _jobs_root()
    try:
        status_paths = list(root.glob("*/status.json"))
    except OSError:
        return

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    status_paths.sort(key=_mtime)

    for status_path in status_paths:
        job_id = status_path.parent.name
        try:
            data = _store().read_status(job_id)
        except (JobNotFoundError, JobStoreError, OSError, ValueError):
            continue
        if str(data.get("status") or "").strip().lower() == JobStatus.QUEUED.value:
            yield job_id, data


def _claim_path(job_id: str) -> Path:
    return _store().job_dir(_validate_job_id(job_id)) / "worker.lock"


def _is_stale_claim(path: Path, stale_seconds: int) -> bool:
    try:
        return path.exists() and (time.time() - float(path.stat().st_mtime)) > float(stale_seconds)
    except (OSError, ValueError):
        return False


def claim_job(job_id: str, worker_id: str, *, stale_seconds: int) -> bool:
    """Atomically claim a conversion job using its conversion-owned lock."""

    lock = _claim_path(job_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if _is_stale_claim(lock, stale_seconds):
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{worker_id}\n{time.time()}\n")
        return True
    except (FileExistsError, OSError):
        return False


def mark_claimed(job_id: str, worker_id: str) -> bool:
    """Record the worker claim while the conversion lock is held."""

    try:
        current = read_job(job_id)
        if str(current.get("status") or "").strip().lower() != JobStatus.QUEUED.value:
            return False
        update_job(job_id, {"worker_id": str(worker_id)}, log=f"YOLO worker 已领取任务: {worker_id}")
        return True
    except ValidationError:
        return False


def release_claim(job_id: str) -> None:
    try:
        _claim_path(job_id).unlink(missing_ok=True)
    except OSError:
        pass


__all__ = [
    "claim_job",
    "create_job",
    "enumerate_queued_jobs",
    "input_path",
    "mark_claimed",
    "read_job",
    "release_claim",
    "resolve_download_path",
    "update_job",
]
