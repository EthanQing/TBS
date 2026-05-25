from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from train_platform.core.config import settings
from train_platform.utils.model_conversion_jobs import _append_log, _read_status, _write_status


def _jobs_root() -> Path:
    root = settings.temp_dir / "model_conversions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _status_files() -> list[Path]:
    root = _jobs_root()
    try:
        return sorted(root.glob("*/status.json"), key=lambda p: p.stat().st_mtime)
    except Exception:
        return []


def _job_id_from_status(path: Path) -> str:
    return str(path.parent.name)


def _lock_path(job_id: str) -> Path:
    return _jobs_root() / str(job_id) / "worker.lock"


def _is_stale_lock(path: Path, *, stale_seconds: int) -> bool:
    try:
        return path.exists() and (time.time() - float(path.stat().st_mtime)) > float(stale_seconds)
    except Exception:
        return False


def _try_claim(job_id: str, worker_id: str, *, stale_seconds: int) -> bool:
    lock = _lock_path(job_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if _is_stale_lock(lock, stale_seconds=stale_seconds):
        try:
            lock.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{worker_id}\n{time.time()}\n")
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _release_claim(job_id: str) -> None:
    try:
        _lock_path(job_id).unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass


def _queued_job_from_status(path: Path) -> tuple[str, Dict[str, Any]] | None:
    job_id = _job_id_from_status(path)
    try:
        data = _read_status(job_id)
    except Exception:
        return None
    if str(data.get("status") or "").strip().lower() != "queued":
        return None
    return job_id, data


class ModelConversionQueueWorker:
    def __init__(self, *, worker_id: str, stale_lock_seconds: Optional[int] = None) -> None:
        self.worker_id = str(worker_id)
        self.stale_lock_seconds = int(stale_lock_seconds or os.getenv("MODEL_CONVERSION_STALE_LOCK_SECONDS", "1800"))

    def tick(self) -> bool:
        for status_path in _status_files():
            queued = _queued_job_from_status(status_path)
            if queued is None:
                continue
            job_id, data = queued
            if not _try_claim(job_id, self.worker_id, stale_seconds=self.stale_lock_seconds):
                continue

            try:
                current = _read_status(job_id)
                if str(current.get("status") or "").strip().lower() != "queued":
                    return False

                current["worker_id"] = self.worker_id
                _append_log(current, f"YOLO worker 已领取任务: {self.worker_id}")
                _write_status(job_id, current)

                from train_platform.workers.model_conversion_task import _run_pt_to_onnx

                opset = current.get("opset")
                dynamic = current.get("dynamic")
                _run_pt_to_onnx(
                    job_id,
                    opset=int(opset) if opset is not None else None,
                    dynamic=bool(dynamic) if dynamic is not None else True,
                )
                return True
            except Exception as e:
                try:
                    failed = _read_status(job_id)
                    failed["status"] = "failed"
                    failed["progress"] = 100
                    failed["error_message"] = f"{type(e).__name__}: {e}"
                    _append_log(failed, failed["error_message"])
                    _write_status(job_id, failed)
                except Exception:
                    pass
                return True
            finally:
                _release_claim(job_id)
        return False
