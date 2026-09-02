from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import psutil

from .status import JobStatus, coerce_status, is_terminal_status


class JobStoreError(ValueError):
    """Base error for invalid or unavailable filesystem job state."""


class JobNotFoundError(JobStoreError):
    pass


class MalformedJobStatusError(JobStoreError):
    pass


class MalformedJobResultError(JobStoreError):
    pass


class _JobLockTimeoutError(JobStoreError):
    pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        return bool(psutil.pid_exists(pid))
    except (OSError, OverflowError, ValueError):
        return False


class _JobFileLock:
    """Small cross-process lock using exclusive creation and stale-owner recovery."""

    def __init__(self, path: Path, *, timeout: float, stale_after: float) -> None:
        self.path = path
        self.timeout = max(0.1, float(timeout))
        self.stale_after = max(self.timeout, float(stale_after))
        self._owner = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"

    def __enter__(self) -> "_JobFileLock":
        deadline = time.monotonic() + self.timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n{self._owner}\n{time.time()}\n")
                return self
            except FileExistsError:
                if self._remove_stale_owner():
                    continue
                if time.monotonic() >= deadline:
                    raise _JobLockTimeoutError(f"Timed out acquiring job lock: {self.path}")
                time.sleep(0.025)

    def _remove_stale_owner(self) -> bool:
        try:
            stat = self.path.stat()
            age = max(0.0, time.time() - stat.st_mtime)
            if age <= self.stale_after:
                return False
            raw = self.path.read_text(encoding="utf-8").splitlines()
            pid = int(raw[0]) if raw else 0
            if pid and _pid_is_alive(pid):
                return False
            self.path.unlink(missing_ok=True)
            return True
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8").splitlines()
            if len(raw) > 1 and raw[1] == self._owner:
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, OSError):
            pass


class JobStore:
    """Concrete persistence boundary for inference and evaluation jobs."""

    def __init__(
        self,
        root: Path,
        *,
        lock_timeout: float = 30.0,
        lock_stale_after: float = 60.0,
    ) -> None:
        self.root = Path(root)
        self.lock_timeout = max(0.1, float(lock_timeout))
        self.lock_stale_after = max(self.lock_timeout, float(lock_stale_after))

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def job_dir(self, job_id: str, *, create: bool = False) -> Path:
        path = self.root / str(job_id)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def results_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "results.jsonl"

    def create(self, job_id: str, status: Mapping[str, Any]) -> Dict[str, Any]:
        job_dir = self.job_dir(job_id, create=True)
        path = job_dir / "status.json"
        with self._lock(job_id):
            if path.exists():
                raise JobStoreError(f"Job already exists: {job_id}")
            payload = dict(status)
            self._validate_status(payload)
            return self._write_status_unlocked(path, payload)

    def read_status(self, job_id: str) -> Dict[str, Any]:
        path = self.status_path(job_id)
        self._require_status(path, job_id)
        with self._lock(job_id):
            return self._read_status_unlocked(path, job_id)

    def update(
        self,
        job_id: str,
        patch: Mapping[str, Any],
        *,
        bump_seq: bool = True,
    ) -> Dict[str, Any]:
        path = self.status_path(job_id)
        self._require_status(path, job_id)
        with self._lock(job_id):
            current = self._read_status_unlocked(path, job_id)
            if is_terminal_status(current["status"]):
                return current
            current.update(dict(patch))
            self._normalize_status_fields(current)
            if bump_seq:
                current["seq"] = int(current.get("seq") or 0) + 1
            self._write_status_unlocked(path, current)
            return current

    def cancel(
        self,
        job_id: str,
        *,
        terminal_if: Sequence[JobStatus],
        terminal_patch: Mapping[str, Any],
    ) -> Dict[str, Any]:
        path = self.status_path(job_id)
        terminal_states = {coerce_status(value) for value in terminal_if}
        self._require_status(path, job_id)
        with self._lock(job_id):
            current = self._read_status_unlocked(path, job_id)
            if is_terminal_status(current["status"]):
                return current
            current["cancel_requested"] = True
            if coerce_status(current["status"]) in terminal_states:
                current.update(dict(terminal_patch))
                current["status"] = JobStatus.CANCELLED.value
                current["cancel_requested"] = True
            self._normalize_status_fields(current)
            current["seq"] = int(current.get("seq") or 0) + 1
            self._write_status_unlocked(path, current)
            return current

    def append_result(self, job_id: str, item: Mapping[str, Any]) -> Dict[str, Any]:
        path = self.status_path(job_id)
        results_path = self.results_path(job_id)
        self._require_status(path, job_id)
        with self._lock(job_id):
            status = self._read_status_unlocked(path, job_id)
            row = dict(item)
            if is_terminal_status(status["status"]):
                return row

            last_result_id = max(
                int(status.get("last_result_id") or 0),
                self._last_result_id_unlocked(results_path),
            )
            result_id = last_result_id + 1
            row["result_id"] = result_id
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            status["last_result_id"] = result_id
            status["seq"] = int(status.get("seq") or 0) + 1
            self._write_status_unlocked(path, status)
            return row

    def read_results_since(self, job_id: str, after_result_id: int = 0) -> list[Dict[str, Any]]:
        status_path = self.status_path(job_id)
        results_path = self.results_path(job_id)
        last = int(after_result_id or 0)
        self._require_status(status_path, job_id)
        with self._lock(job_id):
            self._read_status_unlocked(status_path, job_id)
            if not results_path.exists():
                return []
            rows: list[Dict[str, Any]] = []
            try:
                lines = results_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise MalformedJobResultError(f"Failed to read job results: {type(exc).__name__}: {exc}") from exc
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise MalformedJobResultError(
                        f"Malformed job result at line {line_number}: {type(exc).__name__}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise MalformedJobResultError(f"Invalid job result payload at line {line_number}")
                try:
                    result_id = int(row["result_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise MalformedJobResultError(
                        f"Job result is missing a valid result_id at line {line_number}"
                    ) from exc
                if result_id > last:
                    rows.append(row)
            rows.sort(key=lambda row: int(row["result_id"]))
            return rows

    def list_statuses(self) -> list[Dict[str, Any]]:
        self.ensure_root()
        statuses: list[Dict[str, Any]] = []
        for path in sorted(self.root.glob("*/status.json")):
            statuses.append(self.read_status(path.parent.name))
        return statuses

    @contextmanager
    def _lock(self, job_id: str) -> Iterator[_JobFileLock]:
        lock = _JobFileLock(
            self.job_dir(job_id) / ".job.lock",
            timeout=self.lock_timeout,
            stale_after=self.lock_stale_after,
        )
        with lock:
            yield lock

    def _read_status_unlocked(self, path: Path, job_id: str) -> Dict[str, Any]:
        if not path.exists():
            raise JobNotFoundError(f"Job not found: {job_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MalformedJobStatusError(
                f"Malformed job status: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise MalformedJobStatusError("Invalid job status payload")
        self._validate_status(data)
        return data

    @staticmethod
    def _validate_status(data: Mapping[str, Any]) -> None:
        if "status" not in data:
            raise MalformedJobStatusError("Job status is missing status")
        try:
            coerce_status(data["status"])
        except ValueError as exc:
            raise MalformedJobStatusError(str(exc)) from exc

    @staticmethod
    def _require_status(path: Path, job_id: str) -> None:
        if not path.exists():
            raise JobNotFoundError(f"Job not found: {job_id}")

    @staticmethod
    def _normalize_status_fields(data: Dict[str, Any]) -> None:
        try:
            data["status"] = coerce_status(data["status"]).value
        except (KeyError, ValueError) as exc:
            raise MalformedJobStatusError(str(exc)) from exc
        for field in ("progress", "processed", "total", "seq", "last_result_id"):
            if field in data:
                try:
                    data[field] = max(0, int(data[field] or 0))
                except (TypeError, ValueError) as exc:
                    raise MalformedJobStatusError(f"Invalid {field} in job status") from exc
        if "progress" in data:
            data["progress"] = min(100, data["progress"])

    def _write_status_unlocked(self, path: Path, data: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(data)
        self._normalize_status_fields(payload)
        payload["updated_at"] = _utcnow_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return payload

    def _last_result_id_unlocked(self, path: Path) -> int:
        if not path.exists():
            return 0
        last = 0
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise MalformedJobResultError(f"Failed to read job results: {type(exc).__name__}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result_id = int(row["result_id"])
            except Exception as exc:
                raise MalformedJobResultError(
                    f"Malformed job result at line {line_number}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(row, dict) or result_id <= 0:
                raise MalformedJobResultError(f"Invalid job result payload at line {line_number}")
            last = max(last, result_id)
        return last


__all__ = [
    "JobNotFoundError",
    "JobStore",
    "JobStoreError",
    "MalformedJobResultError",
    "MalformedJobStatusError",
]
