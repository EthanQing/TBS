from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from train_platform.core.config import settings
from train_platform.utils.exceptions import ValidationError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _job_dir(job_id: str) -> Path:
    settings.ensure_dirs()
    root = settings.temp_dir / "model_conversions" / str(job_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _read_status(job_id: str) -> Dict[str, Any]:
    path = _status_path(job_id)
    if not path.exists():
        raise ValidationError("Job not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("status.json is not a dict")
        return data
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to read job status: {type(e).__name__}: {e}") from e


def _write_status(job_id: str, data: Dict[str, Any]) -> None:
    path = _status_path(job_id)
    tmp = path.with_suffix(".json.tmp")
    data = dict(data or {})
    data["updated_at"] = _utcnow().isoformat()
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _append_log(data: Dict[str, Any], msg: str) -> None:
    logs = data.get("logs")
    if not isinstance(logs, list):
        logs = []
    logs.append(str(msg))
    if len(logs) > 400:
        logs = logs[-400:]
    data["logs"] = logs


def _bytes_to_mb(n: int | float | None) -> float | None:
    try:
        if n is None:
            return None
        v = float(n)
        if v < 0:
            return None
        return round(v / (1024 * 1024), 2)
    except Exception:
        return None


def _file_size_mb(path: Path) -> float | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        return _bytes_to_mb(int(path.stat().st_size))
    except Exception:
        return None
