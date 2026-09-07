from __future__ import annotations

import enum
from typing import Any


class JobStatus(str, enum.Enum):
    """Persisted lifecycle states shared by the supported job consumers."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_STATUSES = frozenset({JobStatus.QUEUED, JobStatus.RUNNING})
TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


def coerce_status(value: Any) -> JobStatus:
    if isinstance(value, JobStatus):
        return value
    try:
        return JobStatus(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid job status: {value!r}") from exc


def is_active_status(value: Any) -> bool:
    try:
        return coerce_status(value) in ACTIVE_STATUSES
    except ValueError:
        return False


def is_running_status(value: Any) -> bool:
    try:
        return coerce_status(value) is JobStatus.RUNNING
    except ValueError:
        return False


def is_terminal_status(value: Any) -> bool:
    try:
        return coerce_status(value) in TERMINAL_STATUSES
    except ValueError:
        return False


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "JobStatus",
    "coerce_status",
    "is_active_status",
    "is_running_status",
    "is_terminal_status",
]
