from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


RETENTION_SECONDS = max(60, _env_int("SYSTEM_METRICS_RETENTION_SECONDS", 86400))
MAX_POINTS = max(100, _env_int("SYSTEM_METRICS_MAX_POINTS", 5000))
DEFAULT_STEP_SECONDS = max(1, _env_int("SYSTEM_METRICS_STEP_SECONDS", 5))

_history_lock = Lock()
_history_by_node: dict[str, deque[dict[str, Any]]] = {}


def append_snapshot(snapshot: dict[str, Any]) -> None:
    node_id = str(snapshot.get("node_id") or "backend")
    timestamp = snapshot.get("timestamp")
    if not isinstance(timestamp, datetime):
        return
    cutoff = timestamp - timedelta(seconds=RETENTION_SECONDS)
    with _history_lock:
        history = _history_by_node.setdefault(node_id, deque(maxlen=MAX_POINTS))
        history.append(snapshot)
        while history and isinstance(history[0].get("timestamp"), datetime) and history[0]["timestamp"] < cutoff:
            history.popleft()


def query_history(*, node_id: str, window_seconds: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, int(window_seconds)))
    with _history_lock:
        source = list(_history_by_node.get(str(node_id), ()))
    return [
        point
        for point in source
        if isinstance(point.get("timestamp"), datetime) and point["timestamp"] >= cutoff
    ]


def downsample(points: list[dict[str, Any]], *, step_seconds: int) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    next_allowed: datetime | None = None
    step = max(1, int(step_seconds))
    for point in sorted(points, key=lambda item: item["timestamp"]):
        timestamp = point["timestamp"]
        if next_allowed is None or timestamp >= next_allowed:
            sampled.append(point)
            next_allowed = timestamp + timedelta(seconds=step)
    return sampled
