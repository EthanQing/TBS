from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psutil

from train_platform.platform.runtime.gpu_probe import probe_gpus


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_gpu_metric(
    *,
    gpu_index: int,
    name: str,
    uuid: str | None = None,
    utilization_percent: Any = None,
    memory_used_mb: Any = None,
    memory_total_mb: Any = None,
) -> dict[str, Any]:
    gpu_util = _to_float(utilization_percent)
    memory_used = _to_float(memory_used_mb)
    memory_total = _to_float(memory_total_mb)
    memory_percent = None
    if memory_used is not None and memory_total is not None and memory_total > 0:
        memory_percent = (memory_used / memory_total) * 100.0
    return {
        "gpu_index": int(gpu_index),
        "name": str(name or f"GPU {gpu_index}"),
        "uuid": _to_text(uuid),
        "utilization_percent": gpu_util,
        "memory_used_mb": memory_used,
        "memory_total_mb": memory_total,
        "memory_percent": memory_percent,
    }


def get_gpu_device_metrics() -> list[dict[str, Any]]:
    result = probe_gpus()
    return [
        _build_gpu_metric(
            gpu_index=device.observed_index,
            name=device.name,
            uuid=device.gpu_uuid,
            utilization_percent=device.utilization_percent,
            memory_used_mb=device.memory_used_mib,
            memory_total_mb=device.memory_total_mib,
        )
        for device in result.devices
        if device.observed_index is not None
    ]


def collect_system_snapshot(
    node_id: str = "backend",
    node_type: str = "backend",
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    cpu_percent = float(psutil.cpu_percent(interval=None))
    memory = psutil.virtual_memory()
    memory_percent = float(memory.percent)
    memory_used_mb = float(memory.used) / 1024.0 / 1024.0
    memory_total_mb = float(memory.total) / 1024.0 / 1024.0

    gpus = get_gpu_device_metrics()
    gpu_count = len(gpus)
    gpu_percent = None
    gpu_used_mb = None
    gpu_total_mb = None
    if gpus:
        utilization_values = [float(item["utilization_percent"]) for item in gpus if item.get("utilization_percent") is not None]
        memory_percent_values = [float(item["memory_percent"]) for item in gpus if item.get("memory_percent") is not None]
        used_values = [float(item["memory_used_mb"]) for item in gpus if item.get("memory_used_mb") is not None]
        total_values = [float(item["memory_total_mb"]) for item in gpus if item.get("memory_total_mb") is not None]
        if utilization_values:
            gpu_percent = sum(utilization_values) / float(len(utilization_values))
        elif memory_percent_values:
            gpu_percent = sum(memory_percent_values) / float(len(memory_percent_values))
        if used_values:
            gpu_used_mb = sum(used_values)
        if total_values:
            gpu_total_mb = sum(total_values)

    return {
        "timestamp": timestamp,
        "node_id": str(node_id or "backend"),
        "node_type": str(node_type or "backend"),
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "gpu_available": bool(gpu_count),
        "gpu_count": gpu_count,
        "gpu_percent": gpu_percent,
        "gpu_used_mb": gpu_used_mb,
        "gpu_total_mb": gpu_total_mb,
        "gpus": gpus,
    }
