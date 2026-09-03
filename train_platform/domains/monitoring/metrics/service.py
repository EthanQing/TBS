from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .collector import collect_system_snapshot
from .history import DEFAULT_STEP_SECONDS, append_snapshot, downsample, query_history


def collect_current(node_id: str = "backend", node_type: str = "backend") -> dict[str, Any]:
    return collect_system_snapshot(node_id=node_id, node_type=node_type)


def record_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    append_snapshot(snapshot)
    return snapshot


def sample(node_id: str = "backend", node_type: str = "backend") -> dict[str, Any]:
    return record_snapshot(collect_current(node_id=node_id, node_type=node_type))


def _history_point(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": snapshot.get("timestamp"),
        "cpu_percent": snapshot.get("cpu_percent"),
        "memory_percent": snapshot.get("memory_percent"),
        "memory_used_mb": snapshot.get("memory_used_mb"),
        "memory_total_mb": snapshot.get("memory_total_mb"),
        "gpu_available": snapshot.get("gpu_available"),
        "gpu_count": snapshot.get("gpu_count"),
        "gpu_percent": snapshot.get("gpu_percent"),
        "gpu_used_mb": snapshot.get("gpu_used_mb"),
        "gpu_total_mb": snapshot.get("gpu_total_mb"),
        "gpus": snapshot.get("gpus", []),
    }


def get_system_metrics_history(
    minutes: int = 10,
    node_id: str = "backend",
    node_type: str = "backend",
    step_seconds: int = DEFAULT_STEP_SECONDS,
) -> dict[str, Any]:
    minutes = max(1, int(minutes))
    step_seconds = max(1, int(step_seconds))
    node_id = str(node_id or "backend")
    window_seconds = int(minutes * 60)
    points = query_history(node_id=node_id, window_seconds=window_seconds)
    if not points:
        points = [sample(node_id=node_id, node_type=node_type)]
    return {
        "node_id": node_id,
        "node_type": str(node_type or "backend"),
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "points": [_history_point(point) for point in downsample(points, step_seconds=step_seconds)],
    }


def get_cluster_overview() -> dict[str, Any]:
    nodes = [sample(node_id="backend", node_type="backend")]
    cpu_values = [float(node["cpu_percent"]) for node in nodes if node.get("cpu_percent") is not None]
    memory_values = [float(node["memory_percent"]) for node in nodes if node.get("memory_percent") is not None]
    gpu_values = [float(node["gpu_percent"]) for node in nodes if node.get("gpu_percent") is not None]
    return {
        "timestamp": datetime.now(timezone.utc),
        "total_nodes": len(nodes),
        "online_nodes": len(nodes),
        "cpu_percent_avg": sum(cpu_values) / len(cpu_values) if cpu_values else None,
        "memory_percent_avg": sum(memory_values) / len(memory_values) if memory_values else None,
        "gpu_percent_avg": sum(gpu_values) / len(gpu_values) if gpu_values else None,
        "nodes": nodes,
    }
