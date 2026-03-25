from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import psutil
import pynvml


logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_SECONDS = max(60, int(os.getenv("SYSTEM_METRICS_RETENTION_SECONDS", "86400")))
_DEFAULT_MAX_POINTS = max(100, int(os.getenv("SYSTEM_METRICS_MAX_POINTS", "5000")))
_DEFAULT_STEP_SECONDS = max(1, int(os.getenv("SYSTEM_METRICS_STEP_SECONDS", "5")))


class SystemMetricsService:
    _history_lock = Lock()
    _history_by_node: dict[str, deque[dict[str, Any]]] = {}

    @staticmethod
    def _to_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def get_gpu_device_metrics() -> list[dict[str, Any]]:
        gpu_metrics_list: list[dict[str, Any]] = []
        try:
            pynvml.nvmlInit()
            gpu_count = int(pynvml.nvmlDeviceGetCount())
            for gpu_index in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                name = SystemMetricsService._to_text(pynvml.nvmlDeviceGetName(handle)) or "N/A"
                uuid = SystemMetricsService._to_text(pynvml.nvmlDeviceGetUUID(handle))

                utilization_percent = None
                memory_used_mb = None
                memory_total_mb = None
                memory_percent = None

                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_percent = float(utilization.gpu)
                except Exception:
                    utilization_percent = None

                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_mb = float(memory_info.used) / 1024.0 / 1024.0
                    memory_total_mb = float(memory_info.total) / 1024.0 / 1024.0
                    memory_percent = (
                        (float(memory_info.used) / float(memory_info.total) * 100.0)
                        if float(memory_info.total) > 0
                        else None
                    )
                except Exception:
                    memory_used_mb = None
                    memory_total_mb = None
                    memory_percent = None

                gpu_metrics_list.append(
                    {
                        "gpu_index": gpu_index,
                        "name": name,
                        "uuid": uuid,
                        "utilization_percent": utilization_percent,
                        "memory_used_mb": memory_used_mb,
                        "memory_total_mb": memory_total_mb,
                        "memory_percent": memory_percent,
                    }
                )
        except Exception as e:
            logger.warning("Error occurred while fetching GPU metrics: %s", e)
            return []
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return gpu_metrics_list

    @staticmethod
    def get_gpu_count() -> int:
        return len(SystemMetricsService.get_gpu_device_metrics())

    @staticmethod
    def is_gpu_available() -> bool:
        return SystemMetricsService.get_gpu_count() > 0

    @classmethod
    def _make_snapshot(
        cls,
        *,
        node_id: str = "backend",
        node_type: str = "backend",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        cpu_percent = float(psutil.cpu_percent(interval=None))

        mem = psutil.virtual_memory()
        memory_percent = float(mem.percent)
        memory_used_mb = float(mem.used) / 1024.0 / 1024.0
        memory_total_mb = float(mem.total) / 1024.0 / 1024.0

        gpus = cls.get_gpu_device_metrics()
        gpu_count = len(gpus)
        gpu_available = gpu_count > 0

        gpu_percent = None
        gpu_used_mb = None
        gpu_total_mb = None
        if gpus:
            gpu_utils = [float(x["utilization_percent"]) for x in gpus if x.get("utilization_percent") is not None]
            gpu_used = [float(x["memory_used_mb"]) for x in gpus if x.get("memory_used_mb") is not None]
            gpu_total = [float(x["memory_total_mb"]) for x in gpus if x.get("memory_total_mb") is not None]

            if gpu_utils:
                gpu_percent = sum(gpu_utils) / float(len(gpu_utils))
            if gpu_used:
                gpu_used_mb = sum(gpu_used)
            if gpu_total:
                gpu_total_mb = sum(gpu_total)

        return {
            "timestamp": now,
            "node_id": str(node_id or "backend"),
            "node_type": str(node_type or "backend"),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_percent": gpu_percent,
            "gpu_used_mb": gpu_used_mb,
            "gpu_total_mb": gpu_total_mb,
            "gpus": gpus,
        }

    @classmethod
    def _append_history(cls, snapshot: dict[str, Any]) -> None:
        node_id = str(snapshot.get("node_id") or "backend")
        ts = snapshot.get("timestamp")
        if not isinstance(ts, datetime):
            return

        cutoff = ts - timedelta(seconds=_DEFAULT_RETENTION_SECONDS)
        with cls._history_lock:
            if node_id not in cls._history_by_node:
                cls._history_by_node[node_id] = deque(maxlen=_DEFAULT_MAX_POINTS)
            dq = cls._history_by_node[node_id]
            dq.append(snapshot)
            while dq and isinstance(dq[0].get("timestamp"), datetime) and dq[0]["timestamp"] < cutoff:
                dq.popleft()

    @classmethod
    def get_system_metrics(
        cls,
        node_id: str = "backend",
        node_type: str = "backend",
    ) -> dict[str, Any]:
        snapshot = cls._make_snapshot(node_id=node_id, node_type=node_type)
        cls._append_history(snapshot)
        return snapshot

    @classmethod
    def get_system_metrics_history(
        cls,
        minutes: int = 10,
        node_id: str = "backend",
        node_type: str = "backend",
        step_seconds: int = _DEFAULT_STEP_SECONDS,
    ) -> dict[str, Any]:
        minutes = max(1, int(minutes))
        step_seconds = max(1, int(step_seconds))
        node_id = str(node_id or "backend")
        window_seconds = int(minutes * 60)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        with cls._history_lock:
            source = list(cls._history_by_node.get(node_id, ()))

        points = [x for x in source if isinstance(x.get("timestamp"), datetime) and x["timestamp"] >= cutoff]
        if not points:
            points = [cls.get_system_metrics(node_id=node_id, node_type=node_type)]

        points.sort(key=lambda x: x["timestamp"])
        sampled: list[dict[str, Any]] = []
        next_allowed: datetime | None = None
        for p in points:
            ts = p["timestamp"]
            if next_allowed is None or ts >= next_allowed:
                sampled.append(p)
                next_allowed = ts + timedelta(seconds=step_seconds)

        history_points = [
            {
                "timestamp": p.get("timestamp"),
                "cpu_percent": p.get("cpu_percent"),
                "memory_percent": p.get("memory_percent"),
                "memory_used_mb": p.get("memory_used_mb"),
                "memory_total_mb": p.get("memory_total_mb"),
                "gpu_available": p.get("gpu_available"),
                "gpu_count": p.get("gpu_count"),
                "gpu_percent": p.get("gpu_percent"),
                "gpu_used_mb": p.get("gpu_used_mb"),
                "gpu_total_mb": p.get("gpu_total_mb"),
                "gpus": p.get("gpus", []),
            }
            for p in sampled
        ]

        return {
            "node_id": node_id,
            "node_type": str(node_type or "backend"),
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "points": history_points,
        }

    @classmethod
    def get_cluster_overview(cls) -> dict[str, Any]:
        nodes = [
            cls.get_system_metrics(node_id="backend", node_type="backend"),
        ]

        cpu_values = [float(x["cpu_percent"]) for x in nodes if x.get("cpu_percent") is not None]
        mem_values = [float(x["memory_percent"]) for x in nodes if x.get("memory_percent") is not None]
        gpu_values = [float(x["gpu_percent"]) for x in nodes if x.get("gpu_percent") is not None]

        return {
            "timestamp": datetime.now(timezone.utc),
            "total_nodes": len(nodes),
            "online_nodes": len(nodes),
            "cpu_percent_avg": (sum(cpu_values) / len(cpu_values)) if cpu_values else None,
            "memory_percent_avg": (sum(mem_values) / len(mem_values)) if mem_values else None,
            "gpu_percent_avg": (sum(gpu_values) / len(gpu_values)) if gpu_values else None,
            "nodes": nodes,
        }
