from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

import psutil

try:
    import pynvml
except Exception:
    pynvml = None


logger = logging.getLogger(__name__)


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


def _get_gpu_device_metrics_via_nvml() -> list[dict[str, Any]]:
    if pynvml is None:
        return []
    metrics: list[dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug("NVML init failed while fetching GPU metrics: %s", exc)
        return []

    try:
        gpu_count = int(pynvml.nvmlDeviceGetCount())
        for gpu_index in range(gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as exc:
                logger.debug("NVML get handle failed for GPU %s: %s", gpu_index, exc)
                continue

            name = f"GPU {gpu_index}"
            uuid = None
            utilization_percent = None
            memory_used_mb = None
            memory_total_mb = None
            try:
                name = _to_text(pynvml.nvmlDeviceGetName(handle)) or name
            except Exception:
                pass
            try:
                uuid = _to_text(pynvml.nvmlDeviceGetUUID(handle))
            except Exception:
                pass
            try:
                utilization_percent = getattr(pynvml.nvmlDeviceGetUtilizationRates(handle), "gpu", None)
            except Exception:
                pass
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = float(memory_info.used) / 1024.0 / 1024.0
                memory_total_mb = float(memory_info.total) / 1024.0 / 1024.0
            except Exception:
                pass
            metrics.append(
                _build_gpu_metric(
                    gpu_index=gpu_index,
                    name=name,
                    uuid=uuid,
                    utilization_percent=utilization_percent,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                )
            )
    except Exception as exc:
        logger.debug("NVML metrics collection failed: %s", exc)
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return metrics


def _get_gpu_device_metrics_via_nvidia_smi() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        logger.debug("nvidia-smi execution failed while fetching GPU metrics: %s", exc)
        return []
    if proc.returncode != 0:
        logger.debug("nvidia-smi returned non-zero exit code while fetching GPU metrics: %s", proc.stderr)
        return []

    metrics: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = next(csv.reader([text], skipinitialspace=True))
        except Exception:
            continue
        if len(row) < 6:
            continue
        metrics.append(
            _build_gpu_metric(
                gpu_index=row[0],
                name=row[1],
                uuid=row[2],
                utilization_percent=row[3],
                memory_used_mb=row[4],
                memory_total_mb=row[5],
            )
        )
    return metrics


def get_gpu_device_metrics() -> list[dict[str, Any]]:
    for getter in (_get_gpu_device_metrics_via_nvml, _get_gpu_device_metrics_via_nvidia_smi):
        metrics = getter()
        if metrics:
            return metrics
    return []


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
