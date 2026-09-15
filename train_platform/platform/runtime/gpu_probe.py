from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

try:
    import pynvml
except Exception:
    pynvml = None


ProbeStatus = Literal["success", "empty", "unavailable", "failed"]
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


@dataclass(frozen=True)
class GpuProbeDevice:
    gpu_uuid: str | None
    name: str
    pci_bus_id: str | None = None
    observed_index: int | None = None
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    memory_free_mib: int | None = None
    utilization_percent: int | None = None
    compute_mode: str | None = None
    mig_mode: str | None = None
    status: str = "success"
    error: str | None = None


@dataclass(frozen=True)
class GpuProbeResult:
    status: ProbeStatus
    source: str | None
    sampled_at: datetime
    devices: list[GpuProbeDevice] = field(default_factory=list)
    error: str | None = None
    complete: bool = False


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value).strip()
    return None if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED", "UNKNOWN"} else value


def _number(value: Any) -> float | None:
    text = _text(value)
    if text is None:
        return None
    try:
        result = float(text.split()[0])
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def _floor_mib_bytes(value: Any) -> int | None:
    try:
        return math.floor(int(value) / (1024 * 1024))
    except (ValueError, TypeError):
        return None


def _ceil_mib_bytes(value: Any) -> int | None:
    try:
        return math.ceil(int(value) / (1024 * 1024))
    except (ValueError, TypeError):
        return None


def _mib(value: Any, *, used: bool = False) -> int | None:
    text = _text(value)
    number = _number(text)
    if number is None:
        return None
    unit = (text or "").lower().replace(" ", "")
    if unit.endswith("gib"):
        number *= 1024
    elif unit.endswith("kib"):
        number /= 1024
    elif unit.endswith("bytes") or unit.endswith("byte") or unit.endswith("b") and not unit.endswith("mib"):
        number /= 1024 * 1024
    return math.ceil(number) if used else math.floor(number)


def _nvml_probe(sampled_at: datetime) -> GpuProbeResult:
    if pynvml is None:
        return GpuProbeResult("unavailable", "nvml", sampled_at, error="pynvml is unavailable")
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        code = getattr(exc, "value", None)
        status: ProbeStatus = "unavailable" if code in {9, 12} else "failed"
        return GpuProbeResult(status, "nvml", sampled_at, error=str(exc))
    devices: list[GpuProbeDevice] = []
    complete = True
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        for index in range(count):
            errors: list[str] = []
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception as exc:
                devices.append(GpuProbeDevice(None, f"GPU {index}", observed_index=index, status="failed", error=str(exc)))
                complete = False
                continue
            def read(label: str, fn):
                try:
                    return fn()
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
                    return None
            memory = read("memory", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
            try:
                mig = pynvml.nvmlDeviceGetMigMode(handle)[0] if hasattr(pynvml, "nvmlDeviceGetMigMode") else None
            except Exception:
                mig = None
            util = read("utilization", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
            uuid = _text(read("uuid", lambda: pynvml.nvmlDeviceGetUUID(handle)))
            if not uuid or not GPU_UUID_RE.fullmatch(uuid):
                errors.append("full physical GPU UUID is unavailable")
                complete = False
            else:
                uuid = "GPU-" + uuid[4:].lower()
            devices.append(GpuProbeDevice(
                gpu_uuid=uuid if uuid and GPU_UUID_RE.fullmatch(uuid) else None,
                name=_text(read("name", lambda: pynvml.nvmlDeviceGetName(handle))) or f"GPU {index}",
                pci_bus_id=_text(read("pci_bus_id", lambda: pynvml.nvmlDeviceGetPciInfo(handle).busId)),
                observed_index=index,
                memory_total_mib=_floor_mib_bytes(getattr(memory, "total", None)),
                memory_used_mib=_ceil_mib_bytes(getattr(memory, "used", None)),
                memory_free_mib=_floor_mib_bytes(getattr(memory, "free", None)),
                utilization_percent=int(getattr(util, "gpu", 0)) if getattr(util, "gpu", None) is not None else None,
                compute_mode=_text(read("compute_mode", lambda: pynvml.nvmlDeviceGetComputeMode(handle))),
                mig_mode="enabled" if mig == 1 else ("disabled" if mig == 0 else None),
                status="success" if not errors else "partial",
                error="; ".join(errors) or None,
            ))
        return GpuProbeResult("success" if devices else "empty", "nvml", sampled_at, devices, complete=complete)
    except Exception as exc:
        return GpuProbeResult("failed", "nvml", sampled_at, devices, str(exc), False)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _smi_probe(sampled_at: datetime) -> GpuProbeResult:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return GpuProbeResult("unavailable", "nvidia-smi", sampled_at, error="nvidia-smi is unavailable")
    fields = "index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,compute_mode,mig.mode.current"
    try:
        proc = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return GpuProbeResult("failed", "nvidia-smi", sampled_at, error=str(exc))
    if proc.returncode:
        error = (proc.stderr or proc.stdout).strip()
        status: ProbeStatus = "unavailable" if proc.returncode == 9 or "driver not loaded" in error.lower() else "failed"
        return GpuProbeResult(status, "nvidia-smi", sampled_at, error=error)
    devices: list[GpuProbeDevice] = []
    complete = True
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        row = next(csv.reader([raw], skipinitialspace=True))
        if len(row) != 10:
            complete = False
            continue
        uuid = _text(row[2])
        error = None
        if not uuid or not GPU_UUID_RE.fullmatch(uuid):
            uuid = None
            error = "full physical GPU UUID is unavailable"
            complete = False
        else:
            uuid = "GPU-" + uuid[4:].lower()
        index_number = _number(row[0])
        utilization = _number(row[7])
        devices.append(GpuProbeDevice(uuid, _text(row[1]) or "GPU", _text(row[3]), int(index_number) if index_number is not None else None,
            _mib(row[4]), _mib(row[5], used=True), _mib(row[6]), int(utilization) if utilization is not None else None,
            _text(row[8]), _text(row[9]), "success" if not error else "failed", error))
    if not devices and proc.stdout.strip():
        return GpuProbeResult("failed", "nvidia-smi", sampled_at, error="nvidia-smi returned malformed CSV")
    return GpuProbeResult("success" if devices else "empty", "nvidia-smi", sampled_at, devices,
                          error=None if complete else "one or more GPU rows were incomplete", complete=complete)


def probe_gpus() -> GpuProbeResult:
    sampled_at = datetime.now(timezone.utc)
    nvml_result = _nvml_probe(sampled_at)
    if nvml_result.status in {"success", "empty"}:
        return nvml_result
    smi_result = _smi_probe(sampled_at)
    if smi_result.status in {"success", "empty"}:
        return smi_result
    errors = "; ".join(x for x in [nvml_result.error, smi_result.error] if x)
    status: ProbeStatus = "unavailable" if nvml_result.status == smi_result.status == "unavailable" else "failed"
    return GpuProbeResult(status, smi_result.source, sampled_at, error=errors or None)
