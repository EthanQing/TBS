from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
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
    name: str | None
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
    process_snapshot: dict | None = None


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


def _compute_mode(value: Any) -> str:
    text = (_text(value) or "").lower().replace("_", " ").replace("-", " ")
    if text in {"0", "default"}: return "default"
    if text in {"1", "exclusive thread"}: return "exclusive_thread"
    if text in {"2", "prohibited"}: return "prohibited"
    if text in {"3", "exclusive process"}: return "exclusive_process"
    return "unknown"


def _mig_mode(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            raw = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return "unknown"
    elif isinstance(value, str):
        raw = value
    elif value is None:
        return "unknown"
    else:
        raw = str(value)
    text = raw.strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text in {"1", "enabled"}: return "enabled"
    if text in {"0", "disabled"}: return "disabled"
    if text in {"n/a", "not supported", "not_supported"}: return "not_supported"
    return "unknown"


def _nvml_process_snapshot(handle) -> dict:
    rows: dict[int, int | None] = {}
    errors: list[str] = []
    attempted = False
    for names in (("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses"),
                  ("nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses")):
        name = next((candidate for candidate in names if getattr(pynvml, candidate, None) is not None), None)
        fn = getattr(pynvml, name, None) if name else None
        if fn is None:
            continue
        attempted = True
        try:
            for process in fn(handle) or []:
                pid = int(process.pid)
                raw = getattr(process, "usedGpuMemory", None)
                unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
                used = _floor_mib_bytes(raw) if isinstance(raw, int) and raw >= 0 and raw != unavailable else None
                prior = rows.get(pid)
                rows[pid] = used if prior is None else prior if used is None else max(prior, used)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return {
        "processes": [{"driver_pid": pid, "memory_used_mib": rows[pid]} for pid in sorted(rows)],
        "complete": attempted and not errors,
        "accounting_status": "verified" if attempted and not errors else "conservative",
        "pid_view": "driver",
        "error": "; ".join(errors) or None,
    }


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
        if count == 0:
            return GpuProbeResult("empty", "nvml", sampled_at, complete=True)

        handles_read = 0
        diagnostic_errors: list[str] = []
        for index in range(count):
            errors: list[str] = []
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception as exc:
                error = f"GPU {index} handle: {exc}"
                diagnostic_errors.append(error)
                devices.append(
                    GpuProbeDevice(
                        gpu_uuid=None,
                        name=f"GPU {index}",
                        observed_index=index,
                        status="failed",
                        error=error,
                    )
                )
                complete = False
                continue
            handles_read += 1

            def read(label: str, fn):
                try:
                    return fn()
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
                    return None

            memory = read("memory", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
            try:
                mig = pynvml.nvmlDeviceGetMigMode(handle)[0] if hasattr(pynvml, "nvmlDeviceGetMigMode") else None
            except Exception as exc:
                if getattr(exc, "value", None) == getattr(pynvml, "NVML_ERROR_NOT_SUPPORTED", 3):
                    mig = "not_supported"
                else:
                    mig = None
                    errors.append(f"mig_mode: {exc}")
            util = read("utilization", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
            uuid = _text(read("uuid", lambda: pynvml.nvmlDeviceGetUUID(handle)))
            if not uuid or not GPU_UUID_RE.fullmatch(uuid):
                errors.append("full physical GPU UUID is unavailable")
                complete = False
                uuid = None
            else:
                uuid = "GPU-" + uuid[4:].lower()
            name = _text(read("name", lambda: pynvml.nvmlDeviceGetName(handle)))
            pci_bus_id = _text(read("pci_bus_id", lambda: pynvml.nvmlDeviceGetPciInfo(handle).busId))
            memory_total_mib = _floor_mib_bytes(getattr(memory, "total", None))
            memory_used_mib = _ceil_mib_bytes(getattr(memory, "used", None))
            memory_free_mib = _floor_mib_bytes(getattr(memory, "free", None))
            utilization_percent = int(getattr(util, "gpu", 0)) if getattr(util, "gpu", None) is not None else None
            compute_mode = _compute_mode(read("compute_mode", lambda: pynvml.nvmlDeviceGetComputeMode(handle)))
            mig_mode = _mig_mode(mig)
            process_snapshot = _nvml_process_snapshot(handle)
            process_snapshot.update({
                "gpu_uuid": uuid,
                "memory_total_mib": memory_total_mib,
                "memory_used_mib": memory_used_mib,
                "memory_free_mib": memory_free_mib,
                "sampled_at": sampled_at.isoformat(),
                "source": "nvml",
            })
            fields = {
                "name": name,
                "pci_bus_id": pci_bus_id,
                "memory_total": memory_total_mib,
                "memory_used": memory_used_mib,
                "memory_free": memory_free_mib,
                "utilization": utilization_percent,
                "compute_mode": compute_mode,
            }
            for label, value in fields.items():
                if (value is None or label == "compute_mode" and value == "unknown") and not any(error.startswith(f"{label}:") for error in errors):
                    errors.append(f"{label} is unavailable")
            has_actual_information = any(
                value is not None
                for value in (
                    uuid,
                    name,
                    pci_bus_id,
                    memory_total_mib,
                    memory_used_mib,
                    memory_free_mib,
                    utilization_percent,
                    None if compute_mode == "unknown" else compute_mode,
                )
            )
            status = "success" if not errors else "partial"
            if not has_actual_information:
                status = "failed"
                errors.append("no GPU device information could be read")
                diagnostic_errors.append(f"GPU {index}: {'; '.join(errors)}")
                complete = False
            devices.append(GpuProbeDevice(
                gpu_uuid=uuid if uuid and GPU_UUID_RE.fullmatch(uuid) else None,
                name=name,
                pci_bus_id=pci_bus_id,
                observed_index=index,
                memory_total_mib=memory_total_mib,
                memory_used_mib=memory_used_mib,
                memory_free_mib=memory_free_mib,
                utilization_percent=utilization_percent,
                compute_mode=compute_mode,
                mig_mode=mig_mode,
                status=status,
                error="; ".join(errors) or None,
                process_snapshot=process_snapshot,
            ))
        valid_devices = [device for device in devices if device.status in {"success", "partial"}]
        if handles_read == 0 or not valid_devices:
            return GpuProbeResult(
                "failed",
                "nvml",
                sampled_at,
                devices,
                "; ".join(diagnostic_errors) or "NVML could not read any enumerated GPU",
                False,
            )
        return GpuProbeResult(
            "success",
            "nvml",
            sampled_at,
            devices,
            error="; ".join(diagnostic_errors) or None,
            complete=complete,
        )
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
    diagnostic_errors: list[str] = []
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        row = next(csv.reader([raw], skipinitialspace=True))
        if len(row) != 10:
            complete = False
            diagnostic_errors.append("nvidia-smi returned a malformed GPU row")
            continue
        errors: list[str] = []
        uuid = _text(row[2])
        if not uuid or not GPU_UUID_RE.fullmatch(uuid):
            uuid = None
            errors.append("full physical GPU UUID is unavailable")
            complete = False
        else:
            uuid = "GPU-" + uuid[4:].lower()
        index_number = _number(row[0])
        if index_number is None:
            errors.append("index is unavailable")
        name = _text(row[1])
        pci_bus_id = _text(row[3])
        memory_total_mib = _mib(row[4])
        memory_used_mib = _mib(row[5], used=True)
        memory_free_mib = _mib(row[6])
        utilization = _number(row[7])
        utilization_percent = int(utilization) if utilization is not None else None
        compute_mode = _compute_mode(row[8])
        mig_mode = _mig_mode(row[9])
        fields = {
            "name": name,
            "pci_bus_id": pci_bus_id,
            "memory_total": memory_total_mib,
            "memory_used": memory_used_mib,
            "memory_free": memory_free_mib,
            "utilization": utilization_percent,
            "compute_mode": compute_mode,
        }
        for label, value in fields.items():
            if value is None or label == "compute_mode" and value == "unknown":
                errors.append(f"{label} is unavailable")
        has_actual_information = any(value is not None for value in (uuid, name, pci_bus_id, memory_total_mib, memory_used_mib, memory_free_mib, utilization_percent))
        status = "success" if not errors else "partial"
        if not has_actual_information:
            status = "failed"
            error = f"GPU row {row[0].strip() or '?'}: no GPU device information could be read"
            errors.append("no GPU device information could be read")
            diagnostic_errors.append(error)
            complete = False
        devices.append(
            GpuProbeDevice(
                gpu_uuid=uuid,
                name=name,
                pci_bus_id=pci_bus_id,
                observed_index=int(index_number) if index_number is not None else None,
                memory_total_mib=memory_total_mib,
                memory_used_mib=memory_used_mib,
                memory_free_mib=memory_free_mib,
                utilization_percent=utilization_percent,
                compute_mode=compute_mode,
                mig_mode=mig_mode,
                status=status,
                error="; ".join(errors) or None,
                process_snapshot={
                    "processes": [],
                    "complete": False,
                    "accounting_status": "conservative",
                    "pid_view": "driver",
                    "error": "nvidia-smi GPU and process queries are not an atomic snapshot",
                },
            )
        )
    valid_devices = [device for device in devices if device.status in {"success", "partial"}]
    if not valid_devices and proc.stdout.strip():
        error = "; ".join(diagnostic_errors) or "nvidia-smi returned malformed CSV"
        return GpuProbeResult("failed", "nvidia-smi", sampled_at, devices, error=error, complete=False)
    process_rows: dict[str, dict[int, int | None]] = {}
    process_error = None
    try:
        process_proc = subprocess.run(
            [executable, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if process_proc.returncode:
            process_error = (process_proc.stderr or process_proc.stdout).strip() or "nvidia-smi process query failed"
        else:
            for raw in process_proc.stdout.splitlines():
                row = next(csv.reader([raw], skipinitialspace=True))
                if len(row) != 3:
                    process_error = "nvidia-smi returned a malformed process row"
                    continue
                uuid = _text(row[0])
                pid = _number(row[1])
                memory = _mib(row[2])
                if uuid and pid is not None:
                    normalized = "GPU-" + uuid[4:].lower() if GPU_UUID_RE.fullmatch(uuid) else uuid
                    prior = process_rows.setdefault(normalized, {}).get(int(pid))
                    process_rows[normalized][int(pid)] = memory if prior is None else prior if memory is None else max(prior, memory)
    except Exception as exc:
        process_error = str(exc)
    devices = [
        replace(device, process_snapshot={
            "gpu_uuid": device.gpu_uuid,
            "memory_total_mib": device.memory_total_mib,
            "memory_used_mib": device.memory_used_mib,
            "memory_free_mib": device.memory_free_mib,
            "sampled_at": sampled_at.isoformat(),
            "source": "nvidia-smi",
            "processes": [{"driver_pid": pid, "memory_used_mib": memory} for pid, memory in sorted(process_rows.get(device.gpu_uuid or "", {}).items())],
            "complete": False,
            "accounting_status": "conservative",
            "pid_view": "driver",
            "error": process_error or "nvidia-smi GPU and process queries are not an atomic snapshot",
        })
        for device in devices
    ]
    return GpuProbeResult("success" if devices else "empty", "nvidia-smi", sampled_at, devices,
                          error=None if complete else "; ".join(diagnostic_errors) or "one or more GPU rows were incomplete",
                          complete=complete)


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
