from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from train_platform.platform.runtime.process_scope import compare_process_scope


@dataclass(frozen=True)
class ProcessAttribution:
    usage_by_allocation: dict[str, int]
    complete: bool
    has_duplicates: bool
    error: str | None = None


def _proc_identity(proc_root: Path, pid: int) -> dict | None:
    base = proc_root / str(pid)
    try:
        stat = (base / "stat").read_text(errors="replace")
        close = stat.rfind(")")
        start_ticks = int(stat[close + 2:].split()[19])
        status = (base / "status").read_text(errors="replace")
        nspid = []
        for line in status.splitlines():
            if line.startswith("NSpid:"):
                nspid = [int(item) for item in line.split()[1:]]
                break
        namespace = os.stat(base / "ns" / "pid")
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
        return {"start_ticks": start_ticks, "nspid": nspid,
                "process_scope": {"boot_id": boot_id, "pid_namespace": {
                    "device": namespace.st_dev, "inode": namespace.st_ino}}}
    except (OSError, ValueError, IndexError):
        return None


def load_execution_registrations(root: str | Path) -> list[dict]:
    base = Path(root)
    registrations: list[dict] = []
    if not base.exists():
        return registrations
    for path in base.glob("*/processes/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and value.get("allocation_id") and value.get("pid") is not None:
            registrations.append(value)
    return registrations


def attribute_driver_processes(
    driver_processes: Iterable[dict], registrations: Iterable[dict], *, host_proc_root: str | Path | None,
    gpu_uuid: str, active_owners: dict[str, dict],
) -> ProcessAttribution:
    if not host_proc_root:
        return ProcessAttribution({}, False, False, "GPU_HOST_PROC_ROOT is not configured")
    proc_root = Path(host_proc_root)
    by_driver_pid: dict[int, dict] = {}
    duplicate = False
    for row in driver_processes:
        pid = row.get("driver_pid")
        if not isinstance(pid, int):
            continue
        prior = by_driver_pid.get(pid)
        if prior is not None and prior.get("memory_used_mib") != row.get("memory_used_mib"):
            duplicate = True
        if prior is None:
            by_driver_pid[pid] = row
    usage: dict[str, int] = {}
    registration_list = list(registrations)
    complete = True
    for pid, row in by_driver_pid.items():
        identity = _proc_identity(proc_root, pid)
        if identity is None:
            complete = False
            continue
        matched = []
        for registration in registration_list:
            scope = registration.get("process_scope") or {}
            expected_create = registration.get("create_time")
            allocation_id = registration.get("allocation_id")
            owner = active_owners.get(allocation_id)
            if (not identity or expected_create is None or not owner
                    or registration.get("execution_owner") != owner
                    or owner.get("allocation_id") != allocation_id
                    or gpu_uuid not in registration.get("assigned_gpu_uuids", [])):
                continue
            hz = registration.get("clock_ticks_per_second")
            boot_time = registration.get("boot_time")
            if not isinstance(hz, int) or hz <= 0 or not isinstance(boot_time, (int, float)):
                continue
            observed_create = boot_time + identity["start_ticks"] / hz
            pid_matches = bool(identity["nspid"]) and identity["nspid"][-1] == registration.get("pid")
            namespace_matches = (compare_process_scope(scope, identity["process_scope"]) == "same"
                                 and compare_process_scope(scope, owner.get("process_scope")) == "same")
            if (pid_matches and namespace_matches
                    and registration.get("start_ticks") == identity["start_ticks"]
                    and abs(observed_create - float(expected_create)) < 1 / hz):
                matched.append(registration)
        memory = row.get("memory_used_mib")
        owners = {item["allocation_id"] for item in matched}
        if len(owners) > 1:
            return ProcessAttribution({}, False, True, "Conflicting execution ownership")
        if not owners:
            # Unknown processes remain external occupancy in the whole-card sample.
            continue
        if not isinstance(memory, int) or isinstance(memory, bool) or memory < 0:
            complete = False
            continue
        allocation_id = owners.pop()
        usage[allocation_id] = usage.get(allocation_id, 0) + memory
    return ProcessAttribution(usage, complete and not duplicate, duplicate)
