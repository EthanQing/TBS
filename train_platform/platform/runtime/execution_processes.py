from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil

from train_platform.platform.runtime import process_scope
from train_platform.platform.runtime.execution_identity import process_identity, process_state


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execution_root(training_root: Path, allocation_id: str) -> Path:
    return Path(training_root) / "runtime" / "executions" / str(allocation_id)


def register_execution_process(
    training_root: Path, *, run_id: str, allocation_id: str,
    execution_owner: Mapping[str, Any], pid: int, role: str,
    assigned_gpu_uuids: list[str] | None = None,
) -> dict[str, Any]:
    if execution_owner.get("allocation_id") != allocation_id:
        raise ValueError("execution owner allocation mismatch")
    if process_scope.compare_process_scope(execution_owner.get("process_scope"), process_scope.get_process_scope()) != "same":
        raise ValueError("execution process scope does not match current process")
    identity = process_identity(
        pid, run_id=run_id, allocation_id=allocation_id,
        execution_owner=dict(execution_owner), role=role,
        assigned_gpu_uuids=list(assigned_gpu_uuids or []),
    )
    if os.name != "nt":
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
        close = stat.rfind(")")
        identity["start_ticks"] = int(stat[close + 2:].split()[19])
        identity["clock_ticks_per_second"] = int(os.sysconf("SC_CLK_TCK"))
        identity["boot_time"] = float(psutil.boot_time())
    if process_state(identity)[0] != "live":
        raise psutil.NoSuchProcess(pid)
    _atomic_json(execution_root(training_root, allocation_id) / "processes" / f"{role}-{pid}.json", identity)
    return identity


def start_descendant_registration(
    training_root: Path, *, run_id: str, allocation_id: str,
    execution_owner: Mapping[str, Any], supervisor_pid: int,
    assigned_gpu_uuids: list[str] | None = None, interval_seconds: float = 0.5,
) -> tuple[threading.Event, threading.Thread]:
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(interval_seconds):
            try:
                supervisor = psutil.Process(supervisor_pid)
                children = supervisor.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            except psutil.Error as exc:
                _atomic_json(execution_root(training_root, allocation_id) / "registration-error.json",
                             {"allocation_id": allocation_id, "error": str(exc)})
                children = []
            for child in children:
                try:
                    register_execution_process(
                        training_root, run_id=run_id, allocation_id=allocation_id,
                        execution_owner=execution_owner, pid=child.pid, role="descendant",
                        assigned_gpu_uuids=assigned_gpu_uuids,
                    )
                except (psutil.NoSuchProcess, FileNotFoundError):
                    continue
                except (OSError, ValueError, psutil.Error):
                    try:
                        _atomic_json(execution_root(training_root, allocation_id) / "registration-error.json",
                                     {"allocation_id": allocation_id, "error": "descendant registration failed"})
                    except OSError:
                        pass
                    continue

    thread = threading.Thread(target=watch, name=f"execution-processes-{allocation_id}", daemon=True)
    thread.start()
    return stopped, thread


def cleanup_registered_execution(
    training_root: Path, *, run_id: str, allocation_id: str,
    execution_owner: Mapping[str, Any], terminate: bool = True,
    grace_seconds: float = 2.0, exclude_supervisor: bool = False,
) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    root = execution_root(training_root, allocation_id)
    processes_dir = root / "processes"
    if not processes_dir.is_dir():
        return {"complete": False, "allocation_id": allocation_id, "execution_owner": dict(execution_owner),
                "process_scope": process_scope.get_process_scope(), "checked_at": checked_at,
                "error": "process registration directory is missing", "survivors": []}
    error_path = root / "registration-error.json"
    if error_path.exists():
        return {"complete": False, "allocation_id": allocation_id, "execution_owner": dict(execution_owner),
                "process_scope": process_scope.get_process_scope(), "checked_at": checked_at,
                "error": "process registration was incomplete", "survivors": []}
    identities = []
    try:
        for path in processes_dir.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"invalid process registration: {path}")
            if value.get("run_id") != run_id or value.get("allocation_id") != allocation_id or value.get("execution_owner") != dict(execution_owner):
                raise ValueError(f"process registration ownership mismatch: {path}")
            identities.append(value)
    except (OSError, ValueError) as exc:
        return {"complete": False, "allocation_id": allocation_id, "execution_owner": dict(execution_owner),
                "process_scope": process_scope.get_process_scope(), "checked_at": checked_at,
                "error": str(exc), "survivors": []}
    supervisor = [
        item for item in identities
        if item.get("role") == "supervisor"
        and item.get("pid") == execution_owner.get("guard_pid")
        and item.get("create_time") == execution_owner.get("guard_create_time")
    ]
    if len(supervisor) != 1:
        return {"complete": False, "allocation_id": allocation_id, "execution_owner": dict(execution_owner),
                "process_scope": process_scope.get_process_scope(), "checked_at": checked_at,
                "error": "exact supervisor registration is missing", "survivors": []}
    processes: dict[tuple[int, float], psutil.Process] = {}
    unknown = []
    for identity in identities:
        if exclude_supervisor and identity.get("pid") == execution_owner.get("guard_pid"):
            continue
        state, process = process_state(identity)
        if state == "live":
            processes[(process.pid, identity["create_time"])] = process
        elif state == "unknown":
            unknown.append(identity.get("pid"))
    # Children may outlive their parent between watcher samples. A dedicated
    # session remains owned while its original session leader identity matches,
    # or the leader PID no longer exists. Never claim a reused session leader.
    sessions = {}
    if os.name != "nt" and process_scope.compare_process_scope(execution_owner.get("process_scope"), process_scope.get_process_scope()) == "same":
        for identity in identities:
            sid = identity.get("sid")
            if sid is None or sid != identity.get("pid"):
                continue
            state, _ = process_state(identity)
            if state == "unknown":
                unknown.append(sid)
                continue
            try:
                if psutil.Process(sid).create_time() != identity["create_time"]:
                    continue
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                unknown.append(sid)
                continue
            sessions[sid] = identity
        for candidate in psutil.process_iter():
            sid = None
            try:
                sid = os.getsid(candidate.pid)
                if sid not in sessions or (exclude_supervisor and candidate.pid == execution_owner.get("guard_pid")):
                    continue
                if candidate.create_time() < sessions[sid]["create_time"]:
                    unknown.append(candidate.pid)
                    continue
                item = register_execution_process(
                    training_root, run_id=run_id, allocation_id=allocation_id,
                    execution_owner=execution_owner, pid=candidate.pid, role="session-member",
                    assigned_gpu_uuids=supervisor[0].get("assigned_gpu_uuids", []),
                )
                state, process = process_state(item)
                if state == "live":
                    processes[(process.pid, item["create_time"])] = process
                elif state == "unknown":
                    unknown.append(candidate.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
            except (psutil.Error, OSError):
                # No ownership is inferred for unrelated inaccessible processes.
                if sid in sessions:
                    unknown.append(candidate.pid)
    if terminate:
        for process in sorted(processes.values(), key=lambda item: item.pid, reverse=True):
            try: process.terminate()
            except psutil.Error: pass
        _, alive = psutil.wait_procs(list(processes.values()), timeout=grace_seconds)
        for process in alive:
            try: process.kill()
            except psutil.Error: pass
        psutil.wait_procs(alive, timeout=grace_seconds)
    survivors = []
    for process in processes.values():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                survivors.append(process.pid)
        except psutil.Error:
            unknown.append(process.pid)
    if sessions:
        for candidate in psutil.process_iter():
            sid = None
            try:
                sid = os.getsid(candidate.pid)
                if sid not in sessions or (exclude_supervisor and candidate.pid == execution_owner.get("guard_pid")):
                    continue
                if candidate.is_running() and candidate.status() != psutil.STATUS_ZOMBIE:
                    survivors.append(candidate.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
            except (psutil.Error, OSError):
                if sid in sessions:
                    unknown.append(candidate.pid)
    return {"complete": not survivors and not unknown, "allocation_id": allocation_id,
            "supervisor_excluded": exclude_supervisor,
            "execution_owner": dict(execution_owner), "process_scope": process_scope.get_process_scope(),
            "checked_at": checked_at, "survivors": sorted(set(survivors)), "unknown": sorted(set(x for x in unknown if x is not None))}
