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


_pending_errors_lock = threading.Lock()
_pending_errors: dict[tuple[str, str], dict[str, Any]] = {}


def _flush_pending_registration_errors(root: Path) -> list[dict[str, Any]]:
    root_key = str(root)
    with _pending_errors_lock:
        pending = [((pending_root, item_id), dict(item))
                   for (pending_root, item_id), item in _pending_errors.items() if pending_root == root_key]
    failed = []
    for key, item in pending:
        path = root / "registration-errors" / f"{item['id']}.json"
        try:
            _atomic_json(path, item)
            with _pending_errors_lock:
                _pending_errors.pop(key, None)
        except OSError as exc:
            item.update(stage="registration_io_error", error=str(exc), persistence_pending=True)
            failed.append(item)
    return failed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execution_root(training_root: Path, allocation_id: str) -> Path:
    return Path(training_root) / "runtime" / "executions" / str(allocation_id)


def _record_registration_error(
    training_root: Path, allocation_id: str, execution_owner: Mapping[str, Any],
    *, stage: str, error: str, target: Mapping[str, Any] | None = None,
) -> None:
    target_value = dict(target or {})
    entry_id = uuid.uuid4().hex
    entry = {
        "id": entry_id, "allocation_id": allocation_id,
        "execution_owner": dict(execution_owner), "stage": stage,
        "target": target_value, "error": error,
        "last_checked_at": datetime.now(timezone.utc).isoformat(), "resolved": False,
    }
    key = (str(execution_root(training_root, allocation_id)), entry_id)
    with _pending_errors_lock:
        _pending_errors[key] = entry
    path = execution_root(training_root, allocation_id) / "registration-errors" / f"{entry_id}.json"
    _atomic_json(path, entry)
    with _pending_errors_lock:
        _pending_errors.pop(key, None)


def register_execution_process(
    training_root: Path, *, run_id: str, allocation_id: str,
    execution_owner: Mapping[str, Any], pid: int, role: str,
    assigned_gpu_uuids: list[str] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
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
    if role == "supervisor" and expected_identity is None:
        expected_identity = {
            "pid": execution_owner.get("guard_pid"),
            "create_time": execution_owner.get("guard_create_time"),
            "process_scope": execution_owner.get("process_scope"),
        }
    if expected_identity is not None and any(
        identity.get(field) != expected_identity.get(field)
        for field in ("pid", "create_time", "process_scope", "sid", "pgid")
        if field in expected_identity
    ):
        raise ValueError("process identity changed before registration")
    if role == "supervisor" and any((
        identity.get("pid") != execution_owner.get("guard_pid"),
        identity.get("create_time") != execution_owner.get("guard_create_time"),
        process_scope.compare_process_scope(identity.get("process_scope"), execution_owner.get("process_scope")) != "same",
    )):
        raise ValueError("supervisor identity does not match execution owner")
    try:
        if os.name != "nt":
            stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
            close = stat.rfind(")")
            identity["start_ticks"] = int(stat[close + 2:].split()[19])
            identity["clock_ticks_per_second"] = int(os.sysconf("SC_CLK_TCK"))
            identity["boot_time"] = float(psutil.boot_time())
        identity_state = process_state(identity)[0]
        if identity_state == "dead":
            raise psutil.NoSuchProcess(pid)
        if identity_state != "live":
            raise RuntimeError("process identity cannot be confirmed")
        _atomic_json(execution_root(training_root, allocation_id) / "processes" / f"{role}-{pid}.json", identity)
    except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
        try:
            _record_registration_error(
                training_root, allocation_id, execution_owner,
                stage=f"{role}_registration_write", error=str(exc), target=identity,
            )
        except OSError:
            pass
        raise
    return identity


def start_descendant_registration(
    training_root: Path, *, run_id: str, allocation_id: str,
    execution_owner: Mapping[str, Any], supervisor_pid: int,
    assigned_gpu_uuids: list[str] | None = None, interval_seconds: float = 0.5,
) -> tuple[threading.Event, threading.Thread]:
    stopped = threading.Event()
    supervisor_identity = {
        "pid": int(execution_owner.get("guard_pid", supervisor_pid)),
        "create_time": execution_owner.get("guard_create_time"),
        "process_scope": execution_owner.get("process_scope"),
        "run_id": run_id, "allocation_id": allocation_id,
        "execution_owner": dict(execution_owner),
        "assigned_gpu_uuids": list(assigned_gpu_uuids or []),
    }

    def watch() -> None:
        try:
            while not stopped.wait(interval_seconds):
                _flush_pending_registration_errors(execution_root(training_root, allocation_id))
                _watch_once()
        finally:
            _flush_pending_registration_errors(execution_root(training_root, allocation_id))

    def _watch_once() -> None:
        state, supervisor_process = process_state(supervisor_identity)
        if state == "dead":
            return
        if state != "live" or supervisor_process is None:
            try:
                _record_registration_error(
                    training_root, allocation_id, execution_owner,
                    stage="supervisor_check", error="supervisor identity could not be confirmed",
                    target=supervisor_identity,
                )
            except OSError:
                pass
            return
        if os.name != "nt":
            for field, getter in (("sid", os.getsid), ("pgid", os.getpgid)):
                if supervisor_identity.get(field) is None:
                    try:
                        supervisor_identity[field] = int(getter(supervisor_pid))
                    except OSError:
                        pass
        try:
            children = supervisor_process.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        except psutil.Error as exc:
            try:
                _record_registration_error(training_root, allocation_id, execution_owner,
                                           stage="descendant_scan", error=str(exc), target=supervisor_identity)
            except OSError:
                pass
            children = []
        for child in children:
            target = {"pid": child.pid, "process_scope": execution_owner.get("process_scope"),
                      "run_id": run_id, "allocation_id": allocation_id,
                      "execution_owner": dict(execution_owner),
                      "assigned_gpu_uuids": list(assigned_gpu_uuids or [])}
            try:
                target["create_time"] = child.create_time()
                captured = process_identity(child.pid, run_id=run_id, allocation_id=allocation_id,
                                            execution_owner=dict(execution_owner), role="descendant",
                                            assigned_gpu_uuids=list(assigned_gpu_uuids or []))
                if any(captured.get(field) != target.get(field)
                       for field in ("pid", "create_time", "process_scope")):
                    raise ValueError("enumerated descendant identity changed before capture")
                target = captured
                register_execution_process(
                    training_root, run_id=run_id, allocation_id=allocation_id,
                    execution_owner=execution_owner, pid=child.pid, role="descendant",
                    assigned_gpu_uuids=assigned_gpu_uuids,
                    expected_identity=target,
                )
            except (OSError, RuntimeError, ValueError, psutil.Error):
                try:
                    _record_registration_error(
                        training_root, allocation_id, execution_owner,
                        stage="descendant_registration", error="descendant registration failed",
                        target=target,
                    )
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
    assigned_gpu_uuids: list[str] | None = None,
) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    root = execution_root(training_root, allocation_id)
    processes_dir = root / "processes"
    error_path = root / "registration-error.json"
    registration_errors: list[dict[str, Any]] = []
    if error_path.exists():
        try:
            value = json.loads(error_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("errors"), list):
                registration_errors = [dict(item) for item in value["errors"] if isinstance(item, dict)]
            elif isinstance(value, dict) and value.get("resolved"):
                pass
            elif isinstance(value, dict):
                registration_errors = [{"id": "legacy", "stage": "legacy", "error": value.get("error"),
                                        "resolved": False}]
            else:
                raise ValueError("invalid registration error journal")
        except (OSError, ValueError) as exc:
            registration_errors = [{"id": "journal", "stage": "journal_read", "error": str(exc),
                                    "resolved": False}]
    errors_dir = root / "registration-errors"
    initial_error_paths: set[str] = set()
    if errors_dir.is_dir():
        for path in errors_dir.glob("*.json"):
            initial_error_paths.add(str(path))
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(item, dict):
                    raise ValueError("invalid registration error")
                item["_path"] = str(path)
                registration_errors.append(item)
            except (OSError, ValueError) as exc:
                registration_errors.append({"id": path.name, "stage": "journal_read",
                                            "error": str(exc), "resolved": False})
    registration_errors.extend(_flush_pending_registration_errors(root))
    identities = []
    registration_read_errors = []
    for path in processes_dir.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"invalid process registration: {path}")
            if value.get("run_id") != run_id or value.get("allocation_id") != allocation_id or value.get("execution_owner") != dict(execution_owner):
                raise ValueError(f"process registration ownership mismatch: {path}")
            if (not isinstance(value.get("pid"), int)
                    or not isinstance(value.get("create_time"), (int, float))
                    or process_scope.compare_process_scope(value.get("process_scope"), execution_owner.get("process_scope")) != "same"):
                raise ValueError(f"process registration identity is incomplete: {path}")
            identities.append(value)
        except (OSError, ValueError) as exc:
            registration_read_errors.append({"stage": "registration_read", "error": str(exc)})
    registration_errors.extend(registration_read_errors)
    for item in registration_errors:
        target = item.get("target")
        if item.get("id") == "legacy":
            continue
        if item.get("stage") in {"registration_read", "journal_read"}:
            continue
        trusted = (isinstance(target, dict) and item.get("allocation_id") == allocation_id
                   and item.get("execution_owner") == dict(execution_owner))
        if isinstance(target, dict):
            trusted = trusted and process_scope.compare_process_scope(
                target.get("process_scope"), execution_owner.get("process_scope")) == "same"
            for field, expected_value in (("run_id", run_id), ("allocation_id", allocation_id),
                                          ("execution_owner", dict(execution_owner))):
                if target.get(field) != expected_value:
                    trusted = False
        item["trusted"] = trusted
        if not trusted:
            item["stage"] = "registration_ownership_mismatch"
            item["error"] = "registration error identity is not owned by this execution"
            item["resolved"] = False
    supervisor = [
        item for item in identities
        if item.get("role") == "supervisor"
        and item.get("pid") == execution_owner.get("guard_pid")
        and item.get("create_time") == execution_owner.get("guard_create_time")
    ]
    recovered_base = False
    if len(supervisor) != 1:
        expected = {"pid": execution_owner.get("guard_pid"),
                    "create_time": execution_owner.get("guard_create_time"),
                    "process_scope": execution_owner.get("process_scope")}
        if process_state(expected)[0] == "live":
            try:
                supervisor = [register_execution_process(
                    training_root, run_id=run_id, allocation_id=allocation_id,
                    execution_owner=execution_owner, pid=int(expected["pid"]), role="supervisor",
                    expected_identity=expected,
                    assigned_gpu_uuids=assigned_gpu_uuids,
                )]
                recovered_base = True
            except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
                supervisor = []
                registration_errors.append({
                    "stage": "registration_io_error" if isinstance(exc, OSError) else "process_scope_unconfirmed",
                    "error": str(exc), "last_checked_at": checked_at,
                })
        if not supervisor:
            durable = next((item for item in identities
                            if item.get("pid") == expected["pid"]
                            and item.get("create_time") == expected["create_time"]
                            and item.get("execution_owner") == dict(execution_owner)
                            and process_scope.compare_process_scope(item.get("process_scope"), expected["process_scope"]) == "same"
                            and item.get("sid") == item.get("pid")), None)
            if durable is None:
                error_target = next((item.get("target") for item in registration_errors
                                     if item.get("trusted") and isinstance(item.get("target"), dict)
                                     and item["target"].get("pid") == expected["pid"]
                                     and item["target"].get("create_time") == expected["create_time"]
                                     and item["target"].get("sid") == expected["pid"]
                                     and item["target"].get("run_id") == run_id
                                     and item["target"].get("allocation_id") == allocation_id
                                     and item["target"].get("execution_owner") == dict(execution_owner)), None)
                if error_target is not None:
                    durable = dict(error_target)
                    durable.update(run_id=run_id, allocation_id=allocation_id,
                                   execution_owner=dict(execution_owner), role="supervisor")
            if durable is not None:
                copied = dict(durable)
                copied["role"] = "supervisor"
                copied["assigned_gpu_uuids"] = list(assigned_gpu_uuids or copied.get("assigned_gpu_uuids", []))
                try:
                    _atomic_json(processes_dir / f"supervisor-{expected['pid']}.json", copied)
                    supervisor = [copied]
                    recovered_base = True
                except OSError as exc:
                    registration_errors.append({"stage": "registration_io_error", "error": str(exc)})
        if len(supervisor) != 1:
            registration_errors.append({"id": "missing-base", "stage": "base_registration",
                                        "error": "exact supervisor registration is missing", "resolved": False})
    elif supervisor:
        expected_scope = execution_owner.get("process_scope")
        if process_scope.compare_process_scope(supervisor[0].get("process_scope"), expected_scope) != "same":
            supervisor = []
            registration_errors.append({"id": "invalid-base", "stage": "base_registration",
                                        "error": "supervisor registration scope mismatch", "resolved": False})
    if supervisor and supervisor[0] not in identities:
        identities.append(supervisor[0])

    # Error targets are durable evidence, including after a previous resolution.
    # Persist the original identity; never recapture an exited/reused PID.
    for error in registration_errors:
        target = error.get("target")
        if not error.get("trusted") or not isinstance(target, dict):
            continue
        if not isinstance(target.get("pid"), int) or not isinstance(target.get("create_time"), (int, float)):
            error["resolved"] = False
            continue
        identities.append(dict(target))
        state, _ = process_state(target)
        error["last_checked_at"] = checked_at
        error["last_check"] = state
        error["target_exited"] = state == "dead"
        if state == "live":
            try:
                _atomic_json(processes_dir / f"recovered-{target['pid']}.json", target)
                error["identity_recovered"] = True
                if error.get("stage") == "descendant_registration" or str(error.get("stage", "")).endswith("_registration_write"):
                    error["resolved"] = True
                    error["resolution"] = "original_identity_registered"
            except OSError as exc:
                error["resolved"] = False
                error["recovery_error"] = str(exc)
    identity_by_key = {}
    conflicting_keys = set()
    for identity in identities:
        key = (identity["pid"], identity["create_time"])
        previous = identity_by_key.get(key)
        if previous is not None and any(
            previous.get(field) is not None and identity.get(field) is not None
            and previous[field] != identity[field] for field in ("sid", "pgid", "assigned_gpu_uuids")
        ):
            conflicting_keys.add(key)
        # Only merge evidence for the same validated identity. Keep known
        # fields when another record could not capture them.
        identity_by_key[key] = {**(previous or {}), **{
            field: value for field, value in identity.items() if value is not None}}
    identities = [identity for key, identity in identity_by_key.items() if key not in conflicting_keys]
    processes: dict[tuple[int, float], psutil.Process] = {}
    unknown = [key[0] for key in conflicting_keys]
    for key in conflicting_keys:
        registration_errors.append({"stage": "identity_conflict", "resolved": False,
                                    "target": identity_by_key[key], "error": "conflicting original process identities"})
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
    unconfirmed_sessions = []

    def unconfirmed_session(identity, reason):
        value = {"sid": identity.get("sid"), "target": dict(identity), "reason": reason}
        if value not in unconfirmed_sessions:
            unconfirmed_sessions.append(value)

    def session_candidates():
        try:
            yield from psutil.process_iter()
        except (psutil.Error, OSError) as exc:
            for identity in sessions.values():
                unconfirmed_session(identity, f"session enumeration failed: {exc}")

    if os.name != "nt":
        for identity in identities:
            if not isinstance(identity.get("sid"), int):
                unconfirmed_session(identity, "original session identity is unavailable")
        for error in registration_errors:
            target = error.get("target")
            if error.get("trusted") and isinstance(target, dict):
                evidence = identity_by_key.get((target.get("pid"), target.get("create_time")), target)
                if not isinstance(evidence.get("sid"), int):
                    error["resolved"] = False
                    unconfirmed_session(target, "original session identity is unavailable")

    session_scope_matches = process_scope.compare_process_scope(
        execution_owner.get("process_scope"), process_scope.get_process_scope()) == "same"
    if os.name != "nt" and not session_scope_matches:
        for identity in identities:
            if identity.get("sid") == identity.get("pid"):
                unconfirmed_session(identity, "session process scope cannot be confirmed")
    if os.name != "nt" and session_scope_matches:
        for identity in identities:
            sid = identity.get("sid")
            if sid is None or sid != identity.get("pid"):
                continue
            state, _ = process_state(identity)
            if state == "unknown":
                unknown.append(sid)
                unconfirmed_session(identity, "session leader identity cannot be confirmed")
                continue
            try:
                if psutil.Process(sid).create_time() != identity["create_time"]:
                    unconfirmed_session(identity, "session leader PID was reused")
                    continue
                if os.getsid(sid) != sid:
                    unconfirmed_session(identity, "session leader changed session")
                    continue
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError):
                unknown.append(sid)
                unconfirmed_session(identity, "session ownership cannot be confirmed")
                continue
            sessions[sid] = identity
        for error in registration_errors:
            target = error.get("target")
            if error.get("trusted") and isinstance(target, dict):
                evidence = identity_by_key.get((target.get("pid"), target.get("create_time")), target)
                if evidence.get("sid") is not None and evidence["sid"] not in sessions:
                    unconfirmed_session(evidence, "no confirmed original session leader")
        for candidate in session_candidates():
            sid = None
            try:
                sid = os.getsid(candidate.pid)
                if sid not in sessions or (exclude_supervisor and candidate.pid == execution_owner.get("guard_pid")):
                    continue
                if (candidate.pid, candidate.create_time()) in conflicting_keys:
                    unknown.append(candidate.pid)
                    continue
                if candidate.create_time() < sessions[sid]["create_time"]:
                    unknown.append(candidate.pid)
                    continue
                if not candidate.is_running() or candidate.status() == psutil.STATUS_ZOMBIE:
                    continue
                candidate_identity = {"pid": candidate.pid, "create_time": candidate.create_time(),
                                      "process_scope": execution_owner.get("process_scope"), "sid": sid}
                item = register_execution_process(
                    training_root, run_id=run_id, allocation_id=allocation_id,
                    execution_owner=execution_owner, pid=candidate.pid, role="session-member",
                    assigned_gpu_uuids=sessions[sid].get("assigned_gpu_uuids", []),
                    expected_identity=candidate_identity,
                )
                state, process = process_state(item)
                if state == "live":
                    processes[(process.pid, item["create_time"])] = process
                elif state == "unknown":
                    unknown.append(candidate.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
            except (RuntimeError, ValueError, psutil.Error, OSError):
                # No ownership is inferred for unrelated inaccessible processes.
                if sid in sessions:
                    unknown.append(candidate.pid)
                    unconfirmed_session(sessions[sid], f"member {candidate.pid} cannot be confirmed")
                elif sid is None:
                    for identity in sessions.values():
                        unconfirmed_session(identity, f"session of process {candidate.pid} cannot be read")
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
        # Recheck the leader before accepting the final enumeration as proof.
        for sid, identity in sessions.items():
            try:
                leader = psutil.Process(sid)
                if leader.create_time() != identity["create_time"] or os.getsid(sid) != sid:
                    unconfirmed_session(identity, "session ownership changed during cleanup")
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError):
                unconfirmed_session(identity, "session ownership cannot be confirmed after cleanup")
        for candidate in session_candidates():
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
                    unconfirmed_session(sessions[sid], f"member {candidate.pid} cannot be confirmed")
                elif sid is None:
                    for identity in sessions.values():
                        unconfirmed_session(identity, f"session of process {candidate.pid} cannot be read")
    hard_read_error = any(item.get("stage") in {"registration_read", "journal_read", "registration_io_error"}
                          for item in registration_errors if not item.get("resolved"))
    complete_session_scan = bool(sessions) and execution_owner.get("guard_pid") in sessions and not survivors and not unknown and not unconfirmed_sessions and not hard_read_error
    unresolved_errors = []
    for item in registration_errors:
        if item.get("resolved"):
            continue
        target = item.get("target")
        item["last_checked_at"] = checked_at
        if item.get("trusted") and isinstance(target, dict):
            evidence = identity_by_key.get((target.get("pid"), target.get("create_time")), target)
            sid = evidence.get("sid")
            if (sid in sessions and not survivors and not unknown
                    and not unconfirmed_sessions and not hard_read_error):
                item["session_checked"] = True
                item["last_check"] = "session_complete"
                item["resolution"] = "owned_session_checked"
                item["resolved"] = True
                continue
        if item.get("stage") in {"legacy", "descendant_scan", "supervisor_check"} and complete_session_scan:
            item["last_check"] = "session_complete"
            item["resolved"] = True
            continue
        unresolved_errors.append(item)
    for item in registration_errors:
        path_value = item.get("_path")
        if path_value:
            stored = {key: value for key, value in item.items() if key != "_path"}
            try:
                _atomic_json(Path(path_value), stored)
            except OSError as exc:
                unresolved_errors.append({"stage": "journal_write", "error": str(exc)})
    if error_path.exists() and any(item.get("id") == "legacy" and item.get("resolved") for item in registration_errors):
        try:
            _atomic_json(error_path, {"allocation_id": allocation_id, "resolved": True,
                                      "last_checked_at": checked_at, "last_check": "session_complete"})
        except OSError as exc:
            unresolved_errors.append({"stage": "journal_write", "error": str(exc)})
    missing_directory = not processes_dir.is_dir()
    if missing_directory and not supervisor:
        unresolved_errors.append({"stage": "base_registration", "error": "process registration directory is missing"})
    if errors_dir.is_dir():
        new_error_paths = {str(path) for path in errors_dir.glob("*.json")} - initial_error_paths
        if new_error_paths:
            unresolved_errors.append({"stage": "scan_in_progress",
                                      "error": "registration changed during cleanup",
                                      "paths": sorted(new_error_paths)})
    recovered_registration = recovered_base or any(item.get("resolved") and item.get("last_check") == "live"
                                                    for item in registration_errors)
    return {"complete": not survivors and not unknown and not unconfirmed_sessions and not unresolved_errors, "allocation_id": allocation_id,
            "supervisor_excluded": exclude_supervisor,
            "execution_owner": dict(execution_owner), "process_scope": process_scope.get_process_scope(),
            "checked_at": checked_at, "survivors": sorted(set(survivors)), "unknown": sorted(set(x for x in unknown if x is not None)),
            "unconfirmed_sessions": unconfirmed_sessions,
            "registration_errors": unresolved_errors,
            "recovered_registration": recovered_registration,
            "error": unresolved_errors[0].get("error") if unresolved_errors else None}
