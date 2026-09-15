from __future__ import annotations

import os
from typing import Any, Mapping

import psutil

from train_platform.platform.runtime import process_scope


def process_identity(pid: int, **extra: Any) -> dict[str, Any]:
    current_scope = process_scope.get_process_scope()
    owner = extra.get("execution_owner")
    if isinstance(owner, Mapping):
        comparison = process_scope.compare_process_scope(owner.get("process_scope"), current_scope)
        if comparison != "same":
            raise ValueError(f"execution owner process scope is {comparison}")
    process = psutil.Process(int(pid))
    identity = {"pid": int(pid), "create_time": float(process.create_time()),
                "process_scope": current_scope, **extra}
    if os.name != "nt":
        try:
            identity["pgid"] = int(os.getpgid(int(pid)))
            identity["sid"] = int(os.getsid(int(pid)))
        except OSError:
            pass
    return identity


def process_state(identity: Mapping[str, Any]) -> tuple[str, psutil.Process | None]:
    if process_scope.compare_process_scope(
        process_scope.identity_process_scope(identity), process_scope.get_process_scope()
    ) != "same":
        return "unknown", None
    try:
        process = psutil.Process(int(identity["pid"]))
        if float(process.create_time()) != float(identity["create_time"]):
            return "dead", None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return "dead", None
        return "live", process
    except psutil.NoSuchProcess:
        return "dead", None
    except (psutil.AccessDenied, OSError, KeyError, TypeError, ValueError):
        return "unknown", None
