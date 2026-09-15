from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal, Mapping


ProcessScopeComparison = Literal["same", "different", "missing", "unavailable"]


def get_process_scope() -> dict[str, Any] | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        namespace = os.stat("/proc/self/ns/pid")
    except (OSError, UnicodeError):
        return None
    if not boot_id:
        return None
    return {
        "boot_id": boot_id,
        "pid_namespace": {"device": int(namespace.st_dev), "inode": int(namespace.st_ino)},
    }


def _normalized_scope(value: object) -> tuple[str, int, int] | None:
    if not isinstance(value, Mapping):
        return None
    boot_id = value.get("boot_id")
    namespace = value.get("pid_namespace")
    if not isinstance(boot_id, str) or not boot_id.strip() or not isinstance(namespace, Mapping):
        return None
    device = namespace.get("device")
    inode = namespace.get("inode")
    if (
        isinstance(device, bool) or not isinstance(device, int) or device < 0
        or isinstance(inode, bool) or not isinstance(inode, int) or inode <= 0
    ):
        return None
    return boot_id.strip(), device, inode


def compare_process_scope(expected: object, actual: object) -> ProcessScopeComparison:
    normalized_expected = _normalized_scope(expected)
    if normalized_expected is None:
        return "missing"
    normalized_actual = _normalized_scope(actual)
    if normalized_actual is None:
        return "unavailable"
    return "same" if normalized_expected == normalized_actual else "different"


def identity_process_scope(identity: object) -> object | None:
    if not isinstance(identity, Mapping):
        return None
    direct = identity.get("process_scope")
    owner = identity.get("execution_owner")
    if "execution_owner" in identity:
        if not isinstance(owner, Mapping):
            return None
        nested = owner.get("process_scope")
        if _normalized_scope(nested) is None:
            return None
        if direct is not None and compare_process_scope(direct, nested) != "same":
            return None
        return nested
    return direct


__all__ = ["compare_process_scope", "get_process_scope", "identity_process_scope"]
