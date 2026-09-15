from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
from datetime import datetime, timezone

UUID_RE = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _uuid_text(value: CUuuid) -> str:
    raw = bytes(value.bytes).hex()
    return f"GPU-{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def _load_driver():
    candidates = [ctypes.util.find_library("cuda"), "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            pass
    raise RuntimeError("CUDA driver library is unavailable")


def probe() -> dict:
    sampled_at = datetime.now(timezone.utc).isoformat()
    try:
        driver = _load_driver()
        if driver.cuInit(0) != 0:
            raise RuntimeError("cuInit failed")
        count = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
            raise RuntimeError("cuDeviceGetCount failed")
        bindings = []
        complete = True
        uuid_fn = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
        for ordinal in range(count.value):
            device = ctypes.c_int()
            errors = []
            if driver.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
                bindings.append({"ordinal": ordinal, "gpu_uuid": None, "pci_bus_id": None, "status": "failed", "error": "cuDeviceGet failed"})
                complete = False
                continue
            uuid = CUuuid()
            uuid_text = None
            if uuid_fn(ctypes.byref(uuid), device) == 0:
                uuid_text = _uuid_text(uuid)
            else:
                errors.append("cuDeviceGetUuid failed")
            bus = ctypes.create_string_buffer(32)
            pci = None
            if driver.cuDeviceGetPCIBusId(bus, len(bus), device) == 0:
                pci = bus.value.decode(errors="replace")
            else:
                errors.append("cuDeviceGetPCIBusId failed")
            if uuid_text and not UUID_RE.fullmatch(uuid_text):
                errors.append("CUDA returned a non-physical or invalid GPU UUID")
                uuid_text = None
            status = "success" if not errors else "partial" if uuid_text else "failed"
            bindings.append({"ordinal": ordinal, "gpu_uuid": uuid_text, "pci_bus_id": pci, "status": status, "error": "; ".join(errors) or None})
            complete = complete and uuid_text is not None
        valid = [item for item in bindings if item["gpu_uuid"] and item["status"] in {"success", "partial"}]
        status = "empty" if count.value == 0 else "success" if valid else "failed"
        return {"status": status, "sampled_at": sampled_at, "bindings": bindings, "complete": complete and len(valid) == count.value,
                "error": None if status != "failed" else "CUDA could not identify any enumerated device"}
    except Exception as exc:
        message = str(exc)
        status = "unavailable" if "library is unavailable" in message else "failed"
        return {"status": status, "sampled_at": sampled_at, "bindings": [], "complete": False, "error": message}


def main() -> int:
    print(json.dumps(probe(), separators=(",", ":")))
    return 0
