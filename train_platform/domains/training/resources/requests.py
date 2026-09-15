from __future__ import annotations

import re
from typing import Any


GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def normalize_resource_request(
    request: dict[str, Any] | None,
    *,
    engine: str,
    batch_size: Any,
    device: Any = "auto",
) -> dict[str, Any] | None:
    if request is None:
        return None
    if not isinstance(request, dict):
        raise ValueError("resource_request must be an object")
    if str(device or "auto").strip().lower() != "auto":
        raise ValueError("parameters.device must be 'auto' when resource_request is provided")

    selection = str(request.get("selection", "auto")).strip().lower()
    sharing = str(request.get("sharing", "exclusive")).strip().lower()
    if selection not in {"auto", "manual"}:
        raise ValueError("resource_request.selection must be auto or manual")
    if sharing not in {"exclusive", "shared"}:
        raise ValueError("resource_request.sharing must be exclusive or shared")

    gpu_count = _positive_integer(request.get("gpu_count", 1), "resource_request.gpu_count")
    raw_uuids = request.get("gpu_uuids", [])
    if not isinstance(raw_uuids, list):
        raise ValueError("resource_request.gpu_uuids must be an array")
    gpu_uuids: list[str] = []
    for value in raw_uuids:
        if not isinstance(value, str):
            raise ValueError("resource_request.gpu_uuids entries must be strings")
        gpu_uuid = value.strip()
        if not GPU_UUID_RE.fullmatch(gpu_uuid):
            raise ValueError(f"Invalid full GPU UUID: {gpu_uuid}")
        gpu_uuid = "GPU-" + gpu_uuid[4:].lower()
        if gpu_uuid not in gpu_uuids:
            gpu_uuids.append(gpu_uuid)

    if selection == "auto" and gpu_uuids:
        raise ValueError("gpu_uuids must be empty for auto selection")
    if selection == "manual" and len(gpu_uuids) != gpu_count:
        raise ValueError("manual gpu_uuids unique count must equal gpu_count")

    memory = request.get("memory_mib_per_gpu")
    if memory is not None:
        memory = _positive_integer(memory, "resource_request.memory_mib_per_gpu")
    fixed_batch = (
        isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0
    )
    if sharing == "shared":
        if gpu_count != 1:
            raise ValueError("shared resource requests must use gpu_count=1")
        if memory is None:
            raise ValueError("shared resource requests require positive memory_mib_per_gpu")
        if not fixed_batch:
            raise ValueError("shared resource requests require a fixed positive integer batch_size")

    if gpu_count > 1:
        if sharing != "exclusive":
            raise ValueError("multi-GPU resource requests must use exclusive sharing")
        if str(engine).lower() != "ultralytics-yolo":
            raise ValueError(
                "Multi-GPU resource requests are currently only supported by Ultralytics"
            )
        if not fixed_batch:
            raise ValueError("multi-GPU resource requests require a fixed positive integer batch_size")
        if batch_size % gpu_count:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by gpu_count ({gpu_count})"
            )

    node_id = request.get("node_id")
    if node_id is not None and not isinstance(node_id, str):
        raise ValueError("resource_request.node_id must be a string or null")
    node_id = (node_id.strip() or None) if node_id else None
    if node_id and len(node_id) > 128:
        raise ValueError("resource_request.node_id must be at most 128 characters")

    return {
        "selection": selection,
        "gpu_count": gpu_count,
        "gpu_uuids": gpu_uuids,
        "memory_mib_per_gpu": memory,
        "sharing": sharing,
        "node_id": node_id,
    }
