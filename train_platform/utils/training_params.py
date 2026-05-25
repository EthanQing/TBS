from __future__ import annotations

from typing import Any


AUTO_BATCH_SIZE = -1


def normalize_batch_size(value: Any) -> int:
    try:
        batch_size = int(value)
    except Exception as e:
        raise ValueError("batch_size must be an integer") from e

    if batch_size == 0 or batch_size < AUTO_BATCH_SIZE:
        raise ValueError("batch_size must be > 0, or -1 for auto batch")
    return batch_size


def normalize_device_spec(value: Any) -> str:
    """
    Normalize common training device inputs to a stable backend representation.

    Supported examples:
      - auto / default / "" -> "auto"
      - cpu -> "cpu"
      - gpu / cuda / cuda: -> "0"
      - 0 / "0" / "cuda:0" -> "0"
      - "0,1" / "cuda:0,1" / [0, 1] -> "0,1"

    Unknown non-empty strings are preserved as-is (e.g. "mps") to avoid
    unnecessarily blocking platform-specific runtimes.
    """
    if value is None:
        return "auto"

    if isinstance(value, (list, tuple, set)):
        gpu_ids: list[int] = []
        for item in value:
            gpu_ids.extend(_parse_gpu_id_tokens(str(item)))
        return _join_unique_gpu_ids(gpu_ids) if gpu_ids else "auto"

    if isinstance(value, int):
        if value < 0:
            raise ValueError("device GPU index must be >= 0")
        return str(value)

    raw = str(value).strip()
    if not raw:
        return "auto"

    lowered = raw.lower()
    if lowered in {"auto", "default", "自动"}:
        return "auto"
    if lowered == "cpu":
        return "cpu"
    if lowered in {"gpu", "cuda"}:
        return "0"
    if lowered.startswith("cuda:"):
        suffix = raw.split(":", 1)[1].strip()
        if not suffix:
            return "0"
        parsed = _parse_gpu_id_tokens(suffix)
        if parsed:
            return _join_unique_gpu_ids(parsed)

    parsed = _parse_gpu_id_tokens(raw)
    if parsed:
        return _join_unique_gpu_ids(parsed)

    return raw


def extract_selected_gpu_ids(device_spec: Any) -> list[int]:
    normalized = normalize_device_spec(device_spec)
    if normalized in {"auto", "cpu"}:
        return []

    parsed = _parse_gpu_id_tokens(normalized)
    return _dedupe_gpu_ids(parsed)


def selected_gpu_count(device_spec: Any) -> int:
    return len(extract_selected_gpu_ids(device_spec))


def build_device_runtime(device_spec: Any) -> dict[str, str | None]:
    """
    Build per-process device runtime settings.

    For explicit GPU selection we isolate the training subprocess with
    `CUDA_VISIBLE_DEVICES=<requested ids>` and remap the device argument to the
    local visible index space expected by deep learning runtimes.

    Examples:
      - "auto"   -> {"requested": "auto", "runtime_device": "auto", "cuda_visible_devices": None}
      - "cpu"    -> {"requested": "cpu",  "runtime_device": "cpu",  "cuda_visible_devices": ""}
      - "1"      -> {"requested": "1",    "runtime_device": "0",    "cuda_visible_devices": "1"}
      - "2,5"    -> {"requested": "2,5",  "runtime_device": "0,1",  "cuda_visible_devices": "2,5"}
    """
    requested = normalize_device_spec(device_spec)
    if requested == "cpu":
        return {
            "requested": "cpu",
            "runtime_device": "cpu",
            "cuda_visible_devices": "",
        }

    selected_gpu_ids = extract_selected_gpu_ids(requested)
    if not selected_gpu_ids:
        return {
            "requested": requested,
            "runtime_device": requested,
            "cuda_visible_devices": None,
        }

    return {
        "requested": requested,
        "runtime_device": ",".join(str(idx) for idx in range(len(selected_gpu_ids))),
        "cuda_visible_devices": ",".join(str(idx) for idx in selected_gpu_ids),
    }


def validate_training_params_for_engine(engine: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize and validate training params for the selected backend engine.

    Current policy:
      - batch_size=-1 auto batch is only enabled for Ultralytics YOLO/RT-DETR.
      - explicit multi-GPU device selection is only enabled for Ultralytics.
      - Ultralytics multi-GPU requires a fixed positive batch size divisible by
        the selected GPU count.
    """
    normalized = dict(params or {})
    batch_size = normalize_batch_size(normalized.get("batch_size", 16))
    device = normalize_device_spec(normalized.get("device", "auto"))

    engine_key = str(engine or "").strip().lower()
    gpu_count = selected_gpu_count(device)
    multi_gpu = gpu_count > 1

    if batch_size == AUTO_BATCH_SIZE and engine_key != "ultralytics-yolo":
        raise ValueError("batch_size=-1 auto batch is currently only supported by Ultralytics YOLO / RT-DETR")

    if multi_gpu and engine_key != "ultralytics-yolo":
        raise ValueError(
            f"Multi-GPU device selection is currently only supported by Ultralytics YOLO / RT-DETR; "
            f"engine '{engine_key or 'unknown'}' does not support device='{device}'"
        )

    if engine_key == "ultralytics-yolo" and multi_gpu:
        if batch_size == AUTO_BATCH_SIZE:
            raise ValueError("Ultralytics auto batch (batch_size=-1) only supports single-GPU runs")
        if batch_size % gpu_count != 0:
            raise ValueError(
                f"For Ultralytics multi-GPU runs, batch_size ({batch_size}) must be divisible by "
                f"the selected GPU count ({gpu_count})"
            )

    normalized["batch_size"] = batch_size
    normalized["device"] = device
    return normalized


def _parse_gpu_id_tokens(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return []

    text = text.replace("，", ",").replace(" ", "")
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1]
    text = text.strip("[]()")
    if not text:
        return []

    parts = [part for part in text.split(",") if part != ""]
    if not parts:
        return []

    gpu_ids: list[int] = []
    for part in parts:
        if not part.isdigit():
            return []
        idx = int(part)
        if idx < 0:
            raise ValueError("device GPU indices must be >= 0")
        gpu_ids.append(idx)
    return gpu_ids


def _dedupe_gpu_ids(gpu_ids: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for idx in gpu_ids:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _join_unique_gpu_ids(gpu_ids: list[int]) -> str:
    return ",".join(str(idx) for idx in _dedupe_gpu_ids(gpu_ids))
