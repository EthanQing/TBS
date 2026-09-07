from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal


AugmentationValueType = Literal["number", "integer", "enum"]


@dataclass(frozen=True)
class AugmentationFieldSpec:
    key: str
    label: str
    group: str
    value_type: AugmentationValueType
    default: Any = None
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    options: tuple[Any, ...] = ()
    nullable: bool = False
    tasks: tuple[str, ...] = ("detection", "segmentation")
    description: str | None = None

    def to_meta(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "group": self.group,
            "value_type": self.value_type,
            "default": self.default,
            "tasks": list(self.tasks),
            "nullable": bool(self.nullable),
        }
        if self.min is not None:
            data["min"] = self.min
        if self.max is not None:
            data["max"] = self.max
        if self.step is not None:
            data["step"] = self.step
        if self.options:
            data["options"] = list(self.options)
        if self.description:
            data["description"] = self.description
        return data


ULTRALYTICS_AUGMENTATION_SPECS: tuple[AugmentationFieldSpec, ...] = (
    # Color-space augmentation
    AugmentationFieldSpec(
        key="hsv_h",
        label="HSV Hue",
        group="color",
        value_type="number",
        default=0.015,
        min=0,
        max=1,
        step=0.001,
        description="HSV hue augmentation fraction.",
    ),
    AugmentationFieldSpec(
        key="hsv_s",
        label="HSV Saturation",
        group="color",
        value_type="number",
        default=0.7,
        min=0,
        max=1,
        step=0.01,
        description="HSV saturation augmentation fraction.",
    ),
    AugmentationFieldSpec(
        key="hsv_v",
        label="HSV Value",
        group="color",
        value_type="number",
        default=0.4,
        min=0,
        max=1,
        step=0.01,
        description="HSV value/brightness augmentation fraction.",
    ),
    AugmentationFieldSpec(
        key="bgr",
        label="BGR Channel Swap",
        group="color",
        value_type="number",
        default=0.0,
        min=0,
        max=1,
        step=0.01,
        tasks=("detection", "segmentation", "classification"),
        description="Probability of swapping RGB/BGR color channels.",
    ),
    # Geometric augmentation
    AugmentationFieldSpec(
        key="degrees",
        label="Rotation Degrees",
        group="geometry",
        value_type="number",
        default=0.0,
        min=0,
        max=180,
        step=0.1,
        description="Maximum random rotation angle in degrees.",
    ),
    AugmentationFieldSpec(
        key="translate",
        label="Translate",
        group="geometry",
        value_type="number",
        default=0.1,
        min=0,
        max=1,
        step=0.01,
        description="Maximum random translation fraction.",
    ),
    AugmentationFieldSpec(
        key="scale",
        label="Scale",
        group="geometry",
        value_type="number",
        default=0.5,
        min=0,
        max=1,
        step=0.01,
        description="Image gain scale factor.",
    ),
    AugmentationFieldSpec(
        key="shear",
        label="Shear",
        group="geometry",
        value_type="number",
        default=0.0,
        min=-180,
        max=180,
        step=0.1,
        description="Maximum random shear angle in degrees.",
    ),
    AugmentationFieldSpec(
        key="perspective",
        label="Perspective",
        group="geometry",
        value_type="number",
        default=0.0,
        min=0,
        max=0.001,
        step=0.00001,
        description="Random perspective transformation magnitude.",
    ),
    AugmentationFieldSpec(
        key="flipud",
        label="Flip Up-Down",
        group="geometry",
        value_type="number",
        default=0.0,
        min=0,
        max=1,
        step=0.01,
        description="Probability of vertical flip.",
    ),
    AugmentationFieldSpec(
        key="fliplr",
        label="Flip Left-Right",
        group="geometry",
        value_type="number",
        default=0.5,
        min=0,
        max=1,
        step=0.01,
        description="Probability of horizontal flip.",
    ),
    # Mixed-image augmentation
    AugmentationFieldSpec(
        key="mosaic",
        label="Mosaic",
        group="mix",
        value_type="number",
        default=1.0,
        min=0,
        max=1,
        step=0.01,
        description="Probability of mosaic augmentation.",
    ),
    AugmentationFieldSpec(
        key="mixup",
        label="MixUp",
        group="mix",
        value_type="number",
        default=0.0,
        min=0,
        max=1,
        step=0.01,
        description="Probability of MixUp augmentation.",
    ),
    AugmentationFieldSpec(
        key="cutmix",
        label="CutMix",
        group="mix",
        value_type="number",
        default=0.0,
        min=0,
        max=1,
        step=0.01,
        description="Probability of CutMix augmentation.",
    ),
    AugmentationFieldSpec(
        key="close_mosaic",
        label="Close Mosaic Epochs",
        group="mix",
        value_type="integer",
        default=10,
        min=0,
        max=10000,
        step=1,
        description="Disable mosaic augmentation for the last N epochs.",
    ),
    # Segmentation-focused augmentation
    AugmentationFieldSpec(
        key="copy_paste",
        label="Copy-Paste",
        group="segmentation",
        value_type="number",
        default=0.0,
        min=0,
        max=1,
        step=0.01,
        tasks=("segmentation",),
        description="Probability of copy-paste augmentation for segmentation.",
    ),
    AugmentationFieldSpec(
        key="copy_paste_mode",
        label="Copy-Paste Mode",
        group="segmentation",
        value_type="enum",
        default="flip",
        options=("flip", "mixup"),
        tasks=("segmentation",),
        description="Copy-paste augmentation strategy.",
    ),
    # Classification-focused augmentation
    AugmentationFieldSpec(
        key="auto_augment",
        label="AutoAugment Policy",
        group="classification",
        value_type="enum",
        default="randaugment",
        options=("randaugment", "autoaugment", "augmix"),
        nullable=True,
        tasks=("classification",),
        description="Classification auto augmentation policy; null disables it.",
    ),
    AugmentationFieldSpec(
        key="erasing",
        label="Random Erasing",
        group="classification",
        value_type="number",
        default=0.4,
        min=0,
        max=0.9,
        step=0.01,
        tasks=("classification",),
        description="Probability of random erasing for classification.",
    ),
)

ULTRALYTICS_AUGMENTATION_SPEC_BY_KEY: dict[str, AugmentationFieldSpec] = {
    spec.key: spec for spec in ULTRALYTICS_AUGMENTATION_SPECS
}


def is_ultralytics_engine(engine: Any) -> bool:
    return str(engine or "").strip().lower() == "ultralytics-yolo"


def normalize_task_type(task_type: Any) -> str:
    raw = str(getattr(task_type, "value", task_type) or "").strip().lower()
    if raw in {"seg", "segment"}:
        return "segmentation"
    if raw in {"cls", "class"}:
        return "classification"
    return raw or "detection"


def get_training_augmentation_options(*, engine: Any = "ultralytics-yolo", task_type: Any = "detection") -> dict[str, Any]:
    engine_key = str(engine or "ultralytics-yolo").strip().lower() or "ultralytics-yolo"
    task_key = normalize_task_type(task_type)
    specs = []
    if is_ultralytics_engine(engine_key):
        specs = [spec.to_meta() for spec in ULTRALYTICS_AUGMENTATION_SPECS if task_key in spec.tasks]
    return {
        "engine": engine_key,
        "task_type": task_key,
        "defaults_policy": "omit_uses_ultralytics_defaults",
        "fields": specs,
    }


def normalize_training_augmentation(
    raw: Any,
    *,
    engine: Any,
    task_type: Any,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("parameters.augmentation must be an object")
    if not raw:
        return None

    engine_key = str(engine or "").strip().lower()
    task_key = normalize_task_type(task_type)
    if not is_ultralytics_engine(engine_key):
        raise ValueError("parameters.augmentation is currently only supported for Ultralytics YOLO / RT-DETR")

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        k = str(key or "").strip()
        if not k:
            continue
        spec = ULTRALYTICS_AUGMENTATION_SPEC_BY_KEY.get(k)
        if spec is None:
            raise ValueError(f"Unsupported augmentation parameter: {k}")
        if task_key not in spec.tasks:
            raise ValueError(f"Augmentation parameter '{k}' is not supported for task_type '{task_key}'")
        normalized[k] = _normalize_value(spec, value)

    return normalized or None


def _normalize_value(spec: AugmentationFieldSpec, value: Any) -> Any:
    if value is None:
        if spec.nullable:
            return None
        raise ValueError(f"augmentation.{spec.key} must not be null")

    if spec.value_type == "enum":
        text = str(value).strip().lower()
        allowed = {str(option).lower(): option for option in spec.options}
        if text in allowed:
            return allowed[text]
        if spec.nullable and text in {"", "none", "null"}:
            return None
        raise ValueError(f"augmentation.{spec.key} must be one of: {', '.join(map(str, spec.options))}")

    if isinstance(value, bool):
        raise ValueError(f"augmentation.{spec.key} must be a {spec.value_type}")

    if spec.value_type == "integer":
        try:
            num_f = float(str(value).strip()) if isinstance(value, str) else float(value)
        except Exception as e:
            raise ValueError(f"augmentation.{spec.key} must be an integer") from e
        if not num_f.is_integer():
            raise ValueError(f"augmentation.{spec.key} must be an integer")
        num_i = int(num_f)
        _check_bounds(spec, num_i)
        return num_i

    try:
        num = float(str(value).strip()) if isinstance(value, str) else float(value)
    except Exception as e:
        raise ValueError(f"augmentation.{spec.key} must be a number") from e
    _check_bounds(spec, num)
    return num


def _check_bounds(spec: AugmentationFieldSpec, value: float | int) -> None:
    if spec.min is not None and value < spec.min:
        raise ValueError(f"augmentation.{spec.key} must be >= {spec.min}")
    if spec.max is not None and value > spec.max:
        raise ValueError(f"augmentation.{spec.key} must be <= {spec.max}")



@dataclass(frozen=True)
class LossWeightSpec:
    key: str
    label: str
    default: float
    min: float = 0.0
    step: float = 0.1
    tasks: tuple[str, ...] = ("detection", "segmentation")
    description: str | None = None

    def to_meta(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "value_type": "number",
            "default": self.default,
            "min": self.min,
            "step": self.step,
            "tasks": list(self.tasks),
        }
        if self.description:
            data["description"] = self.description
        return data


ULTRALYTICS_LOSS_WEIGHT_SPECS: tuple[LossWeightSpec, ...] = (
    LossWeightSpec(
        key="box",
        label="Box Loss Weight",
        default=7.5,
        description="Box regression loss gain.",
    ),
    LossWeightSpec(
        key="cls",
        label="Classification Loss Weight",
        default=0.5,
        description="Classification loss gain.",
    ),
    LossWeightSpec(
        key="dfl",
        label="Distribution Focal Loss Weight",
        default=1.5,
        description="Distribution focal loss gain.",
    ),
)

ULTRALYTICS_LOSS_WEIGHT_SPEC_BY_KEY: dict[str, LossWeightSpec] = {
    spec.key: spec for spec in ULTRALYTICS_LOSS_WEIGHT_SPECS
}


def get_training_loss_weight_options(*, engine: Any = "ultralytics-yolo", task_type: Any = "detection") -> dict[str, Any]:
    engine_key = str(engine or "ultralytics-yolo").strip().lower() or "ultralytics-yolo"
    task_key = normalize_task_type(task_type)
    specs = []
    if is_ultralytics_engine(engine_key):
        specs = [spec.to_meta() for spec in ULTRALYTICS_LOSS_WEIGHT_SPECS if task_key in spec.tasks]
    return {
        "engine": engine_key,
        "task_type": task_key,
        "defaults_policy": "omit_uses_ultralytics_defaults",
        "fields": specs,
    }


def normalize_training_loss_weights(raw: Any, *, engine: Any, task_type: Any) -> dict[str, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("parameters.loss_weights must be an object")
    if not raw:
        return None

    engine_key = str(engine or "").strip().lower()
    task_key = normalize_task_type(task_type)
    if not is_ultralytics_engine(engine_key):
        raise ValueError("parameters.loss_weights is currently only supported for Ultralytics YOLO")

    normalized: dict[str, float] = {}
    for key, value in raw.items():
        k = str(key or "").strip()
        if not k:
            continue
        spec = ULTRALYTICS_LOSS_WEIGHT_SPEC_BY_KEY.get(k)
        if spec is None:
            raise ValueError(f"Unsupported loss weight parameter: {k}")
        if task_key not in spec.tasks:
            raise ValueError(f"Loss weight parameter '{k}' is not supported for task_type '{task_key}'")
        normalized[k] = _normalize_loss_weight_value(spec, value)

    return normalized or None


def _normalize_loss_weight_value(spec: LossWeightSpec, value: Any) -> float:
    if value is None:
        raise ValueError(f"loss_weights.{spec.key} must not be null")
    if isinstance(value, bool):
        raise ValueError(f"loss_weights.{spec.key} must be a number")
    try:
        num = float(str(value).strip()) if isinstance(value, str) else float(value)
    except Exception as e:
        raise ValueError(f"loss_weights.{spec.key} must be a number") from e
    if num < spec.min:
        raise ValueError(f"loss_weights.{spec.key} must be >= {spec.min:g}")
    return num



AUTO_BATCH_SIZE = -1
LR_SCHEDULER_LINEAR = "linear"
LR_SCHEDULER_COSINE = "cosine"
SUPPORTED_LR_SCHEDULERS = {LR_SCHEDULER_LINEAR, LR_SCHEDULER_COSINE}


def normalize_batch_size(value: Any) -> int:
    try:
        batch_size = int(value)
    except Exception as e:
        raise ValueError("batch_size must be an integer") from e

    if batch_size == 0 or batch_size < AUTO_BATCH_SIZE:
        raise ValueError("batch_size must be > 0, or -1 for auto batch")
    return batch_size


def normalize_lr_scheduler(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return LR_SCHEDULER_LINEAR
    if raw in {"linear", "linearlr", "linear_lr"}:
        return LR_SCHEDULER_LINEAR
    if raw in {"cosine", "cos", "cos_lr", "cosinelr", "cosine_lr"}:
        return LR_SCHEDULER_COSINE
    raise ValueError("lr_scheduler must be one of: linear, cosine")


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


def parse_visible_host_gpu_ids(value: Any | None = None) -> list[int] | None:
    """
    Parse a Docker/NVIDIA visible-device list as host GPU ids.

    Returns None when the runtime exposes all GPUs or uses non-numeric device
    handles such as GPU UUIDs, because those cannot be mapped from the numeric
    training task device field.
    """
    raw = os.getenv("NVIDIA_VISIBLE_DEVICES") if value is None else value
    text = str(raw or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered == "all":
        return None
    if lowered in {"none", "void"}:
        return []

    parsed = _parse_gpu_id_tokens(text)
    return _dedupe_gpu_ids(parsed) if parsed else None


def worker_can_run_device(device_spec: Any, visible_host_gpu_ids: list[int] | None = None) -> bool:
    requested = normalize_device_spec(device_spec)
    if requested in {"auto", "cpu"}:
        return True

    selected_gpu_ids = extract_selected_gpu_ids(requested)
    if not selected_gpu_ids or visible_host_gpu_ids is None:
        return True

    visible_set = set(int(idx) for idx in visible_host_gpu_ids)
    return all(int(idx) in visible_set for idx in selected_gpu_ids)


def build_device_runtime(device_spec: Any, visible_host_gpu_ids: list[int] | None = None) -> dict[str, str | None]:
    """
    Build per-process device runtime settings.

    For explicit GPU selection we isolate the training subprocess with
    `CUDA_VISIBLE_DEVICES` and remap the device argument to the local visible
    index space expected by deep learning runtimes.

    Examples:
      - "auto"   -> {"requested": "auto", "runtime_device": "auto", "cuda_visible_devices": None}
      - "cpu"    -> {"requested": "cpu",  "runtime_device": "cpu",  "cuda_visible_devices": ""}
      - "1"      -> {"requested": "1",    "runtime_device": "0",    "cuda_visible_devices": "1"}
      - "2,5"    -> {"requested": "2,5",  "runtime_device": "0,1",  "cuda_visible_devices": "2,5"}

    In a Docker worker already restricted by `NVIDIA_VISIBLE_DEVICES`, pass the
    host GPU ids visible to the container. For example, task device "1" inside a
    container with `NVIDIA_VISIBLE_DEVICES=1` becomes local device "0".
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

    if visible_host_gpu_ids is not None:
        missing = [idx for idx in selected_gpu_ids if idx not in visible_host_gpu_ids]
        if missing:
            visible_text = ",".join(str(idx) for idx in visible_host_gpu_ids) or "<none>"
            raise ValueError(
                f"Requested GPU device(s) {requested} are not visible to this worker "
                f"(NVIDIA_VISIBLE_DEVICES={visible_text})"
            )
        local_gpu_ids = [visible_host_gpu_ids.index(idx) for idx in selected_gpu_ids]
        return {
            "requested": requested,
            "runtime_device": ",".join(str(idx) for idx in range(len(local_gpu_ids))),
            "cuda_visible_devices": ",".join(str(idx) for idx in local_gpu_ids),
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
    normalized["lr_scheduler"] = normalize_lr_scheduler(normalized.get("lr_scheduler", LR_SCHEDULER_LINEAR))
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

