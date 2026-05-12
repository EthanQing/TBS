from __future__ import annotations

from dataclasses import dataclass
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
