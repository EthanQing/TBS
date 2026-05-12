from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train_platform.utils.training_augmentations import is_ultralytics_engine, normalize_task_type


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
