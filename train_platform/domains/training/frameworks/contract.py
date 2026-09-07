from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


CancelRequestedFn = Callable[[], bool]
UpsertEpochMetricsFn = Callable[[int, Mapping[str, float]], None]


@dataclass(frozen=True)
class TrainingCallbacks:
    """Callbacks supplied by the execution adapter to a framework plugin."""

    cancel_requested: CancelRequestedFn
    upsert_epoch_metrics: UpsertEpochMetricsFn


@dataclass(frozen=True)
class TrainingExecutionSpec:
    """Persisted training configuration materialized for one plugin run."""

    run_id: str
    dataset_path: Path
    dataset_name: str | None
    run_dir: Path
    engine: str
    family: str
    variant: str
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    lr_scheduler: str
    patience: int
    requested_device: str
    runtime_device: str
    workers: int
    optimizer: str
    use_pretrained: bool
    augmentation: Mapping[str, Any] = field(default_factory=dict)
    loss_weights: Mapping[str, Any] = field(default_factory=dict)
    resume_training: bool = False
    resume_job_id: str | None = None
    pretrained_model_path: str | None = None
    momentum: float | None = None
    weight_decay: float | None = None
    warmup_epochs: float | None = None
    warmup_momentum: float | None = None
    warmup_bias_lr: float | None = None
    framework_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("augmentation", "loss_weights", "framework_config"):
            value = getattr(self, field_name)
            if isinstance(value, MappingProxyType):
                continue
            object.__setattr__(self, field_name, MappingProxyType(dict(value or {})))


class TrainerPlugin(Protocol):
    plugin_id: str
    name: str
    display_name: str
    implemented: bool

    def can_handle(self, model_family: str) -> bool: ...

    def get_config_schema(self) -> dict[str, Any]: ...

    def normalize_config(self, raw: dict[str, Any] | None) -> dict[str, Any]: ...

    def run(self, spec: TrainingExecutionSpec, callbacks: TrainingCallbacks) -> None: ...


__all__ = [
    "CancelRequestedFn",
    "TrainingCallbacks",
    "TrainingExecutionSpec",
    "TrainerPlugin",
    "UpsertEpochMetricsFn",
]
