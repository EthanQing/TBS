from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


CancelRequestedFn = Callable[[], bool]
UpsertEpochMetricsFn = Callable[[int, Mapping[str, float]], None]
ReportArtifactFn = Callable[["TrainingArtifactReport"], None]

_ARTIFACT_ROLE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def validate_artifact_role(role: str) -> str:
    if not isinstance(role, str):
        raise TypeError("artifact role must be a string")
    value = role
    if not _ARTIFACT_ROLE_RE.fullmatch(value):
        raise ValueError("artifact role must match [a-z][a-z0-9_.-]* and be at most 64 characters")
    return value


def validate_artifact_path(path: str | Path) -> str:
    if not isinstance(path, (str, Path)):
        raise TypeError("artifact path must be a string or Path")
    value = str(path)
    if not value or "\x00" in value:
        raise ValueError("artifact path must be a non-empty relative path")
    posix_path = Path(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.root or windows_path.drive or value.startswith(("\\\\", "/")):
        raise ValueError("artifact path must be relative to the training output directory")
    parts = value.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise ValueError("artifact path must not contain parent traversal")
    normalized_parts = [part for part in parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError("artifact path must identify a file below the training output directory")
    return Path(*normalized_parts).as_posix()


def _freeze_manifest(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_manifest(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_manifest(item) for item in value)
    return value


@dataclass(frozen=True)
class TrainingCallbacks:
    """Callbacks supplied by the execution adapter to a framework plugin."""

    cancel_requested: CancelRequestedFn
    upsert_epoch_metrics: UpsertEpochMetricsFn
    report_artifact: ReportArtifactFn


@dataclass(frozen=True)
class TrainingArtifactReport:
    """Framework-neutral intent to register one training output artifact."""

    role: str
    path: str | Path
    format: str | None = None
    meta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", validate_artifact_role(self.role))
        object.__setattr__(self, "path", validate_artifact_path(self.path))
        if self.format is not None and not isinstance(self.format, str):
            raise TypeError("artifact format must be a string")
        if self.meta is not None:
            if not isinstance(self.meta, Mapping):
                raise TypeError("artifact meta must be a mapping")
            object.__setattr__(self, "meta", _freeze_manifest(self.meta))


@dataclass(frozen=True)
class CustomSourceExecutionSpec:
    """Immutable package snapshot required by the custom-source adapter."""

    package_id: int
    source_sha256: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        object.__setattr__(self, "manifest", _freeze_manifest(self.manifest))


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
    custom_source: CustomSourceExecutionSpec | None = None

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
    "CustomSourceExecutionSpec",
    "ReportArtifactFn",
    "TrainingArtifactReport",
    "TrainingCallbacks",
    "TrainingExecutionSpec",
    "TrainerPlugin",
    "UpsertEpochMetricsFn",
    "validate_artifact_path",
    "validate_artifact_role",
]
