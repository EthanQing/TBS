"""Training framework contracts, registry, and concrete plugins."""

from .contract import (
    CancelRequestedFn,
    CustomSourceExecutionSpec,
    ReportArtifactFn,
    TrainingArtifactReport,
    TrainingCallbacks,
    TrainingExecutionSpec,
    TrainerPlugin,
    UpsertEpochMetricsFn,
    validate_artifact_path,
    validate_artifact_role,
)
from .registry import FrameworkPluginInfo, get_plugin, get_trainer, list_plugins, register_plugin
from .architectures import create_architecture, list_architectures

__all__ = [
    "CancelRequestedFn",
    "CustomSourceExecutionSpec",
    "ReportArtifactFn",
    "TrainingArtifactReport",
    "create_architecture",
    "FrameworkPluginInfo",
    "TrainingCallbacks",
    "TrainingExecutionSpec",
    "TrainerPlugin",
    "UpsertEpochMetricsFn",
    "validate_artifact_path",
    "validate_artifact_role",
    "get_plugin",
    "get_trainer",
    "list_plugins",
    "list_architectures",
    "register_plugin",
]
