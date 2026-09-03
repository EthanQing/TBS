"""Training framework contracts, registry, and concrete plugins."""

from .contract import (
    CancelRequestedFn,
    TrainingCallbacks,
    TrainingExecutionSpec,
    TrainerPlugin,
    UpsertEpochMetricsFn,
)
from .registry import FrameworkPluginInfo, get_plugin, get_trainer, list_plugins, register_plugin
from .architectures import create_architecture, list_architectures

__all__ = [
    "CancelRequestedFn",
    "create_architecture",
    "FrameworkPluginInfo",
    "TrainingCallbacks",
    "TrainingExecutionSpec",
    "TrainerPlugin",
    "UpsertEpochMetricsFn",
    "get_plugin",
    "get_trainer",
    "list_plugins",
    "list_architectures",
    "register_plugin",
]
