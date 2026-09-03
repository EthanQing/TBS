"""Training framework contracts, registry, and concrete plugins."""

from .contract import (
    CancelRequestedFn,
    TrainingCallbacks,
    TrainingExecutionSpec,
    TrainerPlugin,
    UpsertEpochMetricsFn,
)
from .registry import FrameworkPluginInfo, get_plugin, get_trainer, list_plugins, register_plugin

__all__ = [
    "CancelRequestedFn",
    "FrameworkPluginInfo",
    "TrainingCallbacks",
    "TrainingExecutionSpec",
    "TrainerPlugin",
    "UpsertEpochMetricsFn",
    "get_plugin",
    "get_trainer",
    "list_plugins",
    "register_plugin",
]
