"""Infrastructure clients for communicating with model worker processes."""

from .model_workers import ModelWorkerClient, ModelWorkerError
from .paddledetection import (
    PADDLE_DET_REQUIRED_CONFIG,
    is_paddledet_repo,
    paddledet_missing_message,
    resolve_paddledet_config_path,
    resolve_paddledet_repo,
)
from .ultralytics import apply_torch_safe_load_patches

__all__ = [
    "ModelWorkerClient",
    "ModelWorkerError",
    "PADDLE_DET_REQUIRED_CONFIG",
    "apply_torch_safe_load_patches",
    "is_paddledet_repo",
    "paddledet_missing_message",
    "resolve_paddledet_config_path",
    "resolve_paddledet_repo",
]

