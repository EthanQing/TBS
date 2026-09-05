from __future__ import annotations

from typing import Any

from train_platform.utils.exceptions import ValidationError

from .contract import TrainerPlugin, TrainingCallbacks, TrainingExecutionSpec


class CustomSourceTrainer:
    """Built-in trainer plugin for custom user-uploaded source models.
    
    In this foundation stage, implemented = False guards against execution.
    Configuration schema and normalization are provided for architectural consistency.
    """

    plugin_id: str = "custom-source"
    name: str = "custom-source"
    display_name: str = "Custom Source Model"
    implemented: bool = False

    def can_handle(self, model_family: str) -> bool:
        # custom-source cannot be selected implicitly by model_family;
        # it must be selected explicitly via engine="custom-source".
        return False

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "custom_args": {
                    "type": "object",
                    "description": "User-defined training configuration arguments passed to custom trainer",
                },
            },
            "additionalProperties": False,
        }

    def normalize_config(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValidationError("custom-source framework_config must be an object")

        unknown_keys = set(raw.keys()) - {"custom_args"}
        if unknown_keys:
            raise ValidationError(
                f"Unknown top-level configuration key(s) for custom-source: {sorted(unknown_keys)}. "
                "Only 'custom_args' is allowed."
            )

        if "custom_args" in raw:
            custom_args = raw["custom_args"]
            if custom_args is not None and not isinstance(custom_args, dict):
                raise ValidationError("'custom_args' must be a JSON object")
            return {"custom_args": dict(custom_args or {})}

        return {}

    def run(self, spec: TrainingExecutionSpec, callbacks: TrainingCallbacks) -> None:
        raise RuntimeError("custom-source runtime is not implemented yet")


__all__ = ["CustomSourceTrainer"]
