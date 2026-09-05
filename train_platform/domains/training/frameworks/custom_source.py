from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from train_platform.platform.filesystem import clear_directory, extract_archive
from train_platform.platform.runtime.custom_training import run_custom_training
from train_platform.utils.exceptions import ValidationError

from train_platform.domains.training.custom_models.storage import (
    compute_file_sha256,
    resolve_package_archive,
)

from .contract import TrainerPlugin, TrainingCallbacks, TrainingExecutionSpec


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_value(item) for item in value]
    return value


class CustomSourceTrainer:
    """Adapter from the training plugin contract to the custom child runtime."""

    plugin_id: str = "custom-source"
    name: str = "custom-source"
    display_name: str = "Custom Source Model"
    implemented: bool = True

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
        package_spec = spec.custom_source
        if package_spec is None:
            raise ValidationError("custom-source execution requires a package snapshot")

        archive_path = resolve_package_archive(package_spec.package_id)
        expected_sha256 = str(package_spec.source_sha256 or "").strip().lower()
        if not expected_sha256:
            raise ValidationError("custom-source execution requires a package checksum snapshot")
        actual_sha256 = compute_file_sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValidationError(
                f"Custom model package checksum mismatch for package_id={package_spec.package_id}"
            )

        run_dir = Path(spec.run_dir)
        custom_dir = run_dir / "custom_model"
        source_workspace = custom_dir / "source"
        output_dir = custom_dir / "output"
        context_path = custom_dir / "custom_training_context.json"
        cancel_marker_path = custom_dir / "custom_training.cancel"

        custom_dir.mkdir(parents=True, exist_ok=True)
        clear_directory(source_workspace)
        source_root = extract_archive(archive_path, source_workspace)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = package_spec.manifest
        entrypoint = manifest.get("entrypoint")
        if not isinstance(entrypoint, Mapping):
            raise ValidationError("Custom model package manifest is missing entrypoint")
        if not str(entrypoint.get("module") or "").strip() or not str(entrypoint.get("class") or "").strip():
            raise ValidationError("Custom model package manifest entrypoint is incomplete")

        custom_args = spec.framework_config.get("custom_args") or {}
        if not isinstance(custom_args, Mapping):
            raise ValidationError("custom_args must be a JSON object")

        context = {
            "run_id": str(spec.run_id),
            "dataset_path": str(Path(spec.dataset_path).resolve(strict=False)),
            "source_root": str(Path(source_root).resolve(strict=False)),
            "output_dir": str(output_dir.resolve(strict=False)),
            "epochs": int(spec.epochs),
            "batch_size": int(spec.batch_size),
            "image_size": int(spec.image_size),
            "learning_rate": float(spec.learning_rate),
            "optimizer": str(spec.optimizer),
            "workers": int(spec.workers),
            "device": str(spec.runtime_device),
            "custom_args": _plain_json_value(custom_args),
            "manifest": _plain_json_value(manifest),
            "cancel_marker_path": str(cancel_marker_path.resolve(strict=False)),
        }

        exit_code = run_custom_training(
            context,
            context_path=context_path,
            cancel_marker_path=cancel_marker_path,
            cancel_requested=callbacks.cancel_requested,
            on_metrics=callbacks.upsert_epoch_metrics,
            on_log=lambda message: print(f"[custom-source] {message}", flush=True),
        )
        if exit_code != 0:
            raise RuntimeError(f"Custom training subprocess exited with code {exit_code}")


__all__ = ["CustomSourceTrainer"]
