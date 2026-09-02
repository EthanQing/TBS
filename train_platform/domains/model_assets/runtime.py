from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.paddledet_paths import resolve_paddledet_config_path
from train_platform.utils.path_utils import resolve_training_path


@dataclass(frozen=True)
class ModelRuntimeSpec:
    model_version_id: int
    run_id: str
    project_id: int
    engine: str
    family: str | None
    variant: str | None
    weights_path: Path
    config_path: Path | None

    def to_payload(self) -> dict[str, str | int | None]:
        return {
            "model_version_id": int(self.model_version_id),
            "run_id": str(self.run_id),
            "project_id": int(self.project_id),
            "engine": str(self.engine),
            "family": self.family,
            "variant": self.variant,
            "weights_path": str(self.weights_path),
            "config_path": str(self.config_path) if self.config_path else None,
        }


def resolve_architecture_config_path(architecture: ModelArchitecture | None) -> Path | None:
    if not architecture:
        return None
    params = architecture.default_params if isinstance(architecture.default_params, dict) else {}
    raw = params.get("config_path")
    if raw is None:
        return None
    config = str(raw).strip().replace("\\", "/")
    if not config:
        return None
    return resolve_paddledet_config_path(config)


def resolve_model_runtime(
    db: Session,
    *,
    model_version_id: int | None = None,
    model_version: ModelVersion | None = None,
) -> ModelRuntimeSpec:
    if model_version is None:
        if model_version_id is None:
            raise ValidationError("model_version_id or model_version is required")
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == int(model_version_id))
            .first()
        )
    if not model_version:
        raise NotFoundError("Model version not found")

    weights = str(model_version.weights_path or "").strip()
    if not weights:
        raise ValidationError("Model version has no weights_path; register from a completed training run first")

    weights_path = resolve_training_path(weights)
    if not weights_path.exists() or not weights_path.is_file():
        raise NotFoundError(f"Weights file not found: {weights_path}")

    run = None
    if model_version.run_id:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == str(model_version.run_id)).first()

    architecture = None
    if run:
        architecture = (
            db.query(ModelArchitecture)
            .filter(ModelArchitecture.architecture_id == int(run.architecture_id))
            .first()
        )

    engine = str(getattr(architecture, "engine", "") or "ultralytics-yolo").strip().lower()
    family = str(getattr(architecture, "family", "") or "").strip() or None
    variant = str(getattr(architecture, "variant", "") or "").strip() or None

    config_path = None
    if engine == "paddle-det":
        config_path = resolve_architecture_config_path(architecture)
        if not config_path:
            raise ValidationError("Paddle model missing valid config_path in architecture.default_params")

    return ModelRuntimeSpec(
        model_version_id=int(model_version.model_version_id),
        run_id=str(model_version.run_id or ""),
        project_id=int(model_version.project_id),
        engine=engine,
        family=family,
        variant=variant,
        weights_path=weights_path,
        config_path=config_path,
    )


__all__ = ["ModelRuntimeSpec", "resolve_architecture_config_path", "resolve_model_runtime"]
