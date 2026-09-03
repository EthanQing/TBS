from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.core.license import assert_valid_license
from train_platform.domains.datasets.storage.paths import resolve_legacy_dataset_path
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.enums import LogLevel, TrainingRunStatus
from train_platform.models.v3.inference import InferenceRun
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.training_run import (
    TrainingRun,
    TrainingRunEvent,
    TrainingRunParameters,
)
from train_platform.platform.filesystem import remove_tree
from train_platform.repositories.v3.training_run_repo import TrainingRunRepository
from train_platform.domains.training.frameworks import get_plugin
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.utils.path_utils import resolve_training_path
from train_platform.utils.training_augmentations import normalize_training_augmentation
from train_platform.utils.training_loss_weights import normalize_training_loss_weights
from train_platform.utils.training_params import validate_training_params_for_engine

from .lifecycle import (
    queue_run as lifecycle_queue_run,
    request_cancel as lifecycle_request_cancel,
    request_delete as lifecycle_request_delete,
    resume_run as lifecycle_resume_run,
)


class TrainingRunService:
    """Training run aggregate operations and lifecycle orchestration."""

    def __init__(self) -> None:
        self.runs = TrainingRunRepository()

    def get_run(self, db: Session, run_id: str) -> TrainingRun:
        run = self.runs.get(db, str(run_id))
        if not run:
            raise NotFoundError("Training run not found")
        return run

    def list_runs(
        self,
        db: Session,
        *,
        project_id: int | None = None,
        status: TrainingRunStatus | None = None,
        standard_dataset_id: int | None = None,
        architecture_id: int | None = None,
        skip: int = 0,
        limit: int = 100,
        include_hidden: bool = False,
    ) -> list[TrainingRun]:
        return self.runs.list(
            db,
            project_id=project_id,
            status=status,
            standard_dataset_id=standard_dataset_id,
            architecture_id=architecture_id,
            skip=skip,
            limit=limit,
            include_hidden=include_hidden,
        )

    def create_run(self, db: Session, *, obj: dict[str, Any]) -> TrainingRun:
        assert_valid_license()
        project_id = int(obj["project_id"])
        architecture_id = int(obj["architecture_id"])
        params = obj["parameters"]

        project = db.query(Project).filter(Project.project_id == project_id).first()
        if not project:
            raise NotFoundError("Project not found")

        dataset = (
            db.query(StandardDataset)
            .filter(StandardDataset.standard_dataset_id == int(project.standard_dataset_id))
            .first()
        )
        if not dataset:
            raise NotFoundError("Standard dataset not found")

        arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == architecture_id).first()
        if not arch:
            raise NotFoundError("Architecture not found")
        if arch.task_type != project.task_type:
            raise ValidationError("Architecture task_type does not match project task_type")
        arch_engine = str(getattr(arch, "engine", "") or "").strip().lower()
        try:
            plugin = get_plugin(arch_engine)
        except Exception as exc:
            raise ValidationError(f"Architecture engine is not registered: {arch_engine}") from exc
        if not bool(getattr(plugin, "implemented", True)):
            raise ValidationError(
                f"Architecture engine '{arch_engine}' is not implemented yet; select another framework plugin"
            )

        try:
            params = validate_training_params_for_engine(arch_engine, params)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        try:
            normalized_augmentation = normalize_training_augmentation(
                params.get("augmentation"),
                engine=arch_engine,
                task_type=project.task_type,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        params = dict(params)
        params["augmentation"] = normalized_augmentation
        try:
            normalized_loss_weights = normalize_training_loss_weights(
                params.get("loss_weights"),
                engine=arch_engine,
                task_type=project.task_type,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        params["loss_weights"] = normalized_loss_weights

        additional_params = params.get("additional_params")
        if additional_params is not None and not isinstance(additional_params, dict):
            raise ValidationError("parameters.additional_params must be an object")
        if isinstance(additional_params, dict) and "framework_config" in additional_params:
            framework_config = additional_params.get("framework_config")
            if framework_config is not None and not isinstance(framework_config, dict):
                raise ValidationError("parameters.additional_params.framework_config must be an object")
            try:
                normalized_framework_config = plugin.normalize_config(framework_config or {})
            except Exception as exc:
                raise ValidationError(f"Invalid framework_config for engine '{arch_engine}': {exc}") from exc
            if not isinstance(normalized_framework_config, dict):
                raise ValidationError("framework_config normalize result must be an object")
            params = dict(params)
            params["additional_params"] = dict(additional_params)
            params["additional_params"]["framework_config"] = normalized_framework_config

        has_split = False
        try:
            dataset_root = resolve_legacy_dataset_path(dataset.storage_path)
            if not dataset_root.exists() or not dataset_root.is_dir():
                raise ConflictError("Standard dataset path does not exist; upload dataset files first")
            data_yaml = find_yolo_dataset_yaml(
                dataset_root,
                dataset_name=str(getattr(dataset, "name", "") or "") or None,
            )
            if data_yaml and data_yaml.exists():
                try:
                    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
                except Exception:
                    cfg = {}
                if isinstance(cfg, dict):
                    train_path = cfg.get("train")
                    val_path = cfg.get("val")

                    def _path_ok(value: Any) -> bool:
                        if not value:
                            return False
                        if isinstance(value, (list, tuple)):
                            return all(_path_ok(item) for item in value) if value else False
                        raw_path = str(value).strip()
                        if not raw_path:
                            return False
                        path = Path(raw_path)
                        if not path.is_absolute():
                            path = (dataset_root / path).resolve(strict=False)
                        return path.exists()

                    if _path_ok(train_path) and _path_ok(val_path):
                        has_split = True
        except (ValidationError, ConflictError):
            raise
        except Exception:
            raise ValidationError("Failed to validate dataset split for training")

        run_id = str(uuid.uuid4())
        name = str(obj.get("name") or "").strip() or f"{arch.variant}-{run_id[:8]}"
        run = TrainingRun(
            run_id=run_id,
            project_id=project.project_id,
            standard_dataset_id=int(dataset.standard_dataset_id),
            architecture_id=arch.architecture_id,
            name=name,
            status=TrainingRunStatus.CREATED,
            progress=0,
            current_epoch=0,
            total_epochs=int(params.get("epochs") or 0) if params else None,
            hidden=False,
            run_dir=run_id,
            config=None,
        )
        db.add(run)
        db.flush()
        db.add(
            TrainingRunParameters(
                run_id=run_id,
                epochs=int(params.get("epochs", 100)),
                batch_size=int(params.get("batch_size", 16)),
                image_size=int(params.get("image_size", 640)),
                learning_rate=float(params.get("learning_rate", 0.01)),
                lr_scheduler=str(params.get("lr_scheduler") or "linear"),
                patience=int(params.get("patience", 50)),
                device=str(params.get("device") or "auto"),
                workers=int(params.get("workers", 8)),
                use_pretrained=bool(params.get("use_pretrained", True)),
                optimizer=str(params.get("optimizer") or "AdamW"),
                augmentation=params.get("augmentation"),
                loss_weights=params.get("loss_weights"),
                additional_params=params.get("additional_params"),
            )
        )
        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="created", message="Run created"))
        if not has_split:
            db.add(
                TrainingRunEvent(
                    run_id=run_id,
                    level=LogLevel.WARNING,
                    event_type="dataset_split_missing",
                    message="Standard dataset has no valid train/val split; proceeding without enforced split",
                )
            )
        db.commit()
        return self.get_run(db, run_id)

    def update_run(self, db: Session, run_id: str, *, patch: dict[str, Any]) -> TrainingRun:
        run = self.get_run(db, run_id)
        if "name" in patch and patch["name"] is not None:
            run.name = str(patch["name"]).strip()
        db.commit()
        db.refresh(run)
        return run

    def queue_run(self, db: Session, run_id: str) -> TrainingRun:
        return lifecycle_queue_run(db, run_id)

    def resume_run(self, db: Session, run_id: str) -> TrainingRun:
        run = self.get_run(db, run_id)
        if run.status == TrainingRunStatus.COMPLETED:
            raise ConflictError("Run is COMPLETED and cannot be resumed; create a new training run instead")
        if run.status not in (TrainingRunStatus.CANCELLED, TrainingRunStatus.FAILED):
            raise ConflictError(f"Run status is {run.status}; must be CANCELLED or FAILED to resume")
        weights_path = settings.training_dir / str(run_id) / "weights" / "last.pt"
        return lifecycle_resume_run(db, run_id, has_resume_checkpoint=weights_path.exists())

    def request_cancel(self, db: Session, run_id: str, *, reason: str | None = None) -> TrainingRun:
        return lifecycle_request_cancel(db, run_id, reason=reason)

    def request_delete(self, db: Session, run_id: str) -> TrainingRun:
        return lifecycle_request_delete(db, run_id)

    def delete_run(self, db: Session, run_id: str, *, force: bool = False) -> TrainingRun:
        run = self.get_run(db, run_id)
        model_versions = db.query(ModelVersion).filter(ModelVersion.run_id == str(run.run_id)).all()
        if model_versions and not force:
            detail = f"{len(model_versions)} model version(s)"
            raise ConflictError(f"Cannot delete training run; {detail} still reference it")
        if model_versions and force:
            mv_ids = [int(model_version.model_version_id) for model_version in model_versions]
            dep_ids: list[int] = []
            if mv_ids:
                deployments = db.query(Deployment).filter(Deployment.model_version_id.in_(mv_ids)).all()
                dep_ids = [int(deployment.deployment_id) for deployment in deployments]
                inf_filters = [InferenceRun.model_version_id.in_(mv_ids)]
                if dep_ids:
                    inf_filters.append(InferenceRun.deployment_id.in_(dep_ids))
                for inference in db.query(InferenceRun).filter(or_(*inf_filters)).all():
                    db.delete(inference)
                for deployment in deployments:
                    db.delete(deployment)
            for model_version in model_versions:
                db.delete(model_version)
        run = self.request_delete(db, str(run.run_id))
        if run.status != TrainingRunStatus.RUNNING:
            remove_tree(settings.training_dir / str(run.run_id), ignore_errors=True)
        return run


__all__ = ["TrainingRunService"]
