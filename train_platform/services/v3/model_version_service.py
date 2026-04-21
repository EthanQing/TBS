from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.enums import ModelStage, TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.repositories.v3.model_version_repo import ModelVersionRepository
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class ModelVersionService:
    def __init__(self) -> None:
        self.repo = ModelVersionRepository()

    def list_model_versions(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        run_id: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelVersion]:
        return self.repo.list(db, project_id=project_id, run_id=run_id, stage=stage, skip=skip, limit=limit)

    def get_model_version(self, db: Session, model_version_id: int) -> ModelVersion:
        mv = self.repo.get(db, int(model_version_id))
        if not mv:
            raise NotFoundError("Model version not found")
        return mv

    def register_from_run(
        self,
        db: Session,
        *,
        run_id: str,
        version: str,
        stage: ModelStage,
        description: Optional[str] = None,
    ) -> ModelVersion:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
        if not run:
            raise NotFoundError("Training run not found")
        if run.status != TrainingRunStatus.COMPLETED:
            raise ConflictError("Only completed runs can be registered as a model version")

        version = str(version or "").strip()
        if not version:
            raise ValidationError("version is required")

        exists = self.repo.get_by_project_and_version(db, int(run.project_id), version)
        if exists:
            raise ConflictError(f"Model version '{version}' already exists in this project")

        weights_path = None
        metrics = None
        if run.result is not None:
            metrics = run.result.best_metrics or run.result.final_metrics
            weights_path = run.result.best_weights_path or run.result.last_weights_path

        row = ModelVersion(
            project_id=int(run.project_id),
            run_id=str(run.run_id),
            version=version,
            stage=stage,
            description=description,
            metrics=metrics,
            weights_path=weights_path,
        )
        db.add(row)

        if stage == ModelStage.PRODUCTION:
            # Ensure at most one production model per project.
            db.query(ModelVersion).filter(
                ModelVersion.project_id == int(run.project_id),
                ModelVersion.stage == ModelStage.PRODUCTION,
                ModelVersion.version != version,
            ).update({ModelVersion.stage: ModelStage.DEPRECATED})

        db.commit()
        db.refresh(row)
        return row

    def update_model_version(self, db: Session, model_version_id: int, *, patch: dict) -> ModelVersion:
        row = self.get_model_version(db, model_version_id)

        if "version" in patch and patch["version"] is not None:
            new_version = str(patch["version"]).strip()
            if not new_version:
                raise ValidationError("version cannot be empty")
            exists = self.repo.get_by_project_and_version(db, int(row.project_id), new_version)
            if exists and int(exists.model_version_id) != int(row.model_version_id):
                raise ConflictError(f"Model version '{new_version}' already exists in this project")
            row.version = new_version

        if "stage" in patch and patch["stage"] is not None:
            row.stage = patch["stage"]

        if "description" in patch:
            row.description = patch["description"]

        if row.stage == ModelStage.PRODUCTION:
            db.query(ModelVersion).filter(
                ModelVersion.project_id == int(row.project_id),
                ModelVersion.model_version_id != int(row.model_version_id),
                ModelVersion.stage == ModelStage.PRODUCTION,
            ).update({ModelVersion.stage: ModelStage.DEPRECATED})

        db.commit()
        db.refresh(row)
        return row

