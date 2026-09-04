from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.enums import ModelStage, TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class ModelVersionService:
    def list_model_versions_page(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        run_id: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[ModelVersion], int]:
        query = db.query(ModelVersion)
        if project_id is not None:
            query = query.filter(ModelVersion.project_id == int(project_id))
        if run_id:
            query = query.filter(ModelVersion.run_id == str(run_id))
        if stage is not None:
            query = query.filter(ModelVersion.stage == stage)
        total = int(query.count())
        items = (
            query.order_by(ModelVersion.updated_at.desc())
            .offset(max(0, int(skip)))
            .limit(max(0, int(limit)))
            .all()
        )
        return items, total

    def get_model_version(self, db: Session, model_version_id: int) -> ModelVersion:
        model_version = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
        if not model_version:
            raise NotFoundError("Model version not found")
        return model_version

    def resolve_for_execution(
        self,
        db: Session,
        *,
        model_version_id: int | None = None,
        run_id: str | None = None,
        description: Optional[str] = None,
    ) -> ModelVersion:
        if model_version_id is not None:
            return self.get_model_version(db, int(model_version_id))

        resolved_run_id = str(run_id or "").strip()
        if not resolved_run_id:
            raise ValidationError("Missing model_version_id/run_id")

        existing = (
            db.query(ModelVersion)
            .filter(ModelVersion.run_id == resolved_run_id)
            .order_by(ModelVersion.created_at.desc(), ModelVersion.model_version_id.desc())
            .first()
        )
        if existing:
            return existing

        run = db.query(TrainingRun).filter(TrainingRun.run_id == resolved_run_id).first()
        if not run:
            raise NotFoundError("Training run not found")
        if run.status != TrainingRunStatus.COMPLETED:
            raise ConflictError("Only completed runs can be used for model execution")

        base = f"run-{resolved_run_id[:8]}"
        for index in range(1, 200):
            version = base if index == 1 else f"{base}-{index}"
            try:
                return self.register_from_run(
                    db,
                    run_id=resolved_run_id,
                    version=version,
                    stage=ModelStage.DEVELOPMENT,
                    description=description,
                )
            except ConflictError:
                continue
        raise ConflictError("Failed to auto-register model version for run")

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

        exists = (
            db.query(ModelVersion)
            .filter(ModelVersion.project_id == int(run.project_id), ModelVersion.version == version)
            .first()
        )
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
            exists = (
                db.query(ModelVersion)
                .filter(ModelVersion.project_id == int(row.project_id), ModelVersion.version == new_version)
                .first()
            )
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


__all__ = ["ModelVersionService"]
