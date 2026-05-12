from __future__ import annotations

from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.qualified_model import QualifiedModel
from train_platform.models.v3.training_run import TrainingRun
from train_platform.repositories.v3.qualified_model_repo import QualifiedModelRepository
from train_platform.utils.exceptions import ConflictError, NotFoundError


class QualifiedModelService:
    def __init__(self) -> None:
        self.repo = QualifiedModelRepository()

    def list_qualified_models(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        standard_dataset_id: Optional[int] = None,
        run_id: Optional[str] = None,
        model_version_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[QualifiedModel]:
        return self.repo.list(
            db,
            project_id=project_id,
            standard_dataset_id=standard_dataset_id,
            run_id=run_id,
            model_version_id=model_version_id,
            skip=skip,
            limit=limit,
        )

    def get_qualified_model(self, db: Session, qualified_model_id: int) -> QualifiedModel:
        row = self.repo.get(db, int(qualified_model_id))
        if not row:
            raise NotFoundError("合格模型记录不存在")
        return row

    def mark_model_qualified(
        self,
        db: Session,
        *,
        model_version_id: int,
        qualified_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> tuple[QualifiedModel, bool]:
        model_version_id = int(model_version_id)

        existing = self.repo.get_by_model_version_id(db, model_version_id)
        if existing:
            return existing, False

        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == model_version_id)
            .first()
        )
        if not model_version:
            raise NotFoundError("模型版本不存在")

        run = db.query(TrainingRun).filter(TrainingRun.run_id == str(model_version.run_id)).first()
        if not run:
            raise NotFoundError("训练任务不存在")

        if run.status == TrainingRunStatus.FAILED:
            raise ConflictError("训练任务已失败，不能标记为合格模型")
        if run.status != TrainingRunStatus.COMPLETED:
            status_value = getattr(run.status, "value", str(run.status))
            raise ConflictError(
                f"训练任务状态为 '{status_value}'，仅 completed 状态可标记为合格模型"
            )

        metrics = model_version.metrics
        weights_path = model_version.weights_path
        if run.result is not None:
            metrics = metrics or run.result.best_metrics or run.result.final_metrics
            weights_path = weights_path or run.result.best_weights_path or run.result.last_weights_path

        row = QualifiedModel(
            model_version_id=model_version_id,
            project_id=int(model_version.project_id),
            run_id=str(model_version.run_id),
            standard_dataset_id=int(run.standard_dataset_id),
            qualified_by=self._normalize_optional_text(qualified_by),
            note=self._normalize_optional_text(note),
            metrics=metrics,
            weights_path=weights_path,
        )
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = self.repo.get_by_model_version_id(db, model_version_id)
            if existing:
                return existing, False
            raise
        db.refresh(row)
        return row, True

    @staticmethod
    def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
