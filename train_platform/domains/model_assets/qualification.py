from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session

from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.qualified_model import QualifiedModel
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import ConflictError, NotFoundError


def _filtered_query(
    db: Session,
    *,
    project_id: int | None = None,
    standard_dataset_id: int | None = None,
    run_id: str | None = None,
    model_version_id: int | None = None,
) -> Query[QualifiedModel]:
    query = db.query(QualifiedModel)
    if project_id is not None:
        query = query.filter(QualifiedModel.project_id == int(project_id))
    if standard_dataset_id is not None:
        query = query.filter(QualifiedModel.standard_dataset_id == int(standard_dataset_id))
    if run_id:
        query = query.filter(QualifiedModel.run_id == str(run_id))
    if model_version_id is not None:
        query = query.filter(QualifiedModel.model_version_id == int(model_version_id))
    return query


def list_qualified_models(
    db: Session,
    *,
    project_id: int | None = None,
    standard_dataset_id: int | None = None,
    run_id: str | None = None,
    model_version_id: int | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[QualifiedModel], int]:
    query = _filtered_query(
        db,
        project_id=project_id,
        standard_dataset_id=standard_dataset_id,
        run_id=run_id,
        model_version_id=model_version_id,
    )
    total = query.count()
    items = (
        query.order_by(QualifiedModel.created_at.desc(), QualifiedModel.qualified_model_id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return items, int(total)


def get_qualified_model(db: Session, qualified_model_id: int) -> QualifiedModel:
    row = db.query(QualifiedModel).filter(QualifiedModel.qualified_model_id == int(qualified_model_id)).first()
    if not row:
        raise NotFoundError("合格模型记录不存在")
    return row


def mark_model_qualified(
    db: Session,
    *,
    model_version_id: int,
    qualified_by: str | None = None,
    note: str | None = None,
) -> tuple[QualifiedModel, bool]:
    model_version_id = int(model_version_id)

    existing = (
        db.query(QualifiedModel)
        .filter(QualifiedModel.model_version_id == model_version_id)
        .first()
    )
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
        raise ConflictError(f"训练任务状态为 '{status_value}'，仅 completed 状态可标记为合格模型")

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
        qualified_by=_normalize_optional_text(qualified_by),
        note=_normalize_optional_text(note),
        metrics=metrics,
        weights_path=weights_path,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(QualifiedModel)
            .filter(QualifiedModel.model_version_id == model_version_id)
            .first()
        )
        if existing:
            return existing, False
        raise
    db.refresh(row)
    return row, True


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["get_qualified_model", "list_qualified_models", "mark_model_qualified"]
