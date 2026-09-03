from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.models.v3.training_run import (
    TrainingRunArtifact,
    TrainingRunEpochMetric,
    TrainingRunEvent,
)

from .service import TrainingRunService


def list_events(db: Session, run_id: str, *, limit: int = 200) -> list[TrainingRunEvent]:
    TrainingRunService().get_run(db, run_id)
    return (
        db.query(TrainingRunEvent)
        .filter(TrainingRunEvent.run_id == str(run_id))
        .order_by(TrainingRunEvent.created_at.desc())
        .limit(int(limit))
        .all()
    )


def list_epoch_metrics(db: Session, run_id: str, *, limit: int = 5000) -> list[TrainingRunEpochMetric]:
    TrainingRunService().get_run(db, run_id)
    return (
        db.query(TrainingRunEpochMetric)
        .filter(TrainingRunEpochMetric.run_id == str(run_id))
        .order_by(TrainingRunEpochMetric.epoch.asc())
        .limit(int(limit))
        .all()
    )


def list_artifacts(db: Session, run_id: str) -> list[TrainingRunArtifact]:
    TrainingRunService().get_run(db, run_id)
    return (
        db.query(TrainingRunArtifact)
        .filter(TrainingRunArtifact.run_id == str(run_id))
        .order_by(TrainingRunArtifact.created_at.desc())
        .all()
    )


__all__ = ["list_artifacts", "list_epoch_metrics", "list_events"]
