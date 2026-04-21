from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session, joinedload

from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun
from train_platform.repositories.v3.base import BaseRepository


class TrainingRunRepository(BaseRepository[TrainingRun]):
    def __init__(self) -> None:
        super().__init__(TrainingRun)

    def get(self, db: Session, run_id: str) -> Optional[TrainingRun]:
        return (
            db.query(TrainingRun)
            .options(
                joinedload(TrainingRun.parameters),
                joinedload(TrainingRun.result),
                joinedload(TrainingRun.meta),
                joinedload(TrainingRun.project),
                joinedload(TrainingRun.standard_dataset),
                joinedload(TrainingRun.architecture),
            )
            .filter(TrainingRun.run_id == str(run_id))
            .first()
        )

    def list(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        status: Optional[TrainingRunStatus] = None,
        standard_dataset_id: Optional[int] = None,
        architecture_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
        include_hidden: bool = False,
    ) -> list[TrainingRun]:
        q = (
            db.query(TrainingRun)
            .options(joinedload(TrainingRun.parameters))
            .options(joinedload(TrainingRun.result))
            .options(joinedload(TrainingRun.meta))
            .options(joinedload(TrainingRun.project))
            .options(joinedload(TrainingRun.standard_dataset))
            .options(joinedload(TrainingRun.architecture))
        )
        if not include_hidden:
            q = q.filter(TrainingRun.hidden == False)  # noqa: E712
        if project_id is not None:
            q = q.filter(TrainingRun.project_id == int(project_id))
        if architecture_id is not None:
            q = q.filter(TrainingRun.architecture_id == int(architecture_id))
        if status is not None:
            q = q.filter(TrainingRun.status == status)
        if standard_dataset_id is not None:
            q = q.filter(TrainingRun.standard_dataset_id == int(standard_dataset_id))
        return q.order_by(TrainingRun.created_at.desc()).offset(skip).limit(limit).all()
