from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.training_run_meta import TrainingRunMeta
from train_platform.repositories.base import BaseRepository


class TrainingRunMetaRepository(BaseRepository[TrainingRunMeta]):
    def __init__(self) -> None:
        super().__init__(TrainingRunMeta)

    def get_by_run_id(self, db: Session, run_id: str) -> Optional[TrainingRunMeta]:
        return db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == str(run_id)).first()

