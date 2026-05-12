from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.illegal_dataset import IllegalDatasetVersion
from train_platform.repositories.v3.base import BaseRepository


class IllegalDatasetVersionRepository(BaseRepository[IllegalDatasetVersion]):
    def __init__(self) -> None:
        super().__init__(IllegalDatasetVersion)

    def list_by_dataset(self, db: Session, illegal_dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[IllegalDatasetVersion]:
        return (
            db.query(IllegalDatasetVersion)
            .filter(IllegalDatasetVersion.illegal_dataset_id == int(illegal_dataset_id))
            .order_by(IllegalDatasetVersion.version.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_latest(self, db: Session, illegal_dataset_id: int) -> Optional[IllegalDatasetVersion]:
        return (
            db.query(IllegalDatasetVersion)
            .filter(IllegalDatasetVersion.illegal_dataset_id == int(illegal_dataset_id))
            .order_by(IllegalDatasetVersion.version.desc())
            .first()
        )
