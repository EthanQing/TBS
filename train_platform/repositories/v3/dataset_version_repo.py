from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.dataset import DatasetVersion
from train_platform.repositories.v3.base import BaseRepository


class DatasetVersionRepository(BaseRepository[DatasetVersion]):
    def __init__(self) -> None:
        super().__init__(DatasetVersion)

    def list_by_dataset(self, db: Session, dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[DatasetVersion]:
        return (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == int(dataset_id))
            .order_by(DatasetVersion.version.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_latest(self, db: Session, dataset_id: int) -> Optional[DatasetVersion]:
        return (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == int(dataset_id))
            .order_by(DatasetVersion.version.desc())
            .first()
        )

    def get_by_dataset_and_version(self, db: Session, dataset_id: int, version: int) -> Optional[DatasetVersion]:
        return (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == int(dataset_id), DatasetVersion.version == int(version))
            .first()
        )

