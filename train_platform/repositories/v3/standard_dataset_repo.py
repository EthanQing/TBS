from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.repositories.v3.base import BaseRepository


class StandardDatasetRepository(BaseRepository[StandardDataset]):
    def __init__(self) -> None:
        super().__init__(StandardDataset)

    def get_by_name(self, db: Session, name: str) -> Optional[StandardDataset]:
        return db.query(StandardDataset).filter(StandardDataset.name == str(name)).first()
