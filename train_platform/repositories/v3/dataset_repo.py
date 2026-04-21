from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.dataset import Dataset
from train_platform.repositories.v3.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self) -> None:
        super().__init__(Dataset)

    def get_by_name(self, db: Session, name: str) -> Optional[Dataset]:
        return db.query(Dataset).filter(Dataset.name == str(name)).first()

