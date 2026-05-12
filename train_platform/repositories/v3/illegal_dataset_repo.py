from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.illegal_dataset import IllegalDataset
from train_platform.repositories.v3.base import BaseRepository


class IllegalDatasetRepository(BaseRepository[IllegalDataset]):
    def __init__(self) -> None:
        super().__init__(IllegalDataset)

    def get_by_name(self, db: Session, name: str) -> Optional[IllegalDataset]:
        return db.query(IllegalDataset).filter(IllegalDataset.name == str(name)).first()
