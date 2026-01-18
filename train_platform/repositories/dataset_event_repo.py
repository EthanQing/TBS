from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.dataset_event import DatasetEvent
from train_platform.repositories.base import BaseRepository


class DatasetEventRepository(BaseRepository[DatasetEvent]):
    def __init__(self) -> None:
        super().__init__(DatasetEvent)

    def list_by_dataset(
        self,
        db: Session,
        dataset_id: int,
        *,
        event_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DatasetEvent]:
        q = db.query(DatasetEvent).filter(DatasetEvent.dataset_id == int(dataset_id))
        if event_type:
            q = q.filter(DatasetEvent.event_type == str(event_type))
        return q.order_by(DatasetEvent.created_at.desc()).offset(int(skip)).limit(int(limit)).all()

