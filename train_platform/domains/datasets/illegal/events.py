from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from train_platform.models.v3.illegal_dataset import IllegalDatasetEvent


def add_event(
    db: Session,
    dataset_id: int,
    event_type: str,
    *,
    version_id: int | None = None,
    message: str | None = None,
    created_by: str | None = None,
    data: dict[str, Any] | None = None,
) -> IllegalDatasetEvent:
    event = IllegalDatasetEvent(
        illegal_dataset_id=int(dataset_id),
        version_id=int(version_id) if version_id is not None else None,
        event_type=str(event_type),
        message=message,
        created_by=created_by,
        data=data,
    )
    db.add(event)
    db.flush()
    return event
