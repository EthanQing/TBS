from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from train_platform.models.v3.standard_dataset import StandardDatasetEvent


def _events_query(db: Session, dataset_id: int):
    return (
        db.query(StandardDatasetEvent)
        .filter(StandardDatasetEvent.standard_dataset_id == int(dataset_id))
        .order_by(StandardDatasetEvent.created_at.desc(), StandardDatasetEvent.event_id.desc())
    )


def add_event(
    db: Session,
    dataset_id: int,
    event_type: str,
    *,
    message: str | None = None,
    created_by: str | None = None,
    data: dict[str, Any] | None = None,
) -> StandardDatasetEvent:
    event = StandardDatasetEvent(
        standard_dataset_id=int(dataset_id),
        event_type=str(event_type),
        message=message,
        created_by=created_by,
        data=data,
    )
    db.add(event)
    db.flush()
    return event


def list_events(
    db: Session,
    dataset_id: int,
    *,
    skip: int = 0,
    limit: int = 100,
) -> list[StandardDatasetEvent]:
    return (
        _events_query(db, dataset_id)
        .offset(max(0, int(skip)))
        .limit(max(0, int(limit)))
        .all()
    )


def list_events_page(
    db: Session,
    dataset_id: int,
    *,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[StandardDatasetEvent], int]:
    query = _events_query(db, dataset_id)
    total = int(query.count())
    items = (
        query
        .offset(max(0, int(skip)))
        .limit(max(0, int(limit)))
        .all()
    )
    return items, total


__all__ = ["add_event", "list_events", "list_events_page"]
