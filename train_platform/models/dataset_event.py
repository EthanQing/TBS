from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base


class DatasetEvent(Base):
    """
    Dataset-level audit/event log.

    Used for upload history (e.g. append images), versioning actions, etc.
    """

    __tablename__ = "dataset_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("dataset_versions.version_id"), nullable=True, index=True)

    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True, default="event")
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    dataset = relationship("Dataset", back_populates="events")
    version = relationship("DatasetVersion")
