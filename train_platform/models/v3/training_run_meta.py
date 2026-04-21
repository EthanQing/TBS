from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base as Base


class TrainingRunMeta(Base):
    """
    Optional user-facing metadata for a training run.

    Stored in a 1:1 table to keep `training_runs` hot-path fields lean.
    """

    __tablename__ = "training_run_meta"

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), primary_key=True)

    creator: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    group_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    tags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    training_run = relationship("TrainingRun", back_populates="meta")

