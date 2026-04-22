from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base
from train_platform.models.v3.enums import DatasetSplit, DatasetType


class StandardDataset(V3Base):
    __tablename__ = "standard_datasets"

    standard_dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_type: Mapped[DatasetType] = mapped_column(
        Enum(DatasetType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    format: Mapped[str] = mapped_column(String(50), nullable=False, server_default="yolo")
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    publish_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    events = relationship("StandardDatasetEvent", back_populates="standard_dataset", cascade="all, delete-orphan")
    images = relationship("StandardDatasetImage", back_populates="standard_dataset", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="standard_dataset")
    training_runs = relationship("TrainingRun", back_populates="standard_dataset")


class StandardDatasetEvent(V3Base):
    __tablename__ = "standard_dataset_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    standard_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("standard_datasets.standard_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    standard_dataset = relationship("StandardDataset", back_populates="events")


class StandardDatasetImage(V3Base):
    __tablename__ = "standard_dataset_images"

    image_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    standard_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("standard_datasets.standard_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    split: Mapped[Optional[DatasetSplit]] = mapped_column(
        Enum(DatasetSplit, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    standard_dataset = relationship("StandardDataset", back_populates="images")

    __table_args__ = (
        UniqueConstraint("standard_dataset_id", "path", name="uq_standard_dataset_images_dataset_path"),
    )
