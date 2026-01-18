from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BIGINT, DateTime, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base
from train_platform.models.enums import DatasetType, DatasetVersionStatus


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_type: Mapped[DatasetType] = mapped_column(
        Enum(DatasetType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    # Store a portable token / relative path under BASE_DATASETS_DIR.
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    active_version_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("dataset_versions.version_id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan", foreign_keys="DatasetVersion.dataset_id")
    active_version = relationship("DatasetVersion", foreign_keys=[active_version_id], post_update=True)

    projects = relationship("Project", back_populates="dataset")
    events = relationship("DatasetEvent", back_populates="dataset", cascade="all, delete-orphan")
    images = relationship("DatasetImage", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True)

    # Monotonic integer version per dataset (v1, v2, ...).
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("dataset_versions.version_id"), nullable=True)

    status: Mapped[DatasetVersionStatus] = mapped_column(
        Enum(DatasetVersionStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DatasetVersionStatus.CREATED,
        index=True,
    )

    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Paths relative to BASE_DATASETS_DIR for portability.
    manifest_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    file_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(BIGINT, nullable=True)

    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    dataset = relationship("Dataset", back_populates="versions", foreign_keys=[dataset_id])
    parent = relationship("DatasetVersion", remote_side=[version_id], foreign_keys=[parent_version_id])

    training_runs = relationship("TrainingRun", back_populates="dataset_version")
    images = relationship("DatasetImage", back_populates="version", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_versions_dataset_version"),)
