from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BIGINT, DateTime, Enum, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base
from train_platform.models.v3.enums import DatasetSplit, DatasetType, DatasetVersionStatus


class IllegalDataset(V3Base):
    __tablename__ = "illegal_datasets"

    illegal_dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_type: Mapped[DatasetType] = mapped_column(
        Enum(DatasetType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    format: Mapped[str] = mapped_column(String(50), nullable=False, server_default="yolo")
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active_version_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("illegal_dataset_versions.version_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
    )

    versions = relationship(
        "IllegalDatasetVersion",
        back_populates="illegal_dataset",
        cascade="all, delete-orphan",
        foreign_keys="IllegalDatasetVersion.illegal_dataset_id",
    )
    active_version = relationship("IllegalDatasetVersion", foreign_keys=[active_version_id], post_update=True)
    events = relationship("IllegalDatasetEvent", back_populates="illegal_dataset", cascade="all, delete-orphan")
    images = relationship("IllegalDatasetImage", back_populates="illegal_dataset", cascade="all, delete-orphan")
    label_mappings = relationship(
        "IllegalDatasetLabelMapping",
        back_populates="illegal_dataset",
        cascade="all, delete-orphan",
    )


class IllegalDatasetVersion(V3Base):
    __tablename__ = "illegal_dataset_versions"

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    illegal_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_datasets.illegal_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("illegal_dataset_versions.version_id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[DatasetVersionStatus] = mapped_column(
        Enum(DatasetVersionStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DatasetVersionStatus.CREATED,
        index=True,
    )
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    manifest_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(BIGINT, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    illegal_dataset = relationship("IllegalDataset", back_populates="versions", foreign_keys=[illegal_dataset_id])
    parent = relationship("IllegalDatasetVersion", remote_side=[version_id], foreign_keys=[parent_version_id])
    images = relationship("IllegalDatasetImage", back_populates="version", cascade="all, delete-orphan")
    events = relationship("IllegalDatasetEvent", back_populates="version")

    __table_args__ = (
        UniqueConstraint("illegal_dataset_id", "version", name="uq_illegal_dataset_versions_dataset_version"),
    )


class IllegalDatasetEvent(V3Base):
    __tablename__ = "illegal_dataset_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    illegal_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_datasets.illegal_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("illegal_dataset_versions.version_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    illegal_dataset = relationship("IllegalDataset", back_populates="events")
    version = relationship("IllegalDatasetVersion", back_populates="events")


class IllegalDatasetImage(V3Base):
    __tablename__ = "illegal_dataset_images"

    image_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    illegal_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_datasets.illegal_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_dataset_versions.version_id", ondelete="CASCADE"),
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

    illegal_dataset = relationship("IllegalDataset", back_populates="images")
    version = relationship("IllegalDatasetVersion", back_populates="images")

    __table_args__ = (UniqueConstraint("version_id", "path", name="uq_illegal_dataset_images_version_path"),)


class IllegalDatasetLabelMapping(V3Base):
    __tablename__ = "illegal_dataset_label_mappings"

    mapping_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    illegal_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_datasets.illegal_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    raw_label: Mapped[str] = mapped_column(String(255), nullable=False)
    mapped_label: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="keep", server_default="keep")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    illegal_dataset = relationship("IllegalDataset", back_populates="label_mappings")

    __table_args__ = (
        UniqueConstraint("illegal_dataset_id", "raw_label", name="uq_illegal_dataset_label_mapping_raw_label"),
    )


class IllegalDatasetPublishJob(V3Base):
    __tablename__ = "illegal_dataset_publish_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    illegal_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_datasets.illegal_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_illegal_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("illegal_dataset_versions.version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    standard_dataset_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("standard_datasets.standard_dataset_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, server_default="queued")
    phase: Mapped[str] = mapped_column(String(32), nullable=False, server_default="queued")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    seq: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    request_payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    request_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    logs: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    illegal_dataset = relationship("IllegalDataset")
    source_version = relationship("IllegalDatasetVersion", foreign_keys=[source_illegal_version_id])
    standard_dataset = relationship("StandardDataset")

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_illegal_dataset_publish_jobs_idempotency_key"),
        Index("ix_illegal_publish_jobs_dataset_status", "illegal_dataset_id", "status"),
    )
