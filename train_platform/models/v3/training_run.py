from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BIGINT, Boolean, DateTime, DECIMAL, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base as Base
from train_platform.models.v3.enums import LogLevel, TrainingRunStatus


class TrainingRun(Base):
    __tablename__ = "training_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    standard_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("standard_datasets.standard_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    architecture_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model_architectures.architecture_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[TrainingRunStatus] = mapped_column(
        Enum(TrainingRunStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TrainingRunStatus.CREATED,
        index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_epochs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    cancel_requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    cancel_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    delete_requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    hidden: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    run_dir: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    project = relationship("Project", back_populates="training_runs")
    standard_dataset = relationship("StandardDataset", back_populates="training_runs")
    architecture = relationship("ModelArchitecture", back_populates="training_runs")
    parameters = relationship("TrainingRunParameters", back_populates="training_run", uselist=False, cascade="all, delete-orphan")
    result = relationship("TrainingRunResult", back_populates="training_run", uselist=False, cascade="all, delete-orphan")
    epoch_metrics = relationship("TrainingRunEpochMetric", back_populates="training_run", cascade="all, delete-orphan")
    artifacts = relationship("TrainingRunArtifact", back_populates="training_run", cascade="all, delete-orphan")
    events = relationship("TrainingRunEvent", back_populates="training_run", cascade="all, delete-orphan")
    model_versions = relationship("ModelVersion", back_populates="training_run")
    meta = relationship("TrainingRunMeta", back_populates="training_run", uselist=False, cascade="all, delete-orphan")


class TrainingRunParameters(Base):
    __tablename__ = "training_run_parameters"

    param_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), nullable=False, unique=True)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=16)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    learning_rate: Mapped[Optional[float]] = mapped_column(DECIMAL(10, 8), nullable=True)
    patience: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    device: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    workers: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    use_pretrained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    optimizer: Mapped[str] = mapped_column(String(64), nullable=False, default="AdamW")
    augmentation: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    additional_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    training_run = relationship("TrainingRun", back_populates="parameters")


class TrainingRunResult(Base):
    __tablename__ = "training_run_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), nullable=False, unique=True)
    best_weights_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    last_weights_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    results_dir: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    final_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    best_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_size_mb: Mapped[Optional[float]] = mapped_column(DECIMAL(10, 2), nullable=True)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(DECIMAL(10, 4), nullable=True)
    flops: Mapped[Optional[int]] = mapped_column(BIGINT, nullable=True)

    training_run = relationship("TrainingRun", back_populates="result")


class TrainingRunEvent(Base):
    __tablename__ = "training_run_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    level: Mapped[LogLevel] = mapped_column(
        Enum(LogLevel, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=LogLevel.INFO,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True, default="event")
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    training_run = relationship("TrainingRun", back_populates="events")


class TrainingRunEpochMetric(Base):
    __tablename__ = "training_run_epoch_metrics"

    metric_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (UniqueConstraint("run_id", "epoch", name="uq_training_run_epoch_metrics_run_epoch"),)

    training_run = relationship("TrainingRun", back_populates="epoch_metrics")


class TrainingRunArtifact(Base):
    __tablename__ = "training_run_artifacts"

    artifact_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column(BIGINT, nullable=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    training_run = relationship("TrainingRun", back_populates="artifacts")
