from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base


class GpuDevice(V3Base):
    __tablename__ = "gpu_devices"

    gpu_uuid: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    pci_bus_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    observations = relationship("GpuWorkerObservation", back_populates="device", cascade="all, delete-orphan")


class GpuWorkerInstance(V3Base):
    __tablename__ = "gpu_worker_instances"

    instance_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    process_scope: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    allowed_engines: Mapped[list] = mapped_column(JSON, nullable=False)
    nvidia_visible_devices: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cuda_visible_devices: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    inventory_status: Mapped[str] = mapped_column(String(32), nullable=False)
    inventory_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_successful_inventory_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cuda_inventory_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    cuda_inventory_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cuda_environment_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_successful_cuda_inventory_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    max_training_slots: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    running_task_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accepting_tasks: Mapped[bool] = mapped_column(nullable=False, default=True)
    launcher_identity: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    observations = relationship("GpuWorkerObservation", back_populates="worker_instance", cascade="all, delete-orphan")


class GpuWorkerObservation(V3Base):
    __tablename__ = "gpu_worker_observations"
    __table_args__ = (UniqueConstraint("instance_id", "gpu_uuid", name="uq_gpu_worker_observation_instance_gpu"),)

    observation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("gpu_worker_instances.instance_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    gpu_uuid: Mapped[str] = mapped_column(String(80), ForeignKey("gpu_devices.gpu_uuid", ondelete="CASCADE"), nullable=False, index=True)
    observed_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    memory_total_mib: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    memory_used_mib: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    memory_free_mib: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    utilization_percent: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    compute_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    mig_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    probe_source: Mapped[str] = mapped_column(String(32), nullable=False)
    probe_status: Mapped[str] = mapped_column(String(32), nullable=False)
    probe_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    present: Mapped[bool] = mapped_column(nullable=False, default=True)
    sampled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    process_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    worker_instance = relationship("GpuWorkerInstance", back_populates="observations")
    device = relationship("GpuDevice", back_populates="observations")


class TrainingRunResourceRequest(V3Base):
    __tablename__ = "training_run_resource_requests"

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), primary_key=True)
    selection: Mapped[str] = mapped_column(String(16), nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    gpu_uuids: Mapped[list] = mapped_column(JSON, nullable=False)
    memory_mib_per_gpu: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sharing: Mapped[str] = mapped_column(String(16), nullable=False)
    node_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    training_run = relationship("TrainingRun", back_populates="resource_request")
