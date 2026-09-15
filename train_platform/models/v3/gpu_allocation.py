from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base


class GpuNodeSchedulingState(V3Base):
    __tablename__ = "gpu_node_scheduling_states"

    node_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    managed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accepting_allocations: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    shared_execution_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    max_shared_tasks_per_device: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    memory_safety_mib: Mapped[int] = mapped_column(Integer, nullable=False, default=4096)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class GpuAllocation(V3Base):
    __tablename__ = "gpu_allocations"

    allocation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="RESTRICT"), nullable=False, index=True)
    active_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True)
    worker_instance_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("gpu_worker_instances.instance_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    authorization_state: Mapped[str] = mapped_column(String(16), nullable=False, default="issued")
    request_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    execution_owner: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    launcher_identity: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    execution_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reserved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    launch_deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    released_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    devices = relationship("GpuAllocationDevice", back_populates="allocation", cascade="all, delete-orphan")


class GpuAllocationDevice(V3Base):
    __tablename__ = "gpu_allocation_devices"
    __table_args__ = (
        UniqueConstraint("allocation_id", "gpu_uuid", name="uq_gpu_allocation_device_gpu"),
        UniqueConstraint("allocation_id", "ordinal", name="uq_gpu_allocation_device_ordinal"),
    )

    allocation_device_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    allocation_id: Mapped[str] = mapped_column(String(36), ForeignKey("gpu_allocations.allocation_id", ondelete="RESTRICT"), nullable=False, index=True)
    gpu_uuid: Mapped[str] = mapped_column(String(80), ForeignKey("gpu_devices.gpu_uuid", ondelete="RESTRICT"), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    sharing: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_memory_mib: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reserved_memory_mib: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    allocation = relationship("GpuAllocation", back_populates="devices")


class GpuCudaBinding(V3Base):
    __tablename__ = "gpu_cuda_bindings"
    __table_args__ = (UniqueConstraint("instance_id", "gpu_uuid", name="uq_gpu_cuda_binding_instance_gpu"),)

    binding_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[str] = mapped_column(String(36), ForeignKey("gpu_worker_instances.instance_id", ondelete="RESTRICT"), nullable=False, index=True)
    gpu_uuid: Mapped[str] = mapped_column(String(80), ForeignKey("gpu_devices.gpu_uuid", ondelete="RESTRICT"), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    pci_bus_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    environment_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
