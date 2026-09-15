from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field
from train_platform.schemas.v3.training_runs import TrainingRunResourceRequestOut

class GpuResourceWorkerOut(BaseModel):
    instance_id: str
    worker_id: str
    node_id: str | None = None
    node_id_unconfigured: bool
    heartbeat_at: datetime
    inventory_status: str
    present: bool
    status: Literal["online", "offline", "stopped"]

class GpuResourceOut(BaseModel):
    gpu_uuid: str
    name: str
    pci_bus_id: str | None = None
    node_id: str | None = None
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    memory_free_mib: int | None = Field(
        None,
        description="Driver-sampled free memory in MiB; this is not schedulable capacity.",
    )
    utilization_percent: int | None = None
    compute_mode: str | None = None
    mig_mode: str | None = None
    probe_source: str | None = None
    sampled_at: datetime | None = None
    source_instance_id: str | None = None
    freshness_status: str
    registration_status: str
    workers: list[GpuResourceWorkerOut] = Field(default_factory=list)

class GpuWorkerObservationOut(BaseModel):
    gpu_uuid: str
    observed_index: int | None = Field(
        None,
        description="Probe-reported index; it is not a verified local CUDA device index.",
    )
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    memory_free_mib: int | None = Field(
        None,
        description="Driver-sampled free memory in MiB; this is not schedulable capacity.",
    )
    utilization_percent: int | None = None
    compute_mode: str | None = None
    mig_mode: str | None = None
    probe_source: str
    probe_status: str
    probe_error: str | None = None
    present: bool
    sampled_at: datetime
    received_at: datetime

class GpuWorkerOut(BaseModel):
    instance_id: str
    worker_id: str
    node_id: str | None = None
    node_id_unconfigured: bool
    hostname: str
    process_scope: dict[str, Any] | None = None
    allowed_engines: list[str]
    nvidia_visible_devices: str | None = None
    cuda_visible_devices: str | None = None
    started_at: datetime
    heartbeat_at: datetime
    stopped_at: datetime | None = None
    heartbeat_status: str
    inventory_status: str
    inventory_error: str | None = None
    last_successful_inventory_at: datetime | None = None
    observations: list[GpuWorkerObservationOut] = Field(default_factory=list)

class GpuResourcesResponse(BaseModel):
    scheduler_stage: Literal["inventory_only"] = "inventory_only"
    allocation_enabled: Literal[False] = False
    items: list[GpuResourceOut] = Field(default_factory=list)

class GpuWorkersResponse(BaseModel):
    scheduler_stage: Literal["inventory_only"] = "inventory_only"
    allocation_enabled: Literal[False] = False
    items: list[GpuWorkerOut] = Field(default_factory=list)

class TrainingRunResourcesResponse(BaseModel):
    run_id: str
    scheduler_stage: Literal["inventory_only"] = "inventory_only"
    allocation_enabled: Literal[False] = False
    resource_request: TrainingRunResourceRequestOut | None = None
    legacy_device_mode: bool
    allocation: None = None
    reason_code: Literal["legacy_device_mode", "resource_scheduler_not_enabled"]
