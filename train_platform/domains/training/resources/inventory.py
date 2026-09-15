from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from train_platform.models.v3.gpu_resource import GpuDevice, GpuWorkerInstance, GpuWorkerObservation
from train_platform.models.v3.gpu_allocation import GpuCudaBinding, GpuNodeSchedulingState
from train_platform.platform.runtime.cuda_devices import CudaBindingResult
from train_platform.platform.runtime.gpu_probe import GpuProbeResult


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def register_worker_instance(
    db: Session,
    *,
    instance_id: str,
    worker_id: str,
    node_id: str | None,
    hostname: str,
    process_scope: dict | None,
    allowed_engines: list[str],
    nvidia_visible_devices: str | None,
    cuda_visible_devices: str | None,
    started_at: datetime | None = None,
    max_training_slots: int = 2,
    accepting_tasks: bool = True,
    launcher_identity: dict | None = None,
    scheduling_policy: dict | None = None,
) -> GpuWorkerInstance:
    if node_id and scheduling_policy is not None:
        node = db.query(GpuNodeSchedulingState).filter_by(node_id=node_id).with_for_update().first()
        if node is None:
            db.add(GpuNodeSchedulingState(node_id=node_id, managed=True, accepting_allocations=True,
                                          **scheduling_policy))
            db.flush()
        elif not node.managed:
            node.managed = True
            node.accepting_allocations = True
    existing = db.get(GpuWorkerInstance, instance_id)
    if existing is not None:
        return existing
    now = started_at or utcnow()
    item = GpuWorkerInstance(
        instance_id=instance_id,
        worker_id=worker_id,
        node_id=node_id,
        hostname=hostname,
        process_scope=process_scope,
        allowed_engines=allowed_engines,
        nvidia_visible_devices=nvidia_visible_devices,
        cuda_visible_devices=cuda_visible_devices,
        started_at=now,
        heartbeat_at=now,
        inventory_status="pending",
        cuda_inventory_status="pending",
        max_training_slots=max_training_slots,
        accepting_tasks=accepting_tasks,
        launcher_identity=launcher_identity,
    )
    db.add(item)
    db.flush()
    return item


def update_worker_heartbeat(db: Session, instance_id: str, *, at: datetime | None = None) -> None:
    item = db.get(GpuWorkerInstance, instance_id)
    if item:
        item.heartbeat_at = at or utcnow()


def save_inventory(db: Session, instance_id: str, result: GpuProbeResult) -> None:
    worker = db.get(GpuWorkerInstance, instance_id)
    if not worker:
        raise ValueError(f"GPU worker instance not found: {instance_id}")
    received = utcnow()
    worker.heartbeat_at = received
    worker.inventory_status = result.status
    worker.inventory_error = result.error
    if result.status not in {"success", "empty"}:
        return
    worker.last_successful_inventory_at = result.sampled_at
    device_errors = [device.error for device in result.devices if not device.gpu_uuid and device.error]
    if device_errors:
        worker.inventory_error = "; ".join(
            error for error in [result.error, *device_errors] if error
        )
    seen: set[str] = set()
    for sampled in result.devices:
        if not sampled.gpu_uuid:
            continue
        seen.add(sampled.gpu_uuid)
        device = db.query(GpuDevice).filter(GpuDevice.gpu_uuid == sampled.gpu_uuid).with_for_update().first()
        if device is None:
            device = GpuDevice(
                gpu_uuid=sampled.gpu_uuid,
                name=sampled.name or "GPU",
                pci_bus_id=sampled.pci_bus_id,
                node_id=worker.node_id,
            )
            try:
                with db.begin_nested():
                    db.add(device)
                    db.flush()
            except IntegrityError:
                device = db.query(GpuDevice).filter_by(gpu_uuid=sampled.gpu_uuid).with_for_update().one()
        conflict = bool(device.node_id and worker.node_id and device.node_id != worker.node_id)
        if not device.node_id and worker.node_id:
            device.node_id = worker.node_id
        observation = db.query(GpuWorkerObservation).filter_by(instance_id=instance_id, gpu_uuid=sampled.gpu_uuid).first()
        if observation is None:
            observation = GpuWorkerObservation(
                instance_id=instance_id,
                gpu_uuid=sampled.gpu_uuid,
            )
            db.add(observation)
        observation.observed_index = sampled.observed_index
        observation.memory_total_mib = sampled.memory_total_mib
        observation.memory_used_mib = sampled.memory_used_mib
        observation.memory_free_mib = sampled.memory_free_mib
        observation.utilization_percent = sampled.utilization_percent
        observation.compute_mode = sampled.compute_mode
        observation.mig_mode = sampled.mig_mode
        observation.probe_source = result.source or "unknown"
        observation.probe_status = sampled.status
        prefix = "node_id conflict; " if conflict else ""
        observation.probe_error = prefix + (sampled.error or "") or None
        observation.present = True
        observation.sampled_at = result.sampled_at
        observation.received_at = received
        observation.process_snapshot = sampled.process_snapshot
    if result.complete:
        missing = db.query(GpuWorkerObservation).filter(
            GpuWorkerObservation.instance_id == instance_id
        )
        if seen:
            missing = missing.filter(GpuWorkerObservation.gpu_uuid.notin_(seen))
        missing.update(
            {GpuWorkerObservation.present: False},
            synchronize_session=False,
        )


def save_cuda_bindings(db: Session, instance_id: str, result: CudaBindingResult) -> None:
    worker = db.query(GpuWorkerInstance).filter_by(instance_id=instance_id).with_for_update().one()
    worker.cuda_inventory_status = result.status
    worker.cuda_inventory_error = result.error
    worker.cuda_environment_fingerprint = result.environment_fingerprint
    if result.status not in {"success", "empty"}:
        return
    worker.last_successful_cuda_inventory_at = result.sampled_at
    seen: set[str] = set()
    for sampled in result.bindings:
        if not sampled.gpu_uuid or sampled.status not in {"success", "partial"}:
            continue
        device = db.get(GpuDevice, sampled.gpu_uuid)
        if device is None:
            continue
        seen.add(sampled.gpu_uuid)
        binding = db.query(GpuCudaBinding).filter_by(instance_id=instance_id, gpu_uuid=sampled.gpu_uuid).first()
        if binding is None:
            binding = GpuCudaBinding(instance_id=instance_id, gpu_uuid=sampled.gpu_uuid)
            db.add(binding)
        binding.ordinal = sampled.ordinal
        binding.pci_bus_id = sampled.pci_bus_id
        binding.verified_at = result.sampled_at
        binding.environment_fingerprint = result.environment_fingerprint
        binding.present = True
        binding.status = sampled.status
        binding.error = sampled.error
    if result.complete:
        query = db.query(GpuCudaBinding).filter(GpuCudaBinding.instance_id == instance_id)
        if seen:
            query = query.filter(GpuCudaBinding.gpu_uuid.notin_(seen))
        query.update({GpuCudaBinding.present: False}, synchronize_session=False)


def mark_worker_stopped(db: Session, instance_id: str, *, at: datetime | None = None) -> None:
    item = db.get(GpuWorkerInstance, instance_id)
    if item:
        item.stopped_at = at or utcnow()
        item.heartbeat_at = item.stopped_at
