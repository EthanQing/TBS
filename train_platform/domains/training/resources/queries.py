from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session, joinedload

from train_platform.models.v3.gpu_resource import (
    GpuDevice,
    GpuWorkerInstance,
    GpuWorkerObservation,
)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo:
        return value
    return value.replace(tzinfo=timezone.utc)


def list_gpu_resources(
    db: Session,
    *,
    node_id: str | None = None,
    gpu_uuid: str | None = None,
    stale_after_seconds: int = 20,
) -> list[dict]:
    query = db.query(GpuDevice).options(
        joinedload(GpuDevice.observations).joinedload(GpuWorkerObservation.worker_instance)
    )
    if node_id is not None:
        query = query.filter(GpuDevice.node_id == node_id)
    if gpu_uuid is not None:
        query = query.filter(GpuDevice.gpu_uuid == gpu_uuid)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=stale_after_seconds)
    resources = []
    for device in query.all():
        observations = sorted(
            device.observations,
            key=lambda observation: _aware(observation.sampled_at),
            reverse=True,
        )
        present = [
            observation
            for observation in observations
            if observation.present and observation.probe_status in {"success", "partial"}
        ]
        fresh_observations = [
            observation
            for observation in present
            if cutoff <= _aware(observation.sampled_at) <= now
            and not observation.worker_instance.stopped_at
            and cutoff <= _aware(observation.worker_instance.heartbeat_at) <= now
        ]
        valid_fresh = [
            observation
            for observation in fresh_observations
            if all(
                value is not None and value >= 0
                for value in (
                    observation.memory_total_mib,
                    observation.memory_used_mib,
                    observation.memory_free_mib,
                )
            )
        ]
        selected = valid_fresh[0] if valid_fresh else None
        worker_nodes = {
            observation.worker_instance.node_id
            for observation in observations
            if observation.worker_instance.node_id
        }
        node_conflict = len(worker_nodes) > 1 or bool(
            device.node_id and worker_nodes - {device.node_id}
        )
        if node_conflict:
            registration_status = "node_id_conflict"
        elif not device.node_id:
            registration_status = "node_id_unconfigured"
        else:
            registration_status = "registered"

        resources.append(
            {
                "gpu_uuid": device.gpu_uuid,
                "name": device.name,
                "pci_bus_id": device.pci_bus_id,
                "node_id": device.node_id,
                "memory_total_mib": getattr(selected, "memory_total_mib", None),
                "memory_used_mib": getattr(selected, "memory_used_mib", None),
                "memory_free_mib": getattr(selected, "memory_free_mib", None),
                "utilization_percent": getattr(selected, "utilization_percent", None),
                "compute_mode": getattr(selected, "compute_mode", None),
                "mig_mode": getattr(selected, "mig_mode", None),
                "probe_source": getattr(selected, "probe_source", None),
                "sampled_at": getattr(selected, "sampled_at", None),
                "source_instance_id": getattr(selected, "instance_id", None),
                "freshness_status": (
                    "fresh" if fresh_observations else "stale" if present else "registered"
                ),
                "registration_status": registration_status,
                "workers": [
                    {
                        "instance_id": observation.instance_id,
                        "worker_id": observation.worker_instance.worker_id,
                        "node_id": observation.worker_instance.node_id,
                        "node_id_unconfigured": observation.worker_instance.node_id is None,
                        "heartbeat_at": observation.worker_instance.heartbeat_at,
                        "inventory_status": observation.worker_instance.inventory_status,
                        "present": observation.present,
                        "status": (
                            "stopped"
                            if observation.worker_instance.stopped_at
                            else "online"
                            if _aware(observation.worker_instance.heartbeat_at) >= cutoff
                            else "offline"
                        ),
                    }
                    for observation in observations
                ],
            }
        )
    return resources


def list_gpu_workers(db: Session, *, stale_after_seconds: int = 20) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    workers = []
    query = (
        db.query(GpuWorkerInstance)
        .options(joinedload(GpuWorkerInstance.observations))
        .order_by(GpuWorkerInstance.started_at.desc())
    )
    for worker in query.all():
        if worker.stopped_at:
            heartbeat_status = "stopped"
        elif _aware(worker.heartbeat_at) >= cutoff:
            heartbeat_status = "online"
        else:
            heartbeat_status = "offline"

        workers.append(
            {
                "instance_id": worker.instance_id,
                "worker_id": worker.worker_id,
                "node_id": worker.node_id,
                "node_id_unconfigured": worker.node_id is None,
                "hostname": worker.hostname,
                "process_scope": worker.process_scope,
                "allowed_engines": worker.allowed_engines,
                "nvidia_visible_devices": worker.nvidia_visible_devices,
                "cuda_visible_devices": worker.cuda_visible_devices,
                "started_at": worker.started_at,
                "heartbeat_at": worker.heartbeat_at,
                "stopped_at": worker.stopped_at,
                "heartbeat_status": heartbeat_status,
                "inventory_status": worker.inventory_status,
                "inventory_error": worker.inventory_error,
                "last_successful_inventory_at": worker.last_successful_inventory_at,
                "observations": [
                    {
                        "gpu_uuid": observation.gpu_uuid,
                        "observed_index": observation.observed_index,
                        "memory_total_mib": observation.memory_total_mib,
                        "memory_used_mib": observation.memory_used_mib,
                        "memory_free_mib": observation.memory_free_mib,
                        "utilization_percent": observation.utilization_percent,
                        "compute_mode": observation.compute_mode,
                        "mig_mode": observation.mig_mode,
                        "probe_source": observation.probe_source,
                        "probe_status": observation.probe_status,
                        "probe_error": observation.probe_error,
                        "present": observation.present,
                        "sampled_at": observation.sampled_at,
                        "received_at": observation.received_at,
                    }
                    for observation in worker.observations
                ],
            }
        )
    return workers
