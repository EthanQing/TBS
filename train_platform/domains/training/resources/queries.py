from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace

from sqlalchemy.orm import Session, joinedload

from train_platform.models.v3.gpu_resource import (
    GpuDevice,
    GpuWorkerInstance,
    GpuWorkerObservation,
)
from train_platform.models.v3.gpu_allocation import GpuAllocation, GpuAllocationDevice, GpuCudaBinding, GpuNodeSchedulingState
from train_platform.domains.training.resources.accounting import ActiveCommitment, calculate_gpu_accounting


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
        node_state = db.get(GpuNodeSchedulingState, device.node_id) if device.node_id else None
        allocation_devices = (
            db.query(GpuAllocationDevice).join(GpuAllocation)
            .filter(GpuAllocationDevice.gpu_uuid == device.gpu_uuid, GpuAllocation.state != "released").all()
        )
        accounting_observation = selected or (observations[0] if observations else None)
        process_snapshot = (accounting_observation.process_snapshot or {}) if accounting_observation else {}
        accounting = calculate_gpu_accounting(
            None if accounting_observation is None else {
                "memory_total_mib": accounting_observation.memory_total_mib,
                "memory_used_mib": accounting_observation.memory_used_mib,
                "memory_free_mib": accounting_observation.memory_free_mib,
                "sampled_at": accounting_observation.sampled_at,
            },
            [ActiveCommitment(item.allocation_id, item.reserved_memory_mib) for item in allocation_devices],
            node_state.memory_safety_mib if node_state else 0,
            reliable_usage_by_allocation=process_snapshot.get("usage_by_allocation"),
            process_mapping_complete=bool(process_snapshot.get("attribution_complete")),
            process_mapping_has_duplicates=bool(process_snapshot.get("has_duplicates")),
            stale_after_seconds=stale_after_seconds,
            now=now,
        )
        if selected is None and accounting.status in {"verified", "conservative"}:
            accounting = replace(accounting, status="stale", available_budget_mib=0)
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
                "scheduling_mode": "managed" if node_state and node_state.managed else "unmanaged",
                "allocation_enabled": bool(node_state and node_state.managed and node_state.accepting_allocations),
                "shared_execution_enabled": bool(node_state and node_state.shared_execution_enabled),
                "reserved_memory_mib": accounting.reserved_mib,
                "reliable_training_used_mib": accounting.reliable_training_used_mib,
                "unattributed_used_mib": accounting.external_used_mib,
                "memory_safety_mib": accounting.safety_mib,
                "available_budget_mib": accounting.available_budget_mib,
                "exclusive_allocation_active": any(item.sharing == "exclusive" for item in allocation_devices),
                "shared_task_count": sum(item.sharing == "shared" for item in allocation_devices),
                "max_shared_tasks_per_device": node_state.max_shared_tasks_per_device if node_state else 0,
                "accounting_status": accounting.status,
                "accounting_sampled_at": getattr(accounting_observation, "sampled_at", None),
                "active_allocations": [
                    {"allocation_id": item.allocation_id, "run_id": item.allocation.run_id,
                     "state": item.allocation.state, "sharing": item.sharing,
                     "ordinal": item.ordinal, "requested_memory_mib": item.requested_memory_mib,
                     "reserved_memory_mib": item.reserved_memory_mib}
                    for item in allocation_devices
                ],
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
        node = db.get(GpuNodeSchedulingState, worker.node_id) if worker.node_id else None
        if worker.stopped_at:
            heartbeat_status = "stopped"
        elif _aware(worker.heartbeat_at) >= cutoff:
            heartbeat_status = "online"
        else:
            heartbeat_status = "offline"

        bindings = db.query(GpuCudaBinding).filter_by(instance_id=worker.instance_id).order_by(GpuCudaBinding.ordinal).all()
        active_count = db.query(GpuAllocation).filter(GpuAllocation.worker_instance_id == worker.instance_id, GpuAllocation.state != "released").count()
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
                "cuda_inventory_status": worker.cuda_inventory_status,
                "cuda_inventory_error": worker.cuda_inventory_error,
                "cuda_environment_fingerprint": worker.cuda_environment_fingerprint,
                "last_successful_cuda_inventory_at": worker.last_successful_cuda_inventory_at,
                "cuda_bindings": [
                    {"gpu_uuid": item.gpu_uuid, "ordinal": item.ordinal, "pci_bus_id": item.pci_bus_id,
                     "verified_at": item.verified_at, "environment_fingerprint": item.environment_fingerprint,
                     "present": item.present, "status": item.status, "error": item.error}
                    for item in bindings
                ],
                "running_task_count": worker.running_task_count,
                "active_allocation_count": active_count,
                "max_training_slots": worker.max_training_slots,
                "accepting_tasks": bool(worker.accepting_tasks and heartbeat_status == "online"
                                        and node and node.managed and node.accepting_allocations),
                "scheduling_mode": "managed" if node and node.managed else "legacy",
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


def get_training_run_resources(db: Session, run) -> dict:
    current = None
    if run.current_allocation_id:
        allocation = db.query(GpuAllocation).options(joinedload(GpuAllocation.devices)).filter_by(allocation_id=run.current_allocation_id).first()
        if allocation:
            current = {
                "allocation_id": allocation.allocation_id, "state": allocation.state,
                "worker_instance_id": allocation.worker_instance_id, "worker_id": allocation.worker_id,
                "node_id": allocation.node_id, "request_snapshot": allocation.request_snapshot,
                "gpu_uuids": [item.gpu_uuid for item in sorted(allocation.devices, key=lambda x: x.ordinal)],
                "reserved_at": allocation.reserved_at, "started_at": allocation.started_at,
                "released_at": allocation.released_at,
            }
    history = db.query(GpuAllocation).filter_by(run_id=run.run_id).order_by(GpuAllocation.reserved_at.desc()).all()
    request = run.resource_request
    user_request = None if request is None else {
        "selection": request.selection, "gpu_count": request.gpu_count, "gpu_uuids": request.gpu_uuids,
        "memory_mib_per_gpu": request.memory_mib_per_gpu, "sharing": request.sharing,
        "node_id": request.node_id, "created_at": request.created_at,
    }
    effective = current["request_snapshot"] if current else (run.resource_wait_details or {}).get("effective_request")
    return {
        "run_id": run.run_id, "resource_request": user_request,
        "effective_request": effective, "legacy_device_mode": request is None,
        "allocation": current,
        "allocation_history": [{"allocation_id": item.allocation_id, "state": item.state, "reserved_at": item.reserved_at, "released_at": item.released_at} for item in history],
        "reason_code": "cleanup_pending" if current and current["state"] == "releasing" else run.resource_wait_reason,
        "reason_details": run.resource_wait_details,
    }
