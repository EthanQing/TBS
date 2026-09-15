from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, exists, func, or_
from sqlalchemy.orm import Session, joinedload

from train_platform.models.v3.enums import LogLevel, TrainingRunStatus
from train_platform.models.v3.gpu_allocation import (
    GpuAllocation, GpuAllocationDevice, GpuCudaBinding, GpuNodeSchedulingState,
)
from train_platform.models.v3.gpu_resource import GpuDevice, GpuWorkerInstance, GpuWorkerObservation
from train_platform.models.v3.training_run import TrainingRun, TrainingRunEvent, TrainingRunParameters
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.gpu_resource import TrainingRunResourceRequest
from train_platform.domains.training.resources.accounting import ActiveCommitment, calculate_gpu_accounting
from train_platform.domains.training.parameters import parse_visible_host_gpu_ids


@dataclass(frozen=True)
class AllocationDecision:
    allocation: GpuAllocation | None
    reason_code: str | None
    details: dict[str, Any]
    effective_request: dict[str, Any] | None


@dataclass
class AllocationScanCursor:
    queued_at: datetime | None = None
    created_at: datetime | None = None
    run_id: str | None = None

    def reset(self) -> None:
        self.queued_at = self.created_at = self.run_id = None


def _now(value: datetime | None) -> datetime:
    return value or datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _set_wait_reason(db: Session, run: TrainingRun, reason: str, details: dict[str, Any]) -> None:
    effective = (run.resource_wait_details or {}).get("effective_request")
    if effective is not None and "effective_request" not in details:
        details = {**details, "effective_request": effective}
    if run.resource_wait_reason == reason and (run.resource_wait_details or {}) == details:
        return
    run.resource_wait_reason = reason
    run.resource_wait_details = details
    db.add(TrainingRunEvent(run_id=run.run_id, level=LogLevel.INFO, event_type="resource_wait", message=reason, data=details))


def _request(run: TrainingRun, worker: GpuWorkerInstance, bindings: list[GpuCudaBinding]) -> tuple[dict[str, Any] | None, str | None]:
    request = run.resource_request
    if request:
        return {
            "selection": request.selection, "gpu_count": request.gpu_count,
            "gpu_uuids": list(request.gpu_uuids or []), "memory_mib_per_gpu": request.memory_mib_per_gpu,
            "sharing": request.sharing, "node_id": request.node_id, "source": "user",
        }, None
    device = str(run.parameters.device if run.parameters else "auto").strip().lower()
    if device == "cpu":
        return None, "cpu_legacy_mode"
    if device in {"", "auto", "cuda", "gpu"}:
        return {"selection": "auto", "gpu_count": 1, "gpu_uuids": [], "memory_mib_per_gpu": None, "sharing": "exclusive", "node_id": None, "source": "legacy"}, None
    try:
        ordinals = list(dict.fromkeys(int(part.strip()) for part in device.split(",")))
    except ValueError:
        return None, "cuda_binding_unavailable"
    visible_host_ids = parse_visible_host_gpu_ids(worker.nvidia_visible_devices)
    host_to_local = ({host_id: local for local, host_id in enumerate(visible_host_ids)}
                     if visible_host_ids is not None else {index: index for index in ordinals})
    by_local = {item.ordinal: item.gpu_uuid for item in bindings if item.present and item.status in {"success", "partial"}}
    cuda_mask = worker.cuda_visible_devices
    if cuda_mask is not None:
        try:
            base_ordinals = [int(part.strip()) for part in cuda_mask.split(",")]
        except ValueError:
            return None, "cuda_binding_unavailable"
        by_local = {base: by_local[local] for local, base in enumerate(base_ordinals) if local in by_local}
    if any(host_id not in host_to_local or host_to_local[host_id] not in by_local for host_id in ordinals):
        return None, "cuda_binding_unavailable"
    return {"selection": "manual", "gpu_count": len(ordinals), "gpu_uuids": [by_local[host_to_local[x]] for x in ordinals], "memory_mib_per_gpu": None, "sharing": "exclusive", "node_id": None, "source": "legacy"}, None


def _active_for_gpu(db: Session, gpu_uuid: str) -> list[GpuAllocationDevice]:
    return (
        db.query(GpuAllocationDevice)
        .join(GpuAllocation)
        .options(joinedload(GpuAllocationDevice.allocation))
        .filter(GpuAllocationDevice.gpu_uuid == gpu_uuid, GpuAllocation.state != "released")
        .populate_existing().with_for_update()
        .all()
    )


def allocate_run(
    db: Session, run_id: str, worker_instance_id: str, *, scheduler_enabled: bool,
    shared_execution_enabled: bool, node_defaults: dict[str, Any],
    stale_after_seconds: int, start_timeout_seconds: int = 30, now: datetime | None = None,
) -> AllocationDecision:
    instant = _now(now)
    db.flush()
    run = db.query(TrainingRun).filter_by(run_id=run_id).populate_existing().with_for_update().one()
    worker_hint = db.get(GpuWorkerInstance, worker_instance_id)
    active_allocation = db.query(GpuAllocation.allocation_id).filter(
        GpuAllocation.run_id == run_id, GpuAllocation.state != "released",
    ).populate_existing().with_for_update().first()
    if (run.status != TrainingRunStatus.QUEUED or run.queued_at is None or run.hidden
            or run.cancel_requested_at or run.delete_requested_at or run.current_allocation_id
            or active_allocation):
        return AllocationDecision(None, "not_eligible", {}, None)
    if not worker_hint or not worker_hint.node_id:
        _set_wait_reason(db, run, "node_id_unconfigured", {})
        return AllocationDecision(None, "node_id_unconfigured", {}, None)
    node = db.query(GpuNodeSchedulingState).filter_by(node_id=worker_hint.node_id).populate_existing().with_for_update().first()
    if node is None:
        node = GpuNodeSchedulingState(
            node_id=worker_hint.node_id, managed=bool(scheduler_enabled),
            accepting_allocations=bool(scheduler_enabled),
            shared_execution_enabled=bool(shared_execution_enabled),
            max_shared_tasks_per_device=int(node_defaults.get("max_shared_tasks_per_device", 2)),
            memory_safety_mib=int(node_defaults.get("memory_safety_mib", 4096)),
        )
        db.add(node); db.flush()
    elif scheduler_enabled and not node.managed:
        node.managed = True
        node.accepting_allocations = True
    worker = db.query(GpuWorkerInstance).filter_by(instance_id=worker_instance_id).populate_existing().with_for_update().one()
    engine = str(run.architecture.engine).lower()
    cutoff = instant - timedelta(seconds=stale_after_seconds)
    if engine not in set(worker.allowed_engines or []) or worker.stopped_at or _aware(worker.heartbeat_at) < cutoff:
        return AllocationDecision(None, "no_compatible_worker", {}, None)
    if run.hidden or run.claimed_at is not None and run.worker_id not in {None, worker.worker_id}:
        return AllocationDecision(None, "not_eligible", {}, None)
    if not scheduler_enabled or not node.managed or not node.accepting_allocations or not worker.accepting_tasks:
        _set_wait_reason(db, run, "scheduler_disabled", {})
        return AllocationDecision(None, "scheduler_disabled", {}, None)
    bindings = db.query(GpuCudaBinding).filter_by(instance_id=worker.instance_id, present=True).order_by(GpuCudaBinding.gpu_uuid).populate_existing().with_for_update().all()
    request, error = _request(run, worker, bindings)
    if error:
        _set_wait_reason(db, run, error, {})
        return AllocationDecision(None, error, {}, request)
    if request is None:
        return AllocationDecision(None, "cpu_legacy_mode", {}, None)
    run.resource_wait_details = {**(run.resource_wait_details or {}), "effective_request": request}
    if request.get("node_id") and request["node_id"] != worker.node_id:
        return AllocationDecision(None, "no_compatible_worker", {}, request)
    if request["gpu_count"] > 1 and engine != "ultralytics-yolo":
        _set_wait_reason(db, run, "no_compatible_worker", {"engine": engine})
        return AllocationDecision(None, "no_compatible_worker", {}, request)
    if request["sharing"] == "shared" and engine != "ultralytics-yolo":
        _set_wait_reason(db, run, "unsupported_sharing_engine", {"engine": engine})
        return AllocationDecision(None, "unsupported_sharing_engine", {}, request)
    if request["sharing"] == "shared" and not node.shared_execution_enabled:
        _set_wait_reason(db, run, "scheduler_disabled", {"shared_execution_enabled": False})
        return AllocationDecision(None, "scheduler_disabled", {}, request)
    untracked = (db.query(TrainingRun)
                 .outerjoin(GpuWorkerInstance, GpuWorkerInstance.worker_id == TrainingRun.worker_id)
                 .outerjoin(TrainingRunParameters, TrainingRunParameters.run_id == TrainingRun.run_id)
                 .filter(or_(GpuWorkerInstance.node_id == worker.node_id, GpuWorkerInstance.node_id.is_(None)),
                         or_(TrainingRunParameters.device != "cpu", TrainingRunParameters.device.is_(None)),
                         TrainingRun.status == TrainingRunStatus.RUNNING,
                         TrainingRun.current_allocation_id.is_(None)).first())
    if untracked:
        _set_wait_reason(db, run, "legacy_execution_untracked", {"run_id": untracked.run_id})
        return AllocationDecision(None, "legacy_execution_untracked", {}, request)
    candidates: list[tuple[GpuDevice, GpuWorkerObservation, GpuCudaBinding, int]] = []
    wanted = set(request["gpu_uuids"]) if request["selection"] == "manual" else None
    locked_devices = {item.gpu_uuid: item for item in
        db.query(GpuDevice).filter(GpuDevice.gpu_uuid.in_(sorted({x.gpu_uuid for x in bindings})))
        .order_by(GpuDevice.gpu_uuid).populate_existing().with_for_update().all()}
    active_count = len(db.query(GpuAllocation).filter(
        GpuAllocation.worker_instance_id == worker.instance_id,
        GpuAllocation.state != "released",
    ).populate_existing().with_for_update().all())
    if active_count >= worker.max_training_slots:
        _set_wait_reason(db, run, "worker_slot_limit", {"max_training_slots": worker.max_training_slots})
        return AllocationDecision(None, "worker_slot_limit", {}, request)
    rejected: set[str] = set()
    for binding in bindings:
        if binding.environment_fingerprint != worker.cuda_environment_fingerprint or _aware(binding.verified_at) < cutoff:
            rejected.add("cuda_binding_unavailable"); continue
        device = locked_devices.get(binding.gpu_uuid)
        if not device or device.node_id != worker.node_id or wanted is not None and device.gpu_uuid not in wanted:
            rejected.add("node_id_unconfigured"); continue
        conflicting_node = (db.query(GpuWorkerObservation.instance_id)
            .join(GpuWorkerInstance, GpuWorkerInstance.instance_id == GpuWorkerObservation.instance_id)
            .filter(GpuWorkerObservation.gpu_uuid == device.gpu_uuid,
                    GpuWorkerInstance.node_id.is_not(None), GpuWorkerInstance.node_id != device.node_id)
            .first())
        if conflicting_node:
            rejected.add("node_id_unconfigured")
            continue
        observation = (
            db.query(GpuWorkerObservation).filter_by(instance_id=worker.instance_id, gpu_uuid=device.gpu_uuid, present=True)
            .filter(GpuWorkerObservation.sampled_at >= cutoff).order_by(GpuWorkerObservation.sampled_at.desc()).populate_existing().with_for_update().first()
        )
        if not observation or None in (observation.memory_total_mib, observation.memory_used_mib, observation.memory_free_mib):
            rejected.add("gpu_observation_stale"); continue
        if observation.mig_mode == "enabled" or observation.compute_mode == "prohibited":
            rejected.add("cuda_binding_unavailable"); continue
        if request["sharing"] == "shared" and (observation.compute_mode != "default" or observation.mig_mode not in {"disabled", "not_supported"}):
            rejected.add("cuda_binding_unavailable"); continue
        active = _active_for_gpu(db, device.gpu_uuid)
        if active and (request["sharing"] == "exclusive" or any(x.sharing == "exclusive" for x in active)):
            rejected.add("gpu_exclusive_busy"); continue
        if request["sharing"] == "shared" and len(active) >= node.max_shared_tasks_per_device:
            rejected.add("shared_task_limit"); continue
        commitments = [ActiveCommitment(x.allocation_id, x.reserved_memory_mib) for x in active]
        accounting = calculate_gpu_accounting(
            {"memory_total_mib": observation.memory_total_mib, "memory_used_mib": observation.memory_used_mib,
             "memory_free_mib": observation.memory_free_mib, "sampled_at": observation.sampled_at},
            commitments, node.memory_safety_mib,
            reliable_usage_by_allocation=(observation.process_snapshot or {}).get("usage_by_allocation"),
            process_mapping_complete=bool((observation.process_snapshot or {}).get("attribution_complete")),
            process_mapping_has_duplicates=bool((observation.process_snapshot or {}).get("has_duplicates")),
            stale_after_seconds=stale_after_seconds, now=instant,
        )
        requested = request["memory_mib_per_gpu"] if request["sharing"] == "shared" else observation.memory_total_mib
        if request["sharing"] == "shared" and not accounting.admits(requested):
            rejected.add("insufficient_memory_budget"); continue
        if request["sharing"] == "exclusive" and not accounting.admits(request["memory_mib_per_gpu"] or 1):
            rejected.add("insufficient_memory_budget"); continue
        candidates.append((device, observation, binding, accounting.available_budget_mib))
    candidates.sort(key=lambda item: (-item[3], item[0].gpu_uuid))
    selected = candidates[:int(request["gpu_count"])]
    if wanted is not None:
        selected.sort(key=lambda item: request["gpu_uuids"].index(item[0].gpu_uuid))
    if len(selected) != int(request["gpu_count"]):
        priority = ("gpu_exclusive_busy", "shared_task_limit", "insufficient_memory_budget", "cuda_binding_unavailable", "gpu_observation_stale", "node_id_unconfigured")
        reason = next((item for item in priority if item in rejected), "insufficient_memory_budget")
        _set_wait_reason(db, run, reason, {"requested_gpu_count": request["gpu_count"], "eligible_gpu_count": len(candidates)})
        return AllocationDecision(None, reason, {}, request)
    allocation = GpuAllocation(
        allocation_id=str(uuid.uuid4()), run_id=run.run_id, active_run_id=run.run_id,
        worker_instance_id=worker.instance_id, worker_id=worker.worker_id, node_id=worker.node_id,
        state="reserved", authorization_state="issued", request_snapshot=request,
        launcher_identity=worker.launcher_identity, reserved_at=instant,
        launch_deadline_at=instant + timedelta(seconds=start_timeout_seconds), heartbeat_at=instant,
    )
    db.add(allocation); db.flush()
    for ordinal, (device, observation, _binding, _available) in enumerate(selected):
        requested = request["memory_mib_per_gpu"]
        reserved = requested if request["sharing"] == "shared" else observation.memory_total_mib
        db.add(GpuAllocationDevice(allocation_id=allocation.allocation_id, gpu_uuid=device.gpu_uuid, ordinal=ordinal, sharing=request["sharing"], requested_memory_mib=requested, reserved_memory_mib=reserved))
    run.current_allocation_id = allocation.allocation_id
    run.worker_id = worker.worker_id
    run.claimed_at = instant
    run.resource_wait_reason = None
    run.resource_wait_details = None
    db.flush()
    return AllocationDecision(allocation, None, {}, request)


def reserve_next(
    db: Session, worker_instance_id: str, *, scan_cursor: AllocationScanCursor | None = None,
    **kwargs,
) -> AllocationDecision:
    worker = db.get(GpuWorkerInstance, worker_instance_id)
    if not worker:
        return AllocationDecision(None, "no_compatible_worker", {}, None)
    active = exists().where(and_(
        GpuAllocation.run_id == TrainingRun.run_id,
        GpuAllocation.state != "released",
    ))
    query = (
        db.query(TrainingRun)
        .join(ModelArchitecture, ModelArchitecture.architecture_id == TrainingRun.architecture_id)
        .outerjoin(TrainingRunResourceRequest, TrainingRunResourceRequest.run_id == TrainingRun.run_id)
        .filter(
            TrainingRun.status == TrainingRunStatus.QUEUED,
            TrainingRun.queued_at.is_not(None), TrainingRun.hidden.is_(False),
            TrainingRun.cancel_requested_at.is_(None), TrainingRun.delete_requested_at.is_(None),
            TrainingRun.current_allocation_id.is_(None), ~active,
            func.lower(ModelArchitecture.engine).in_([str(item).lower() for item in (worker.allowed_engines or [])]),
            or_(TrainingRunResourceRequest.node_id.is_(None),
                TrainingRunResourceRequest.node_id == worker.node_id),
        )
    )
    if scan_cursor and scan_cursor.queued_at is not None:
        query = query.filter(or_(
            TrainingRun.queued_at > scan_cursor.queued_at,
            and_(TrainingRun.queued_at == scan_cursor.queued_at,
                 TrainingRun.created_at > scan_cursor.created_at),
            and_(TrainingRun.queued_at == scan_cursor.queued_at,
                 TrainingRun.created_at == scan_cursor.created_at,
                 TrainingRun.run_id > scan_cursor.run_id),
        ))
    run = query.order_by(
        TrainingRun.queued_at.asc(), TrainingRun.created_at.asc(), TrainingRun.run_id.asc(),
    ).first()
    if run is None:
        if scan_cursor:
            scan_cursor.reset()
        return AllocationDecision(None, "scan_exhausted", {}, None)
    if scan_cursor:
        scan_cursor.queued_at = run.queued_at
        scan_cursor.created_at = run.created_at
        scan_cursor.run_id = str(run.run_id)
    return allocate_run(db, run.run_id, worker_instance_id, **kwargs)


def adopt_legacy_execution(
    db: Session, *, run_id: str, worker_instance_id: str, execution_owner: dict,
    assigned_gpu_uuids: list[str], cuda_environment_fingerprint: str,
    stale_after_seconds: int = 20, now: datetime | None = None,
) -> AllocationDecision:
    """Record a proven pre-scheduler execution as exclusive, without relaunching it."""
    from train_platform.platform.runtime.process_scope import compare_process_scope

    db.flush()
    instant = _now(now)
    cutoff = instant - timedelta(seconds=stale_after_seconds)
    run = db.query(TrainingRun).filter_by(run_id=run_id).populate_existing().with_for_update().one()
    unavailable = AllocationDecision(None, "legacy_execution_untracked", {}, None)
    if run.status != TrainingRunStatus.RUNNING or run.current_allocation_id or not assigned_gpu_uuids:
        return unavailable
    if len(set(assigned_gpu_uuids)) != len(assigned_gpu_uuids):
        return unavailable
    hint = db.get(GpuWorkerInstance, worker_instance_id)
    if not hint or not hint.node_id:
        return unavailable
    node = db.query(GpuNodeSchedulingState).filter_by(node_id=hint.node_id).populate_existing().with_for_update().first()
    worker = db.query(GpuWorkerInstance).filter_by(instance_id=worker_instance_id).populate_existing().with_for_update().one()
    if (not node or not node.managed or worker.stopped_at or _aware(worker.heartbeat_at) < cutoff
            or compare_process_scope(execution_owner.get("process_scope"), worker.process_scope) != "same"
            or execution_owner.get("guard_pid") != run.pid
            or execution_owner.get("worker_id") != run.worker_id
            or not isinstance(execution_owner.get("guard_create_time"), (int, float))
            or worker.cuda_environment_fingerprint != cuda_environment_fingerprint):
        return unavailable
    devices = db.query(GpuDevice).filter(GpuDevice.gpu_uuid.in_(assigned_gpu_uuids)).order_by(
        GpuDevice.gpu_uuid).populate_existing().with_for_update().all()
    if len(devices) != len(assigned_gpu_uuids) or any(item.node_id != node.node_id for item in devices):
        return unavailable
    bindings = db.query(GpuCudaBinding).filter(
        GpuCudaBinding.instance_id == worker_instance_id, GpuCudaBinding.gpu_uuid.in_(assigned_gpu_uuids),
        GpuCudaBinding.present.is_(True), GpuCudaBinding.verified_at >= cutoff,
        GpuCudaBinding.environment_fingerprint == cuda_environment_fingerprint,
        GpuCudaBinding.status.in_(("success", "partial")),
    ).populate_existing().with_for_update().all()
    if len(bindings) != len(devices):
        return unavailable
    observations = {}
    for device in devices:
        if _active_for_gpu(db, device.gpu_uuid):
            return unavailable
        observation = db.query(GpuWorkerObservation).filter(
            GpuWorkerObservation.instance_id == worker_instance_id,
            GpuWorkerObservation.gpu_uuid == device.gpu_uuid, GpuWorkerObservation.present.is_(True),
            GpuWorkerObservation.sampled_at >= cutoff,
        ).populate_existing().with_for_update().first()
        if (not observation or not observation.memory_total_mib
                or observation.mig_mode not in {"disabled", "not_supported"}):
            return unavailable
        observations[device.gpu_uuid] = observation
    allocation_id = str(uuid.uuid4())
    request = {"source": "legacy_adopted", "selection": "manual", "gpu_count": len(devices),
               "gpu_uuids": list(assigned_gpu_uuids), "sharing": "exclusive",
               "memory_mib_per_gpu": None, "node_id": node.node_id,
               "legacy_execution_owner": dict(execution_owner)}
    owner = {**execution_owner, "allocation_id": allocation_id, "worker_instance_id": worker_instance_id}
    allocation = GpuAllocation(
        allocation_id=allocation_id, run_id=run_id, active_run_id=run_id,
        worker_instance_id=worker_instance_id, worker_id=run.worker_id, node_id=node.node_id,
        state="running", authorization_state="consumed", request_snapshot=request,
        execution_owner=owner, launcher_identity=worker.launcher_identity,
        reserved_at=instant, launch_deadline_at=instant, started_at=run.started_at or instant,
        heartbeat_at=instant,
    )
    db.add(allocation)
    db.flush()
    for ordinal, gpu_uuid in enumerate(assigned_gpu_uuids):
        db.add(GpuAllocationDevice(allocation_id=allocation_id, gpu_uuid=gpu_uuid, ordinal=ordinal,
                                  sharing="exclusive", reserved_memory_mib=observations[gpu_uuid].memory_total_mib))
    run.current_allocation_id = allocation_id
    run.resource_wait_reason = None
    run.resource_wait_details = None
    db.add(TrainingRunEvent(run_id=run_id, event_type="resource_adopted",
                           message="Verified legacy GPU execution registered as exclusive"))
    db.flush()
    return AllocationDecision(allocation, None, {}, request)
