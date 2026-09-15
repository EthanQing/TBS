from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from train_platform.domains.training.runs.lifecycle import finalize_execution, mark_started
from train_platform.models.v3.gpu_allocation import GpuAllocation
from train_platform.models.v3.gpu_allocation import GpuAllocationDevice, GpuNodeSchedulingState
from train_platform.models.v3.gpu_resource import GpuDevice, GpuWorkerInstance
from train_platform.models.v3.training_run import TrainingRun
from train_platform.platform.runtime.process_scope import compare_process_scope


def _now(value: datetime | None) -> datetime:
    return value or datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _locked(db: Session, allocation_id: str) -> GpuAllocation:
    # Callers may have recorded cancellation or an execution result in this
    # transaction. Current reads must not discard those pending mutations.
    db.flush()
    hint = db.get(GpuAllocation, allocation_id)
    if hint is None:
        raise ValueError("allocation not found")
    db.query(TrainingRun).filter_by(run_id=hint.run_id).populate_existing().with_for_update().one()
    db.query(GpuNodeSchedulingState).filter_by(node_id=hint.node_id).populate_existing().with_for_update().first()
    db.query(GpuWorkerInstance).filter_by(instance_id=hint.worker_instance_id).populate_existing().with_for_update().one()
    gpu_uuids = [row[0] for row in db.query(GpuAllocationDevice.gpu_uuid).filter_by(allocation_id=allocation_id).all()]
    if gpu_uuids:
        db.query(GpuDevice).filter(GpuDevice.gpu_uuid.in_(sorted(gpu_uuids))).order_by(GpuDevice.gpu_uuid).populate_existing().with_for_update().all()
    return db.query(GpuAllocation).filter_by(allocation_id=allocation_id).populate_existing().with_for_update().one()


def issue_start_authorization(db: Session, allocation_id: str, launcher_identity: dict, deadline_at: datetime) -> GpuAllocation:
    allocation = _locked(db, allocation_id)
    if allocation.state != "reserved" or allocation.authorization_state != "issued":
        raise ValueError("allocation cannot be authorized for start")
    allocation.state = "starting"
    allocation.launcher_identity = dict(launcher_identity)
    allocation.launch_deadline_at = deadline_at
    return allocation


def activate_allocation(
    db: Session, allocation_id: str, *, run_id: str, worker_instance_id: str,
    process_scope: dict, supervisor_pid: int, supervisor_create_time: float,
    assigned_gpu_uuids: list[str], now: datetime | None = None,
) -> dict[str, Any]:
    instant = _now(now)
    allocation = _locked(db, allocation_id)
    run = db.query(TrainingRun).filter_by(run_id=run_id).with_for_update().one()
    if allocation.run_id != run_id or allocation.worker_instance_id != worker_instance_id:
        raise ValueError("allocation execution identity mismatch")
    if allocation.authorization_state != "issued" or allocation.state not in {"reserved", "starting"}:
        raise ValueError("start authorization is not active")
    if instant > _aware(allocation.launch_deadline_at):
        raise ValueError("start authorization has expired")
    if run.cancel_requested_at or run.delete_requested_at:
        raise ValueError("training run is cancelled or pending deletion")
    worker = db.get(GpuWorkerInstance, worker_instance_id)
    launcher_scope = (allocation.launcher_identity or {}).get("process_scope")
    if compare_process_scope(worker.process_scope, process_scope) != "same" or compare_process_scope(launcher_scope, process_scope) != "same":
        raise ValueError("execution process scope does not match launcher")
    expected = [item.gpu_uuid for item in sorted(allocation.devices, key=lambda item: item.ordinal)]
    if assigned_gpu_uuids != expected:
        raise ValueError("assigned GPU UUIDs do not match allocation")
    owner = {
        "allocation_id": allocation_id,
        "worker_instance_id": worker_instance_id,
        "worker_id": allocation.worker_id,
        "process_scope": process_scope,
        "guard_pid": int(supervisor_pid),
        "guard_create_time": float(supervisor_create_time),
    }
    allocation.execution_owner = owner
    allocation.authorization_state = "consumed"
    allocation.state = "running"
    allocation.started_at = instant
    allocation.heartbeat_at = instant
    run.current_allocation_id = allocation_id
    mark_started(db, run_id, worker_id=allocation.worker_id, pid=supervisor_pid, started_at=instant, allocation_id=allocation_id, commit=False)
    return {"execution_owner": owner, "assigned_gpu_uuids": expected, "allocation_id": allocation_id}


def record_execution_result(db: Session, allocation_id: str, execution_owner: dict, exit_code: int, error_message: str | None) -> GpuAllocation:
    allocation = _locked(db, allocation_id)
    if allocation.execution_owner != execution_owner or allocation.state == "released":
        raise ValueError("execution owner mismatch")
    allocation.exit_code = int(exit_code)
    allocation.error_message = error_message
    allocation.execution_result = {"exit_code": int(exit_code), "error_message": error_message}
    return allocation


def request_releasing(db: Session, allocation_id: str, execution_owner: dict) -> GpuAllocation:
    allocation = _locked(db, allocation_id)
    if allocation.execution_owner != execution_owner:
        raise ValueError("execution owner mismatch")
    if allocation.state != "released":
        allocation.state = "releasing"
    return allocation


def revoke_unactivated_allocation(
    db: Session, allocation_id: str, *, run_id: str, worker_instance_id: str,
    reason: str, now: datetime | None = None,
) -> bool:
    allocation = _locked(db, allocation_id)
    if allocation.run_id != run_id or allocation.worker_instance_id != worker_instance_id:
        raise ValueError("allocation identity mismatch")
    if allocation.state == "released":
        return True
    if allocation.authorization_state == "consumed":
        allocation.state = "releasing"
        return False
    if allocation.state not in {"reserved", "starting"}:
        return allocation.state == "released"
    allocation.authorization_state = "revoked"
    allocation.state = "released"
    allocation.active_run_id = None
    allocation.released_at = _now(now)
    allocation.error_message = reason
    run = db.query(TrainingRun).filter_by(run_id=run_id).with_for_update().one()
    if run.current_allocation_id == allocation_id:
        run.current_allocation_id = None
        run.worker_id = None
        run.claimed_at = None
    return True


def finish_allocation(
    db: Session, allocation_id: str, execution_owner: dict, cleanup_proof: dict,
    *, now: datetime | None = None,
) -> GpuAllocation:
    allocation = _locked(db, allocation_id)
    if allocation.execution_owner != execution_owner:
        raise ValueError("execution owner mismatch")
    if allocation.exit_code is None:
        raise ValueError("execution result has not been recorded")
    if allocation.authorization_state != "consumed":
        raise ValueError("allocation was never activated")
    if (not cleanup_proof.get("complete") or cleanup_proof.get("allocation_id") != allocation_id
            or cleanup_proof.get("supervisor_excluded")):
        raise ValueError("complete allocation-scoped cleanup proof is required")
    if cleanup_proof.get("execution_owner") != execution_owner:
        raise ValueError("cleanup proof execution owner mismatch")
    if compare_process_scope(cleanup_proof.get("process_scope"), execution_owner.get("process_scope")) != "same":
        raise ValueError("cleanup proof process scope mismatch")
    owner = allocation.execution_owner or {}
    result = finalize_execution(
        db, allocation.run_id, exit_code=int(allocation.exit_code or 0),
        expected_pid=owner.get("guard_pid"), error_message=allocation.error_message,
        allocation_id=allocation_id, expected_create_time=owner.get("guard_create_time"), commit=False,
    )
    if not result.changed:
        raise ValueError("training run execution identity could not be finalized")
    allocation.state = "released"
    allocation.active_run_id = None
    allocation.released_at = _now(now)
    run = result.run
    if run.current_allocation_id == allocation_id:
        run.current_allocation_id = None
    return allocation
