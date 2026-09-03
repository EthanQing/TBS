from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from train_platform.domains.deployment import activation
from train_platform.domains.deployment.logs import append_run_log
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.models.v3.enums import (
    DeploymentRunPhase,
    DeploymentRunStatus,
    DeploymentTriggerType,
    LogLevel,
)
from train_platform.utils.exceptions import NotFoundError


STEPS: tuple[tuple[str, str, DeploymentRunPhase], ...] = (
    ("validate_artifacts", "Validate Artifacts", DeploymentRunPhase.VALIDATE_ARTIFACTS),
    ("materialize_runtime", "Materialize Runtime", DeploymentRunPhase.MATERIALIZE_RUNTIME),
    ("smoke_test", "Smoke Test", DeploymentRunPhase.SMOKE_TEST),
    ("activate", "Activate", DeploymentRunPhase.ACTIVATE),
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_steps() -> list[dict[str, Any]]:
    return [{"key": key, "name": name, "status": "pending"} for key, name, _ in STEPS]


def new_queued_run(
    *,
    run_id: str,
    deployment_id: int,
    project_id: int,
    model_version_id: int,
    snapshot: dict[str, Any],
) -> DeploymentRun:
    """Construct the only initial run state used by the run service."""
    return DeploymentRun(
        run_id=str(run_id),
        deployment_id=int(deployment_id),
        project_id=int(project_id),
        model_version_id=int(model_version_id),
        trigger_type=DeploymentTriggerType.MANUAL,
        status=DeploymentRunStatus.QUEUED,
        phase=DeploymentRunPhase.PREPARING,
        current_step=None,
        progress=0,
        cancel_requested=False,
        snapshot=deepcopy(snapshot),
    )


def get_locked_run(db: Session, run_id: str) -> DeploymentRun:
    run = (
        db.query(DeploymentRun)
        .filter(DeploymentRun.run_id == str(run_id))
        .with_for_update()
        .first()
    )
    if not run:
        raise NotFoundError("Deployment run not found")
    return run


def _snapshot(run: DeploymentRun) -> dict[str, Any]:
    return deepcopy(run.snapshot) if isinstance(run.snapshot, dict) else {}


def _set_step(run: DeploymentRun, *, key: str, status: str, detail: str | None = None) -> None:
    snapshot = _snapshot(run)
    steps = snapshot.get("steps")
    if not isinstance(steps, list):
        steps = snapshot_steps()
    for step in steps:
        if isinstance(step, dict) and str(step.get("key") or "") == str(key):
            step["status"] = str(status)
            if detail:
                step["detail"] = str(detail)
            step["updated_at"] = utcnow().isoformat()
    snapshot["steps"] = steps
    run.snapshot = snapshot


def _begin_step_locked(
    db: Session,
    run: DeploymentRun,
    *,
    key: str,
    progress: int,
) -> DeploymentRun:
    phase = next((phase for step_key, _, phase in STEPS if step_key == key), None)
    if phase is None:
        raise ValueError(f"Unknown deployment step: {key}")
    run.phase = phase
    run.current_step = str(key)
    run.progress = int(progress)
    _set_step(run, key=key, status="running")
    names = {step_key: name for step_key, name, _ in STEPS}
    append_run_log(
        db,
        run.run_id,
        level=LogLevel.INFO,
        message=f"{names[key]} started",
        step_key=key,
        action="start",
    )
    return run


def update_execution_metadata(db: Session, run_id: str, **values: Any) -> DeploymentRun:
    run = get_locked_run(db, run_id)
    snapshot = _snapshot(run)
    for key, value in values.items():
        snapshot[str(key)] = deepcopy(value)
    run.snapshot = snapshot
    return run


def mark_running(db: Session, run_id: str) -> DeploymentRun | None:
    run = get_locked_run(db, run_id)
    if run.status in {
        DeploymentRunStatus.COMPLETED,
        DeploymentRunStatus.FAILED,
        DeploymentRunStatus.CANCELLED,
    }:
        return None
    if run.status == DeploymentRunStatus.QUEUED:
        run.status = DeploymentRunStatus.RUNNING
        run.phase = DeploymentRunPhase.PREPARING
        run.started_at = run.started_at or utcnow()
        run.current_step = None
        run.progress = 1
        activation.prepare_deployment_for_run(db, deployment_id=int(run.deployment_id))
    return run


def begin_step(
    db: Session,
    run_id: str,
    *,
    key: str,
    progress: int,
) -> DeploymentRun | None:
    run = get_locked_run(db, run_id)
    if run.status != DeploymentRunStatus.RUNNING or run.cancel_requested:
        return None
    return _begin_step_locked(db, run, key=key, progress=progress)


def begin_activation(db: Session, run_id: str) -> DeploymentRun | None:
    """Atomically admit the final activation step or complete cancellation."""
    run = get_locked_run(db, run_id)
    if run.status != DeploymentRunStatus.RUNNING:
        return None
    if run.cancel_requested:
        _mark_cancelled_locked(db, run, reason="Cancelled before activation")
        return None
    return _begin_step_locked(db, run, key="activate", progress=90)


def complete_step(
    db: Session,
    run_id: str,
    *,
    key: str,
    progress: int,
    message: str,
    detail: dict[str, Any] | None = None,
) -> DeploymentRun:
    run = get_locked_run(db, run_id)
    if run.status != DeploymentRunStatus.RUNNING or run.cancel_requested:
        return run
    _set_step(run, key=key, status="completed")
    run.progress = int(progress)
    append_run_log(
        db,
        run_id,
        level=LogLevel.INFO,
        message=message,
        step_key=key,
        action="completed",
        detail=detail,
    )
    return run


def request_cancel(db: Session, run_id: str) -> DeploymentRun:
    run = get_locked_run(db, run_id)
    if run.status in {
        DeploymentRunStatus.COMPLETED,
        DeploymentRunStatus.FAILED,
        DeploymentRunStatus.CANCELLED,
    }:
        return run
    run.cancel_requested = True
    if run.status == DeploymentRunStatus.QUEUED:
        _mark_cancelled_locked(db, run, reason="Cancelled before execution")
    else:
        append_run_log(
            db,
            run_id,
            level=LogLevel.WARNING,
            message="Cancellation requested",
            step_key=run.current_step,
            action="cancel_requested",
        )
    return run


def _mark_cancelled_locked(db: Session, run: DeploymentRun, *, reason: str) -> DeploymentRun:
    if run.status == DeploymentRunStatus.CANCELLED:
        return run
    run.status = DeploymentRunStatus.CANCELLED
    run.phase = DeploymentRunPhase.CANCELLED
    run.finished_at = utcnow()
    run.error_message = str(reason)
    if run.current_step:
        _set_step(run, key=str(run.current_step), status="cancelled", detail="cancel requested")
    activation.mark_deployment_cancelled(db, deployment_id=int(run.deployment_id))
    append_run_log(
        db,
        run.run_id,
        level=LogLevel.WARNING,
        message="Deployment run cancelled",
        step_key=run.current_step,
        action="cancelled",
        detail={"reason": str(reason)},
    )
    return run


def mark_cancelled(db: Session, run_id: str, *, reason: str = "Cancellation requested") -> DeploymentRun:
    run = get_locked_run(db, run_id)
    return _mark_cancelled_locked(db, run, reason=reason)


def cancel_if_requested(db: Session, run_id: str) -> bool:
    run = get_locked_run(db, run_id)
    if run.status == DeploymentRunStatus.CANCELLED:
        return True
    if run.status != DeploymentRunStatus.RUNNING or not run.cancel_requested:
        return False
    _mark_cancelled_locked(db, run, reason="Cancelled at pipeline boundary")
    return True


def mark_failed(db: Session, run_id: str, *, error: str) -> DeploymentRun:
    run = get_locked_run(db, run_id)
    if run.status in {DeploymentRunStatus.COMPLETED, DeploymentRunStatus.CANCELLED}:
        return run
    if run.status == DeploymentRunStatus.FAILED:
        return run
    message = str(error)
    run.status = DeploymentRunStatus.FAILED
    run.finished_at = utcnow()
    run.error_message = message
    if run.current_step:
        _set_step(run, key=str(run.current_step), status="failed", detail=message)
    activation.mark_deployment_failed(db, deployment_id=int(run.deployment_id))
    append_run_log(
        db,
        run.run_id,
        level=LogLevel.ERROR,
        message=f"Deployment run failed: {message}",
        step_key=run.current_step,
        action="failed",
        detail={"error": message},
    )
    return run


def mark_completed(db: Session, run_id: str) -> DeploymentRun:
    run = get_locked_run(db, run_id)
    if run.status != DeploymentRunStatus.RUNNING:
        return run
    if run.cancel_requested:
        return _mark_cancelled_locked(db, run, reason="Cancelled before completion")
    run.status = DeploymentRunStatus.COMPLETED
    run.phase = DeploymentRunPhase.DONE
    run.progress = 100
    run.finished_at = utcnow()
    run.error_message = None
    return run


__all__ = [
    "STEPS",
    "begin_activation",
    "begin_step",
    "cancel_if_requested",
    "complete_step",
    "get_locked_run",
    "mark_cancelled",
    "mark_completed",
    "mark_failed",
    "mark_running",
    "new_queued_run",
    "request_cancel",
    "snapshot_steps",
    "update_execution_metadata",
]
