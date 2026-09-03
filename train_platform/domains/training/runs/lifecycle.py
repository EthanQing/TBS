from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from train_platform.models.v3.enums import LogLevel, TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun, TrainingRunEvent
from train_platform.utils.exceptions import ConflictError, NotFoundError


_TERMINAL_STATUSES = {
    TrainingRunStatus.COMPLETED,
    TrainingRunStatus.FAILED,
    TrainingRunStatus.CANCELLED,
    TrainingRunStatus.DELETED,
}


@dataclass(frozen=True)
class FinalizeResult:
    run_id: str
    changed: bool
    status: TrainingRunStatus
    run: TrainingRun


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_run(db: Session, run_id: str, *, for_update: bool = False) -> TrainingRun:
    query = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id))
    if for_update:
        try:
            query = query.with_for_update()
        except Exception:
            pass
    try:
        run = query.populate_existing().first()
    except Exception:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
    if not run:
        raise NotFoundError("Training run not found")
    return run


def _event(db: Session, run_id: str, event_type: str, message: str, *, level: LogLevel = LogLevel.INFO) -> None:
    db.add(TrainingRunEvent(run_id=str(run_id), level=level, event_type=event_type, message=message))


def _clear_claim(run: TrainingRun) -> None:
    run.worker_id = None
    run.claimed_at = None
    run.pid = None
    run.heartbeat_at = None


def _queue_locked(run: TrainingRun, *, now: datetime) -> None:
    run.queued_at = run.queued_at or now
    run.hidden = False
    run.status = TrainingRunStatus.QUEUED
    run.started_at = None
    run.finished_at = None
    run.error_message = None
    run.cancel_requested_at = None
    run.cancel_reason = None
    _clear_claim(run)


def queue_run(db: Session, run_id: str) -> TrainingRun:
    run = _load_run(db, run_id, for_update=True)
    if run.status in (TrainingRunStatus.RUNNING, TrainingRunStatus.COMPLETED):
        raise ConflictError(f"Run status is {run.status}; cannot queue")
    if run.status == TrainingRunStatus.DELETED:
        raise ConflictError("Run is deleted")
    _queue_locked(run, now=_utcnow())
    _event(db, run.run_id, "queued", "Run queued")
    db.commit()
    db.refresh(run)
    return run


def resume_run(db: Session, run_id: str, *, has_resume_checkpoint: bool) -> TrainingRun:
    run = _load_run(db, run_id, for_update=True)
    if run.status == TrainingRunStatus.COMPLETED:
        raise ConflictError("Run is COMPLETED and cannot be resumed; create a new training run instead")
    if run.status not in (TrainingRunStatus.CANCELLED, TrainingRunStatus.FAILED):
        raise ConflictError(f"Run status is {run.status}; must be CANCELLED or FAILED to resume")

    if run.parameters:
        additional = run.parameters.additional_params or {}
        if not isinstance(additional, dict):
            additional = {}
        additional["resume_training"] = bool(has_resume_checkpoint)
        additional["resume_job_id"] = None
        run.parameters.additional_params = additional
        db.add(run.parameters)

    if not has_resume_checkpoint:
        run.current_epoch = 0
        run.progress = 0
    _queue_locked(run, now=_utcnow())
    resume_message = (
        "Run resume requested using weights/last.pt"
        if has_resume_checkpoint
        else "No weights/last.pt found; queued run to restart with saved parameters"
    )
    _event(db, run.run_id, "resumed", resume_message)
    _event(db, run.run_id, "queued", "Run queued")
    db.commit()
    db.refresh(run)
    return run


def mark_started(
    db: Session,
    run_id: str,
    *,
    worker_id: str,
    pid: int,
    started_at: datetime | None = None,
) -> TrainingRun:
    run = _load_run(db, run_id, for_update=True)
    if run.status != TrainingRunStatus.QUEUED:
        raise ConflictError(f"Run status is {run.status}; cannot start")
    if run.worker_id is not None and str(run.worker_id) != str(worker_id):
        raise ConflictError("Run is already claimed by another worker")
    now = started_at or _utcnow()
    run.queued_at = run.queued_at or now
    run.claimed_at = now
    run.worker_id = str(worker_id)
    run.pid = int(pid)
    run.heartbeat_at = now
    run.started_at = run.started_at or now
    run.finished_at = None
    run.error_message = None
    run.status = TrainingRunStatus.RUNNING
    _event(db, run.run_id, "started", f"Run started by worker {worker_id}")
    db.commit()
    db.refresh(run)
    return run


def touch_heartbeat(
    db: Session,
    run_id: str,
    *,
    execution_owner: str | None = None,
    heartbeat_at: datetime | None = None,
    commit: bool = True,
) -> bool:
    run = _load_run(db, run_id)
    if run.status != TrainingRunStatus.RUNNING:
        return False
    if execution_owner is not None and str(run.worker_id or "") != str(execution_owner):
        return False
    run.heartbeat_at = heartbeat_at or _utcnow()
    if commit:
        db.commit()
    return True


def release_stale_claim(db: Session, run_id: str) -> TrainingRun:
    run = _load_run(db, run_id, for_update=True)
    if run.status != TrainingRunStatus.QUEUED or run.worker_id is None:
        return run
    _clear_claim(run)
    _event(db, run.run_id, "requeue", "Released stale claim; re-queued for another worker")
    db.commit()
    db.refresh(run)
    return run


def request_cancel(db: Session, run_id: str, *, reason: str | None = None) -> TrainingRun:
    run = _load_run(db, run_id, for_update=True)
    if run.status in _TERMINAL_STATUSES:
        return run
    if run.cancel_requested_at is None:
        run.cancel_requested_at = _utcnow()
    if reason:
        run.cancel_reason = str(reason)
    _event(db, run.run_id, "cancel_requested", reason or "Cancel requested")
    if run.status in (TrainingRunStatus.CREATED, TrainingRunStatus.QUEUED):
        run.status = TrainingRunStatus.CANCELLED
        run.finished_at = _utcnow()
        _clear_claim(run)
        _event(db, run.run_id, "cancelled", "Run cancelled")
    db.commit()
    db.refresh(run)
    return run


def request_delete(db: Session, run_id: str) -> TrainingRun:
    run = _load_run(db, run_id, for_update=True)
    if run.status == TrainingRunStatus.DELETED:
        return run
    run.hidden = True
    if run.delete_requested_at is None:
        run.delete_requested_at = _utcnow()
    if run.cancel_requested_at is None:
        run.cancel_requested_at = _utcnow()
    _event(db, run.run_id, "delete_requested", "Delete requested")
    if run.status != TrainingRunStatus.RUNNING:
        run.status = TrainingRunStatus.DELETED
        run.finished_at = run.finished_at or _utcnow()
        _clear_claim(run)
        _event(db, run.run_id, "deleted", "Run marked as deleted")
    db.commit()
    db.refresh(run)
    return run


def finalize_execution(
    db: Session,
    run_id: str,
    *,
    exit_code: int,
    error_message: str | None = None,
) -> FinalizeResult:
    """Finalize one execution exactly once using the authoritative DB row."""

    run = _load_run(db, run_id, for_update=True)
    if run.status in _TERMINAL_STATUSES:
        return FinalizeResult(str(run.run_id), False, run.status, run)
    if run.status != TrainingRunStatus.RUNNING:
        return FinalizeResult(str(run.run_id), False, run.status, run)

    now = _utcnow()
    delete_requested = run.delete_requested_at is not None
    cancel_requested = run.cancel_requested_at is not None or delete_requested
    if delete_requested:
        status = TrainingRunStatus.DELETED
        message = "Run marked as deleted"
        run.hidden = True
        run.error_message = None
    elif cancel_requested:
        status = TrainingRunStatus.CANCELLED
        message = "Run cancelled"
        run.error_message = None
    elif int(exit_code) == 0:
        status = TrainingRunStatus.COMPLETED
        message = "Run completed"
        run.error_message = None
        run.progress = max(int(run.progress or 0), 100)
    else:
        status = TrainingRunStatus.FAILED
        message = str(error_message or f"Training subprocess exited with code {int(exit_code)}")
        run.error_message = message

    run.status = status
    run.finished_at = run.finished_at or now
    _clear_claim(run)
    _event(
        db,
        run.run_id,
        status.value,
        message,
        level=LogLevel.ERROR if status == TrainingRunStatus.FAILED else LogLevel.INFO,
    )
    db.commit()

    if status == TrainingRunStatus.COMPLETED:
        from .artifacts import index_completion_artifacts

        try:
            index_completion_artifacts(db, str(run.run_id))
            db.commit()
        except Exception:
            db.rollback()

    return FinalizeResult(str(run.run_id), True, status, run)


__all__ = [
    "FinalizeResult",
    "finalize_execution",
    "mark_started",
    "queue_run",
    "release_stale_claim",
    "request_cancel",
    "request_delete",
    "resume_run",
    "touch_heartbeat",
]
