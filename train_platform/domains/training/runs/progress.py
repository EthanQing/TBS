from __future__ import annotations

from typing import Mapping

from train_platform.db.session import SessionLocal
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun, TrainingRunEpochMetric

from .lifecycle import touch_heartbeat


def _merge_metrics(existing: Mapping | None, incoming: Mapping | None) -> dict:
    merged: dict = {}
    if isinstance(existing, Mapping):
        merged.update({str(key): value for key, value in existing.items()})
    if isinstance(incoming, Mapping):
        merged.update({str(key): value for key, value in incoming.items()})
    return merged


def upsert_epoch_metrics(
    run_id: str,
    epoch: int,
    metrics: Mapping,
    *,
    expected_pid: int | None = None,
) -> None:
    """Persist epoch metrics and progress without reviving a terminal run."""

    db = SessionLocal()
    try:
        run_query = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id))
        try:
            run_query = run_query.with_for_update()
        except Exception:
            pass
        try:
            run = run_query.first()
        except Exception:
            db.rollback()
            run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
        if not run or run.status != TrainingRunStatus.RUNNING:
            return
        if expected_pid is not None and (run.pid is None or int(run.pid) != int(expected_pid)):
            return

        row = (
            db.query(TrainingRunEpochMetric)
            .filter(
                TrainingRunEpochMetric.run_id == str(run_id),
                TrainingRunEpochMetric.epoch == int(epoch),
            )
            .first()
        )
        payload = _merge_metrics({}, metrics)
        if row:
            row.metrics = _merge_metrics(row.metrics, payload)
        else:
            db.add(TrainingRunEpochMetric(run_id=str(run_id), epoch=int(epoch), metrics=payload))

        touch_heartbeat(db, str(run_id), expected_pid=expected_pid, commit=False)
        run.current_epoch = int(epoch)
        if run.total_epochs and int(run.total_epochs) > 0:
            run.progress = int(min(100, max(0, 100 * float(epoch + 1) / float(run.total_epochs))))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


__all__ = ["upsert_epoch_metrics"]
