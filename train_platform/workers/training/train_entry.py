from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from sqlalchemy.orm import Session
from train_platform.models.enums import TrainingRunStatus
from train_platform.models.training_run import TrainingRun, TrainingRunEpochMetric
from train_platform.repositories.training_run_repo import TrainingRunRepository
from train_platform.training.plugins.base import TrainContext
from train_platform.training.registry import get_trainer
from train_platform.utils.path_utils import resolve_dataset_path
from train_platform.utils.mlflow_utils import init_mlflow_logger
from train_platform.utils.training_artifacts import index_completion_artifacts as _index_completion_artifacts


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_HEARTBEAT_LOST_ERROR = "Worker heartbeat lost; marking as failed"


def _touch_run_liveness(db: Session, run: TrainingRun) -> None:
    """
    Keep the run from being marked stale when the queue worker is restarted/crashed.

    We update `heartbeat_at` from the training subprocess itself (best-effort) and also
    repair the common false FAILED status caused by worker heartbeat loss while training
    is still producing progress/metrics.
    """
    now = _utcnow()
    run.heartbeat_at = now

    # If a worker falsely marked the run as FAILED due to heartbeat loss, but we are still
    # actively training (we're inside the training subprocess), heal it back to RUNNING.
    if run.status == TrainingRunStatus.FAILED and (str(run.error_message or "").strip() == _HEARTBEAT_LOST_ERROR):
        run.status = TrainingRunStatus.RUNNING
        run.finished_at = None
        run.error_message = None


def _finalize_run_status(run_id: str, *, exit_code: int, error_message: str | None = None) -> None:
    """
    Best-effort terminal status update from the training subprocess.

    This prevents UI from getting stuck in FAILED (heartbeat lost) while training is still
    running, and also lets runs become COMPLETED even if the queue worker died.
    """
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
        if not run:
            return

        _touch_run_liveness(db, run)

        # Match the worker's priority: delete > cancel > exit_code.
        delete_requested = run.delete_requested_at is not None
        cancel_requested = (run.cancel_requested_at is not None) or delete_requested

        if delete_requested:
            run.status = TrainingRunStatus.DELETED
            run.hidden = True
            run.error_message = None
            run.finished_at = run.finished_at or _utcnow()
        elif cancel_requested:
            run.status = TrainingRunStatus.CANCELLED
            run.error_message = None
            run.finished_at = run.finished_at or _utcnow()
        elif int(exit_code) == 0:
            run.status = TrainingRunStatus.COMPLETED
            run.error_message = None
            run.finished_at = _utcnow()
            try:
                run.progress = max(int(getattr(run, "progress", 0) or 0), 100)
            except Exception:
                run.progress = 100
            try:
                _index_completion_artifacts(db, run_id)
            except Exception:
                pass
        else:
            run.status = TrainingRunStatus.FAILED
            run.finished_at = _utcnow()
            if error_message:
                run.error_message = str(error_message)

        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _cancel_requested(run_id: str) -> bool:
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
        if not run:
            return False
        _touch_run_liveness(db, run)
        db.commit()
        return bool(run.cancel_requested_at is not None or run.delete_requested_at is not None)
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def _upsert_epoch_metrics(run_id: str, epoch: int, metrics: Dict[str, float]) -> None:
    db = SessionLocal()
    try:
        row = (
            db.query(TrainingRunEpochMetric)
            .filter(TrainingRunEpochMetric.run_id == run_id, TrainingRunEpochMetric.epoch == int(epoch))
            .first()
        )
        if row:
            row.metrics = metrics
        else:
            db.add(TrainingRunEpochMetric(run_id=run_id, epoch=int(epoch), metrics=metrics))

        run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
        if run:
            _touch_run_liveness(db, run)
            run.current_epoch = int(epoch)
            if run.total_epochs and int(run.total_epochs) > 0:
                # Ultralytics epoch is 0-based; progress is best-effort.
                pct = int(min(100, max(0, 100 * float(epoch + 1) / float(run.total_epochs))))
                run.progress = pct
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run_id = str(args.run_id)

    db = SessionLocal()
    mlflow_logger = None
    mlflow_status = None
    exit_code = 1
    error_message: str | None = None
    try:
        run = TrainingRunRepository().get(db, run_id)
        if not run or not run.parameters or not run.project or not run.project.dataset or not run.dataset_version or not run.architecture:
            print(f"[train_entry] run not found or missing relations: {run_id}", file=sys.stderr, flush=True)
            exit_code = 2
            error_message = "Run not found or missing relations"
            return exit_code

        # Prefer snapshot_path if present (reproducibility). Fallback to dataset.storage_path.
        dataset_path_token = run.dataset_version.snapshot_path or run.project.dataset.storage_path
        dataset_path = resolve_dataset_path(dataset_path_token)
        if not dataset_path.exists():
            print(f"[train_entry] dataset path does not exist: {dataset_path}", file=sys.stderr, flush=True)
            exit_code = 2
            error_message = f"Dataset path does not exist: {dataset_path}"
            return exit_code

        if _cancel_requested(run_id):
            print(f"[train_entry] cancel requested before start run_id={run_id}", file=sys.stderr, flush=True)
            exit_code = 0
            return exit_code

        engine = str(getattr(run.architecture, "engine", "") or "")
        family = str(getattr(run.architecture, "family", "") or "")
        key = engine or family or "yolo"

        trainer = get_trainer(model_family=key)
        run_dir = settings.training_dir / run_id

        mlflow_logger = init_mlflow_logger(run, dataset_path=str(dataset_path), run_dir=str(run_dir))

        def upsert_epoch_metrics(epoch: int, metrics: Dict[str, float]) -> None:
            _upsert_epoch_metrics(run_id, epoch, metrics)
            if mlflow_logger:
                mlflow_logger.log_metrics(metrics, step=int(epoch))

        ctx = TrainContext(
            job_id=run_id,
            job=run,
            dataset_path=dataset_path,
            run_dir=run_dir,
            cancel_requested=lambda: _cancel_requested(run_id),
            upsert_epoch_metrics=upsert_epoch_metrics,
        )

        print(f"[train_entry] start run_id={run_id} trainer={getattr(trainer, 'name', type(trainer).__name__)}", flush=True)
        trainer.run(ctx)
        mlflow_status = "FINISHED"
        print(f"[train_entry] completed run_id={run_id}", flush=True)
        exit_code = 0
        return exit_code
    except KeyboardInterrupt:
        mlflow_status = "KILLED"
        print(f"[train_entry] interrupted run_id={run_id}", file=sys.stderr, flush=True)
        exit_code = 130
        error_message = "Interrupted"
        return exit_code
    except SystemExit as e:
        mlflow_status = "KILLED" if _cancel_requested(run_id) else "FAILED"
        try:
            exit_code = int(e.code or 0)
        except Exception:
            exit_code = 0
        if mlflow_status == "FAILED":
            error_message = "Exited"
        return exit_code
    except Exception as e:
        mlflow_status = "FAILED"
        print(f"[train_entry] error run_id={run_id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        exit_code = 1
        error_message = f"{type(e).__name__}: {e}"
        return exit_code
    finally:
        # Best-effort: ensure DB status does not incorrectly remain FAILED due to worker heartbeat loss.
        try:
            _finalize_run_status(run_id, exit_code=exit_code, error_message=error_message)
        except Exception:
            pass
        if mlflow_logger:
            mlflow_logger.terminate(status=mlflow_status or "FAILED")
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
