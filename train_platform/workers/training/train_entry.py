from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.training_run import TrainingRun, TrainingRunEpochMetric
from train_platform.repositories.training_run_repo import TrainingRunRepository
from train_platform.training.plugins.base import TrainContext
from train_platform.training.registry import get_trainer
from train_platform.utils.path_utils import resolve_dataset_path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cancel_requested(run_id: str) -> bool:
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
        return bool(run and (run.cancel_requested_at is not None or run.delete_requested_at is not None))
    except Exception:
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
    try:
        run = TrainingRunRepository().get(db, run_id)
        if not run or not run.parameters or not run.project or not run.project.dataset or not run.dataset_version or not run.architecture:
            print(f"[train_entry] run not found or missing relations: {run_id}", file=sys.stderr, flush=True)
            return 2

        # Prefer snapshot_path if present (reproducibility). Fallback to dataset.storage_path.
        dataset_path_token = run.dataset_version.snapshot_path or run.project.dataset.storage_path
        dataset_path = resolve_dataset_path(dataset_path_token)
        if not dataset_path.exists():
            print(f"[train_entry] dataset path does not exist: {dataset_path}", file=sys.stderr, flush=True)
            return 2

        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            print(f"[train_entry] data.yaml not found: {data_yaml}", file=sys.stderr, flush=True)
            return 2

        if _cancel_requested(run_id):
            print(f"[train_entry] cancel requested before start run_id={run_id}", file=sys.stderr, flush=True)
            return 0

        engine = str(getattr(run.architecture, "engine", "") or "")
        family = str(getattr(run.architecture, "family", "") or "")
        key = engine or family or "yolo"

        trainer = get_trainer(model_family=key)
        run_dir = settings.training_dir / run_id

        ctx = TrainContext(
            job_id=run_id,
            job=run,
            dataset_path=dataset_path,
            run_dir=run_dir,
            cancel_requested=lambda: _cancel_requested(run_id),
            upsert_epoch_metrics=lambda epoch, metrics: _upsert_epoch_metrics(run_id, epoch, metrics),
        )

        print(f"[train_entry] start run_id={run_id} trainer={getattr(trainer, 'name', type(trainer).__name__)}", flush=True)
        trainer.run(ctx)
        print(f"[train_entry] completed run_id={run_id}", flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"[train_entry] interrupted run_id={run_id}", file=sys.stderr, flush=True)
        return 130
    except SystemExit as e:
        try:
            return int(e.code or 0)
        except Exception:
            return 0
    except Exception as e:
        print(f"[train_entry] error run_id={run_id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

