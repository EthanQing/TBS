from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from typing import Dict

from train_platform.core.config import settings
from train_platform.domains.datasets.storage.paths import resolve_legacy_dataset_path
from train_platform.core.license import assert_valid_license
from train_platform.db.session import SessionLocal
from train_platform.domains.training.runs import (
    finalize_execution,
    touch_heartbeat,
    upsert_epoch_metrics as persist_epoch_metrics,
)
from train_platform.repositories.v3.training_run_repo import TrainingRunRepository
from train_platform.models.v3.training_run import TrainingRun
from train_platform.services.v3.alarm_service import AlarmService
from train_platform.training.plugins.base import TrainContext
from train_platform.training.registry import get_trainer
from train_platform.utils.mlflow_utils import init_mlflow_logger
from train_platform.utils.training_params import build_device_runtime, parse_visible_host_gpu_ids
from train_platform.workers.training.vdl_bridge import VisualDLScalarBridge


def _cancel_requested(run_id: str) -> bool:
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
        if not run:
            return False
        return bool(run.cancel_requested_at is not None or run.delete_requested_at is not None)
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def _heartbeat_tick(run_id: str, *, expected_pid: int) -> None:
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
        if not run:
            return
        if touch_heartbeat(db, run_id, expected_pid=expected_pid):
            AlarmService.try_evaluate_training_rules(db, run_ids=[str(run_id)])
    except Exception:
        db.rollback()
    finally:
        db.close()


def _heartbeat_loop(
    run_id: str,
    stop_event: threading.Event,
    *,
    expected_pid: int,
    interval_sec: float = 5.0,
) -> None:
    while not stop_event.wait(max(1.0, float(interval_sec))):
        _heartbeat_tick(run_id, expected_pid=expected_pid)


def main(argv: list[str] | None = None) -> int:
    assert_valid_license()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run_id = str(args.run_id)
    execution_pid = os.getpid()

    db = SessionLocal()
    mlflow_logger = None
    mlflow_status = None
    exit_code = 1
    error_message: str | None = None
    heartbeat_stop: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    vdl_bridge: VisualDLScalarBridge | None = None
    try:
        run = TrainingRunRepository().get(db, run_id)
        if not run or not run.parameters or not run.project or not run.project.standard_dataset or not run.architecture:
            print(f"[train_entry] run not found or missing relations: {run_id}", file=sys.stderr, flush=True)
            exit_code = 2
            error_message = "Run not found or missing relations"
            return exit_code

        dataset_path_token = run.standard_dataset.storage_path
        dataset_path = resolve_legacy_dataset_path(dataset_path_token)
        if not dataset_path.exists():
            print(f"[train_entry] dataset path does not exist: {dataset_path}", file=sys.stderr, flush=True)
            exit_code = 2
            error_message = f"Dataset path does not exist: {dataset_path}"
            return exit_code

        if _cancel_requested(run_id):
            print(f"[train_entry] cancel requested before start run_id={run_id}", file=sys.stderr, flush=True)
            exit_code = 0
            return exit_code

        visible_host_gpu_ids = parse_visible_host_gpu_ids()
        device_runtime = build_device_runtime(
            getattr(run.parameters, "device", "auto") or "auto",
            visible_host_gpu_ids=visible_host_gpu_ids,
        )
        requested_device = str(device_runtime.get("requested") or "auto")
        runtime_device = str(device_runtime.get("runtime_device") or requested_device)
        visible_devices = device_runtime.get("cuda_visible_devices")
        if visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        os.environ["TRAIN_PLATFORM_DEVICE_REQUEST"] = requested_device
        os.environ["TRAIN_PLATFORM_DEVICE_RUNTIME"] = runtime_device
        print(
            "[train_entry] device binding "
            f"run_id={run_id} "
            f"requested={requested_device} "
            f"runtime={runtime_device} "
            f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '<inherit>')} "
            f"nvidia_visible_devices={os.getenv('NVIDIA_VISIBLE_DEVICES', '<inherit>')}",
            flush=True,
        )

        engine = str(getattr(run.architecture, "engine", "") or "")
        family = str(getattr(run.architecture, "family", "") or "")
        trainer = get_trainer(
            model_family=(family or engine or "yolo"),
            engine=(engine or None),
        )
        run_dir = settings.training_dir / run_id

        mlflow_logger = init_mlflow_logger(run, dataset_path=str(dataset_path), run_dir=str(run_dir))

        def upsert_epoch_metrics(epoch: int, metrics: Dict[str, float]) -> None:
            persist_epoch_metrics(run_id, epoch, metrics, expected_pid=execution_pid)
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

        # Keep heartbeat alive even if plugin callbacks fail to report metrics.
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(run_id, heartbeat_stop),
            kwargs={"expected_pid": execution_pid, "interval_sec": 5.0},
            daemon=True,
        )
        heartbeat_thread.start()

        # Optional phase-2 bridge: enrich Paddle metrics from VisualDL scalars.
        additional_params = getattr(run.parameters, "additional_params", None) or {}
        framework_config_raw = additional_params.get("framework_config")
        plugin_config = trainer.normalize_config(framework_config_raw) if isinstance(framework_config_raw, dict) else {}
        metrics_source = str(
            plugin_config.get("metrics_source")
            or additional_params.get("metrics_source")
            or "callback"
        ).strip().lower()
        if str(engine or "").strip().lower() == "paddle-det" and metrics_source == "hybrid":
            vdl_bridge = VisualDLScalarBridge(
                run_id=run_id,
                run_dir=run_dir,
                upsert_epoch_metrics=upsert_epoch_metrics,
                poll_interval_sec=5.0,
            )
            vdl_bridge.start()

        print(f"[train_entry] start run_id={run_id} trainer={getattr(trainer, 'name', type(trainer).__name__)}", flush=True)
        trainer.run(ctx, config=plugin_config)
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
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)
        if vdl_bridge is not None:
            vdl_bridge.stop()
        try:
            lifecycle_db = SessionLocal()
            try:
                result = finalize_execution(
                    lifecycle_db,
                    run_id,
                    exit_code=exit_code,
                    expected_pid=execution_pid,
                    error_message=error_message,
                )
                if result.changed:
                    AlarmService.try_evaluate_training_rules(lifecycle_db, run_ids=[str(result.run_id)])
            finally:
                lifecycle_db.close()
        except Exception:
            pass
        if mlflow_logger:
            mlflow_logger.terminate(status=mlflow_status or "FAILED")
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
