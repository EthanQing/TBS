from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
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
from train_platform.domains.training.frameworks import (
    TrainingCallbacks,
    TrainingExecutionSpec,
    TrainerPlugin,
    get_trainer,
)
from train_platform.repositories.v3.training_run_repo import TrainingRunRepository
from train_platform.models.v3.training_run import TrainingRun
from train_platform.services.v3.alarm_service import AlarmService
from train_platform.utils.mlflow_utils import init_mlflow_logger
from train_platform.utils.training_params import build_device_runtime, parse_visible_host_gpu_ids
from train_platform.workers.training.vdl_bridge import VisualDLScalarBridge


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _materialize_execution_spec(
    run: TrainingRun,
    *,
    dataset_path: Path,
    run_dir: Path,
    requested_device: str,
    runtime_device: str,
    trainer: TrainerPlugin,
) -> TrainingExecutionSpec:
    parameters = run.parameters
    architecture = run.architecture
    additional_params = getattr(parameters, "additional_params", None)
    additional = dict(additional_params) if isinstance(additional_params, dict) else {}

    raw_framework_config = additional.get("framework_config")
    normalized_framework_config = (
        trainer.normalize_config(raw_framework_config)
        if isinstance(raw_framework_config, dict)
        else {}
    )
    if not isinstance(normalized_framework_config, dict):
        normalized_framework_config = {}

    # The nested framework config wins over legacy top-level values, matching
    # the persisted API behavior while exposing one explicit spec source.
    effective = {key: value for key, value in additional.items() if key != "framework_config"}
    effective.update(normalized_framework_config)
    engine = str(getattr(architecture, "engine", "") or "").strip().lower()
    if engine == "ultralytics-yolo":
        framework_keys = ("amp", "save_period", "pin_memory")
    elif engine == "paddle-det":
        framework_keys = (
            "config_path",
            "metrics_source",
            "eval_during_train",
            "eval_interval",
            "snapshot_epoch",
            "save_period",
        )
    else:
        framework_keys = tuple(normalized_framework_config.keys())
    framework_config = {key: effective[key] for key in framework_keys if key in effective}
    if engine == "paddle-det" and not framework_config.get("config_path"):
        defaults = getattr(architecture, "default_params", None)
        if isinstance(defaults, dict) and defaults.get("config_path"):
            framework_config["config_path"] = defaults["config_path"]

    resume_training = _coerce_bool(effective.get("resume_training"), False)
    resume_job_id = effective.get("resume_job_id")
    resume_job_id = str(resume_job_id).strip() if resume_job_id is not None else None
    if not resume_job_id:
        resume_job_id = None
    pretrained_model_path = effective.get("pretrained_model_path") or getattr(
        architecture, "pretrained_path", None
    )

    return TrainingExecutionSpec(
        run_id=str(run.run_id),
        dataset_path=dataset_path,
        dataset_name=str(getattr(run.standard_dataset, "name", "") or "") or None,
        run_dir=run_dir,
        engine=engine,
        family=str(getattr(architecture, "family", "") or ""),
        variant=str(getattr(architecture, "variant", "") or ""),
        epochs=int(getattr(parameters, "epochs", 100) or 100),
        batch_size=int(getattr(parameters, "batch_size", 16) or 16),
        image_size=int(getattr(parameters, "image_size", 640) or 640),
        learning_rate=float(getattr(parameters, "learning_rate", 0.01) or 0.01),
        lr_scheduler=str(getattr(parameters, "lr_scheduler", "linear") or "linear"),
        patience=int(getattr(parameters, "patience", 50) or 50),
        requested_device=requested_device,
        runtime_device=runtime_device,
        workers=int(getattr(parameters, "workers", 8) or 8),
        optimizer=str(getattr(parameters, "optimizer", "AdamW") or "AdamW"),
        use_pretrained=_coerce_bool(
            effective.get("use_pretrained"),
            bool(getattr(parameters, "use_pretrained", True)),
        ),
        augmentation=getattr(parameters, "augmentation", None) or {},
        loss_weights=getattr(parameters, "loss_weights", None) or {},
        resume_training=resume_training,
        resume_job_id=resume_job_id,
        pretrained_model_path=str(pretrained_model_path) if pretrained_model_path else None,
        momentum=_optional_float(effective.get("momentum")),
        weight_decay=_optional_float(effective.get("weight_decay")),
        warmup_epochs=_optional_float(effective.get("warmup_epochs")),
        warmup_momentum=_optional_float(effective.get("warmup_momentum")),
        warmup_bias_lr=_optional_float(effective.get("warmup_bias_lr")),
        framework_config=framework_config,
    )


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
        if not run or not run.parameters or not run.standard_dataset or not run.architecture:
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
        spec = _materialize_execution_spec(
            run,
            dataset_path=dataset_path,
            run_dir=run_dir,
            requested_device=requested_device,
            runtime_device=runtime_device,
            trainer=trainer,
        )

        mlflow_logger = init_mlflow_logger(run, dataset_path=str(dataset_path), run_dir=str(run_dir))

        def upsert_epoch_metrics(epoch: int, metrics: Dict[str, float]) -> None:
            persist_epoch_metrics(run_id, epoch, metrics, expected_pid=execution_pid)
            if mlflow_logger:
                mlflow_logger.log_metrics(metrics, step=int(epoch))

        callbacks = TrainingCallbacks(
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
        metrics_source = str(spec.framework_config.get("metrics_source") or "callback").strip().lower()
        if str(engine or "").strip().lower() == "paddle-det" and metrics_source == "hybrid":
            vdl_bridge = VisualDLScalarBridge(
                run_id=run_id,
                run_dir=run_dir,
                upsert_epoch_metrics=upsert_epoch_metrics,
                poll_interval_sec=5.0,
            )
            vdl_bridge.start()

        print(f"[train_entry] start run_id={run_id} trainer={getattr(trainer, 'name', type(trainer).__name__)}", flush=True)
        trainer.run(spec, callbacks)
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
