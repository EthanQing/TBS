from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from train_platform.core.config import settings
from train_platform.platform.runtime.ultralytics import apply_torch_safe_load_patches
from .contract import TrainingCallbacks, TrainingExecutionSpec
from train_platform.domains.datasets.yolo import find_yolo_dataset_yaml
from train_platform.domains.training.execution_paths import (
    prepare_ultralytics_execution,
    prepare_ultralytics_resume_output,
    reset_ultralytics_output,
    resolve_ultralytics_resume_checkpoint,
    stage_ultralytics_input,
)
from train_platform.platform.filesystem.locations import resolve_pretrain_path, resolve_temp_path
from train_platform.domains.training.parameters import (
    AUTO_BATCH_SIZE,
    ULTRALYTICS_AUGMENTATION_SPEC_BY_KEY,
    ULTRALYTICS_LOSS_WEIGHT_SPEC_BY_KEY,
    extract_selected_gpu_ids,
    normalize_device_spec,
    normalize_lr_scheduler,
)

logger = logging.getLogger("train_platform.domains.training.frameworks.ultralytics")


def _lr_scheduler_to_ultralytics_args(value: Any) -> Dict[str, bool]:
    return {"cos_lr": normalize_lr_scheduler(value) == "cosine"}


def _coerce_bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(value)


def _wrap_build_dataloader_pin_memory(func: Any, pin_memory_enabled: bool) -> Any:
    if not callable(func):
        return func
    if (
        getattr(func, "_train_platform_pin_memory_wrapped", False)
        and getattr(func, "_train_platform_pin_memory_enabled", None) == bool(pin_memory_enabled)
    ):
        return func
    original = getattr(func, "_train_platform_original", func)

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["pin_memory"] = bool(pin_memory_enabled)
        return original(*args, **kwargs)

    wrapped._train_platform_pin_memory_wrapped = True
    wrapped._train_platform_pin_memory_enabled = bool(pin_memory_enabled)
    wrapped._train_platform_original = original
    return wrapped


def _patch_ultralytics_dataloader_pin_memory(pin_memory_enabled: bool) -> None:
    """
    Force Ultralytics dataloaders to use a stable pin_memory policy.

    PyTorch on Windows can fail in the pin-memory thread with
    `CUDA error: resource already mapped` during validation. Ultralytics 8.4
    does not expose pin_memory as a public train argument, so patch the imported
    builder functions in-process before `model.train()`.
    """
    module_names = (
        "ultralytics.data",
        "ultralytics.data.build",
        "ultralytics.models.yolo.detect.train",
        "ultralytics.models.yolo.detect.val",
        "ultralytics.models.yolo.segment.train",
        "ultralytics.models.yolo.segment.val",
        "ultralytics.models.yolo.classify.train",
        "ultralytics.models.yolo.classify.val",
    )
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["build_dataloader"])
        except Exception:
            continue
        if hasattr(module, "build_dataloader"):
            module.build_dataloader = _wrap_build_dataloader_pin_memory(
                getattr(module, "build_dataloader"),
                pin_memory_enabled,
            )


def _ensure_amp_check_weight() -> bool:
    """
    Ultralytics AMP self-check loads probe weights from CWD.

    On ultralytics 8.4.x this probe file is `yolo26n.pt` (hard-coded inside
    `ultralytics.utils.checks.check_amp`). In older versions it could be
    `yolov8n.pt`.
    """
    try:
        cwd = Path.cwd()
        probe_names = ("yolo26n.pt", "yolov8n.pt")
        sources = [
            (settings.pretrain_models_dir / "yolo26n.pt").resolve(strict=False),
            (settings.pretrain_models_dir / "yolo11n.pt").resolve(strict=False),
            (settings.pretrain_models_dir / "yolov8n.pt").resolve(strict=False),
        ]

        fallback = None
        for src in sources:
            if src.exists():
                fallback = src
                break
        if fallback is None:
            try:
                fallback = next(settings.pretrain_models_dir.glob("*.pt"))
            except Exception:
                fallback = None

        probe_ready = False
        for probe_name in probe_names:
            dest = (cwd / probe_name).resolve(strict=False)
            if dest.exists():
                probe_ready = True
                continue
            src = (settings.pretrain_models_dir / probe_name).resolve(strict=False)
            if not src.exists():
                src = fallback
            if src is None or not src.exists():
                continue
            try:
                os.symlink(src, dest)
            except Exception:
                try:
                    shutil.copy2(src, dest)
                except Exception:
                    continue
            probe_ready = probe_ready or dest.exists()
        return probe_ready
    except Exception:
        return False


def _collect_metrics(trainer: Any) -> Dict[str, float]:
    values: Dict[str, Any] = {}
    if isinstance(getattr(trainer, "metrics", None), dict):
        values.update(trainer.metrics)
    if getattr(trainer, "tloss", None) is not None and callable(getattr(trainer, "label_loss_items", None)):
        losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        if isinstance(losses, Mapping):
            values.update(losses)
    lr = getattr(trainer, "lr", None)
    if isinstance(lr, Mapping):
        values.update(lr)
    elif isinstance(getattr(trainer, "lrs", None), (list, tuple)):
        values.update({f"lr/pg{i}": v for i, v in enumerate(trainer.lrs)})
    result = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            pass
    return result


def _register_epoch_callbacks(model: Any, emit: Any, *, collect: Any = _collect_metrics) -> None:
    pending: dict[str, int | None] = {"epoch": None}

    def started(trainer: Any) -> None:
        pending["epoch"] = int(trainer.epoch)

    def finished(trainer: Any) -> None:
        # final_eval fires on_fit_epoch_end without starting a training epoch.
        epoch = pending["epoch"]
        if epoch is None:
            return
        pending["epoch"] = None
        metrics = collect(trainer)
        if metrics:
            emit(epoch, metrics)

    model.add_callback("on_train_epoch_start", started)
    model.add_callback("on_fit_epoch_end", finished)


def _cleanup_pretrained_file(path: Path) -> None:
    try:
        if settings.temp_dir.resolve() in path.resolve().parents:
            path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary pretrained weight %s", path, exc_info=True)


@dataclass(frozen=True)
class PreparedUltralyticsExecution:
    run_id: str
    run_root: str
    runtime_dir: str
    output_dir: str
    model_type: str
    model_path: str
    requested_device: str
    runtime_device: str
    cuda_visible_devices: str
    world_size: int
    pin_memory: bool
    amp: bool
    train_args: dict[str, Any]
    execution_owner: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class UltralyticsYOLOTrainer:
    plugin_id = name = "ultralytics-yolo"
    display_name = "Ultralytics YOLO"
    implemented = True

    def can_handle(self, model_family: str) -> bool:
        mf = (model_family or "").strip().lower()
        return ("yolo" in mf) or ("rtdetr" in mf) or ("rt-detr" in mf)

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "use_pretrained": {"type": "boolean", "default": True},
                "pretrained_model_path": {"type": "string"},
                "resume_training": {"type": "boolean", "default": False},
                "resume_job_id": {"type": "string"},
                "save_period": {"type": "integer", "default": -1, "minimum": -1},
                "amp": {"type": "boolean", "default": True},
                "momentum": {"type": "number", "default": 0.937},
                "weight_decay": {"type": "number", "default": 0.0005},
                "warmup_epochs": {"type": "number", "default": 3.0},
                "warmup_momentum": {"type": "number", "default": 0.8},
                "warmup_bias_lr": {"type": "number", "default": 0.1},
            },
            "additionalProperties": True,
        }

    def normalize_config(self, raw: Dict[str, Any] | None) -> Dict[str, Any]:
        return dict(raw or {})

    def _prepare(self, spec: TrainingExecutionSpec, cleanup: ExitStack) -> PreparedUltralyticsExecution:
        import torch
        from ultralytics import RTDETR, YOLO
        variant = (str(spec.variant or "") or "yolov8n").strip()
        model_type = "rtdetr" if variant.lower().startswith("rtdetr") else "yolo"
        loader = RTDETR if model_type == "rtdetr" else YOLO
        run_root = Path(spec.run_dir).resolve(strict=False)
        resume = bool(spec.resume_training)
        checkpoint_path = None
        resolved_pretrain: Path | None = None
        model_path = ""
        if resume:
            source_root = run_root
            if spec.resume_job_id and str(spec.resume_job_id) != str(spec.run_id):
                source_root = settings.training_dir / str(spec.resume_job_id)
            checkpoint_path = resolve_ultralytics_resume_checkpoint(source_root)
            if checkpoint_path is None:
                raise ValueError(f"resume weights not found for run: {spec.resume_job_id or spec.run_id}")
            if checkpoint_path:
                model_path = str(checkpoint_path.resolve())
        if not model_path and spec.use_pretrained:
            if spec.pretrained_model_path:
                direct = Path(spec.pretrained_model_path)
                if direct.exists():
                    resolved_pretrain = direct
                else:
                    temporary = resolve_temp_path(spec.pretrained_model_path)
                    if temporary.exists():
                        resolved_pretrain = temporary
                        cleanup.callback(_cleanup_pretrained_file, temporary)
                    else:
                        candidate = resolve_pretrain_path(spec.pretrained_model_path)
                        if candidate.exists():
                            resolved_pretrain = candidate
                if resolved_pretrain is None:
                    raise ValueError(f"pretrained weights not found: {spec.pretrained_model_path}")
                model_path = str(resolved_pretrain.resolve())
            else:
                official = resolve_pretrain_path(f"{variant}.pt")
                model_path = str(official.resolve()) if official.exists() else f"{variant}.pt"
        if not model_path:
            model_path = f"{variant}.yaml"

        logger.info(
            "Preparing Ultralytics run_id=%s model_type=%s model_path=%s resume=%s",
            spec.run_id, model_type, model_path, resume,
        )
        apply_torch_safe_load_patches()
        amp_probe = _ensure_amp_check_weight()
        metadata_model = loader(model_path)
        epoch = None
        train_results = None
        checkpoint = None
        if checkpoint_path:
            checkpoint = getattr(metadata_model, "ckpt", None)
            if (
                not isinstance(checkpoint, dict)
                or isinstance(checkpoint.get("epoch"), bool)
                or not isinstance(checkpoint.get("epoch"), int)
                or checkpoint["epoch"] < 0
            ):
                raise ValueError("resume checkpoint metadata is unavailable or invalid")
            epoch = int(checkpoint["epoch"])
            raw_results = checkpoint.get("train_results")
            if isinstance(raw_results, Mapping):
                train_results = {str(k): list(v) if isinstance(v, (list, tuple)) else v for k, v in raw_results.items()}
        loaded_path = getattr(metadata_model, "ckpt_path", None)
        if loaded_path and Path(str(loaded_path)).exists():
            model_path = str(Path(str(loaded_path)).resolve())
        module = getattr(metadata_model, "model", None)
        if callable(getattr(module, "cpu", None)):
            module.cpu()

        if checkpoint_path:
            paths = prepare_ultralytics_execution(
                run_root,
                mode="resume",
                resume_checkpoint=checkpoint_path,
                reset_state="ready",
            )
        else:
            model_file = Path(model_path)
            original_model = model_file.resolve(strict=False)
            if model_file.is_file():
                staged_model = stage_ultralytics_input(run_root, model_file)
                if staged_model != model_file.resolve():
                    model_path = str(staged_model)
            if resolved_pretrain is not None:
                staged_pretrain = (
                    Path(model_path)
                    if resolved_pretrain.resolve(strict=False) == original_model
                    else stage_ultralytics_input(run_root, resolved_pretrain)
                )
                if staged_pretrain != resolved_pretrain.resolve():
                    resolved_pretrain = staged_pretrain
            paths = reset_ultralytics_output(run_root)
        if checkpoint_path and epoch is not None:
            prepare_ultralytics_resume_output(
                checkpoint_path.parent.parent,
                paths.output_dir,
                checkpoint_epoch=epoch,
                train_results=train_results,
            )
        data_yaml = find_yolo_dataset_yaml(spec.dataset_path, dataset_name=spec.dataset_name)
        if data_yaml is None:
            raise ValueError(f"Dataset YAML not found under: {spec.dataset_path}")
        with data_yaml.open("r", encoding="utf-8", errors="replace") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            data = {}
        data.pop("path", None)
        data["path"] = str(Path(spec.dataset_path).resolve(strict=False))
        paths.data_runtime_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

        requested = normalize_device_spec(spec.requested_device or "auto")
        runtime = normalize_device_spec(spec.runtime_device or requested)
        gpu_ids = extract_selected_gpu_ids(runtime)
        world_size = len(gpu_ids)
        if gpu_ids and not torch.cuda.is_available():
            raise RuntimeError(f"GPU device(s) requested ({requested}) but CUDA is not available")
        if gpu_ids and any(index >= torch.cuda.device_count() for index in gpu_ids):
            raise RuntimeError(f"Requested local devices {runtime} but only {torch.cuda.device_count()} are visible")
        batch = int(spec.batch_size or 16)
        if world_size > 1 and batch == AUTO_BATCH_SIZE:
            raise RuntimeError("Ultralytics auto batch is not supported for multi-GPU training")
        if world_size > 1 and batch > 0 and batch % world_size:
            raise RuntimeError(f"batch_size ({batch}) must be divisible by GPU count ({world_size})")
        visible = os.getenv("CUDA_VISIBLE_DEVICES", runtime if gpu_ids else "")
        if world_size > 1:
            if len(extract_selected_gpu_ids(visible)) != world_size:
                raise ValueError("Frozen CUDA_VISIBLE_DEVICES does not match the assigned GPU count")
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        logger.info(
            "Ultralytics devices run_id=%s requested_device=%s CUDA_VISIBLE_DEVICES=%s runtime_device=%s",
            spec.run_id, requested, visible, runtime,
        )
        pin_memory = _coerce_bool_config(spec.framework_config.get("pin_memory"), os.name != "nt")
        amp = _coerce_bool_config(spec.framework_config.get("amp", True), True)
        if amp and not amp_probe:
            amp = False
            logger.warning("AMP probe unavailable; disabling AMP for run_id=%s", spec.run_id)
        logger.info("Ultralytics settings run_id=%s pin_memory=%s amp=%s", spec.run_id, pin_memory, amp)
        args: Dict[str, Any] = {
            "data": str(paths.data_runtime_yaml.resolve()),
            "epochs": int(spec.epochs),
            "batch": batch,
            "imgsz": int(spec.image_size),
            "workers": int(spec.workers or 8),
            "project": str(run_root),
            "name": "output",
            "save_dir": str(paths.output_dir),
            "device": visible if world_size > 1 else runtime,
            "exist_ok": True,
            "save_period": int(spec.framework_config.get("save_period", -1)),
            "amp": amp,
            "lr0": float(spec.learning_rate),
            **_lr_scheduler_to_ultralytics_args(spec.lr_scheduler),
            "optimizer": str(spec.optimizer or "auto"),
            "patience": int(spec.patience or 50),
            "weight_decay": float(spec.weight_decay if spec.weight_decay is not None else 0.0005),
        }
        if model_type == "yolo":
            args.update(
                momentum=float(spec.momentum if spec.momentum is not None else 0.937),
                warmup_epochs=float(spec.warmup_epochs if spec.warmup_epochs is not None else 3.0),
                warmup_momentum=float(spec.warmup_momentum if spec.warmup_momentum is not None else 0.8),
                warmup_bias_lr=float(spec.warmup_bias_lr if spec.warmup_bias_lr is not None else 0.1),
            )
        for source, allowed in (
            (spec.augmentation, ULTRALYTICS_AUGMENTATION_SPEC_BY_KEY),
            (spec.loss_weights, ULTRALYTICS_LOSS_WEIGHT_SPEC_BY_KEY),
        ):
            for key, value in source.items():
                key = str(key or "").strip()
                if key in allowed:
                    args[key] = value
        if resume:
            args["resume"] = True
        elif resolved_pretrain is not None:
            args["pretrained"] = str(resolved_pretrain.resolve())
        else:
            args["pretrained"] = bool(spec.use_pretrained)
        prepared = PreparedUltralyticsExecution(
            run_id=str(spec.run_id),
            run_root=str(run_root),
            runtime_dir=str(paths.runtime_dir),
            output_dir=str(paths.output_dir),
            model_type=model_type,
            model_path=model_path,
            requested_device=requested,
            runtime_device=runtime,
            cuda_visible_devices=str(visible),
            world_size=world_size,
            pin_memory=pin_memory,
            amp=amp,
            train_args=args,
            execution_owner=dict(spec.execution_owner),
        )
        del metadata_model, module, checkpoint
        gc.collect()
        return prepared

    def run(self, spec: TrainingExecutionSpec, callbacks: TrainingCallbacks) -> None:
        with ExitStack() as cleanup:
            prepared = self._prepare(spec, cleanup)
            if prepared.world_size > 1:
                from train_platform.platform.runtime.ultralytics_ddp import run_ultralytics_ddp

                run_ultralytics_ddp(
                    prepared.to_json(),
                    cancel_requested=callbacks.cancel_requested,
                    upsert_epoch_metrics=callbacks.upsert_epoch_metrics,
                )
                return

            from ultralytics import RTDETR, YOLO, settings as ultralytics_settings
            from .ultralytics_trainers import platform_trainer_for

            ultralytics_settings.update({"mlflow": False})
            _patch_ultralytics_dataloader_pin_memory(prepared.pin_memory)
            model = (RTDETR if prepared.model_type == "rtdetr" else YOLO)(prepared.model_path)
            _register_epoch_callbacks(model, callbacks.upsert_epoch_metrics)
            last_check = 0.0

            def cancel_on_batch(_trainer: Any) -> None:
                nonlocal last_check
                now = time.monotonic()
                if now - last_check >= 2.0:
                    last_check = now
                    if callbacks.cancel_requested():
                        raise SystemExit(0)

            model.add_callback("on_train_batch_end", cancel_on_batch)
            model.train(trainer=platform_trainer_for(model._smart_load("trainer")), **prepared.train_args)
            try:
                model.val(
                    data=prepared.train_args["data"],
                    project=prepared.run_root,
                    name="output",
                    save_dir=prepared.output_dir,
                    exist_ok=True,
                )
            except Exception:
                pass


__all__ = ["PreparedUltralyticsExecution", "UltralyticsYOLOTrainer"]
