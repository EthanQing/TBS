from __future__ import annotations

import logging
import os
import shutil
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict

import yaml

from train_platform.core.config import settings
from train_platform.platform.runtime.ultralytics import apply_torch_safe_load_patches
from .contract import TrainingCallbacks, TrainingExecutionSpec
from train_platform.domains.datasets.yolo import find_yolo_dataset_yaml
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
    out: Dict[str, Any] = {}
    if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
        out.update(trainer.metrics)
    lrs = getattr(trainer, "lrs", [])
    if isinstance(lrs, (list, tuple)):
        if len(lrs) > 0:
            out["lr/pg0"] = lrs[0]
        if len(lrs) > 1:
            out["lr/pg1"] = lrs[1]
        if len(lrs) > 2:
            out["lr/pg2"] = lrs[2]

    cleaned: Dict[str, float] = {}
    for k, v in out.items():
        try:
            cleaned[str(k)] = float(v)
        except Exception:
            continue
    return cleaned


class UltralyticsYOLOTrainer:
    plugin_id = "ultralytics-yolo"
    name = "ultralytics-yolo"
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

    def run(self, spec: TrainingExecutionSpec, callbacks: TrainingCallbacks) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch not installed") from exc
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Ultralytics not installed") from exc

        model_variant = (str(spec.variant or "") or "yolov8n").strip()
        model_variant_lower = model_variant.lower()
        is_rtdetr_variant = model_variant_lower.startswith("rtdetr")
        framework_config = dict(spec.framework_config)

        resume_training = bool(spec.resume_training)
        resume_job_id = spec.resume_job_id
        use_pretrained = bool(spec.use_pretrained)
        pretrained_model_path = spec.pretrained_model_path

        model_loader_cls = YOLO
        if is_rtdetr_variant:
            try:
                from ultralytics import RTDETR
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Ultralytics RT-DETR runtime not available") from exc
            model_loader_cls = RTDETR

        cleanup_candidate: Path | None = None
        resolved_pretrain: Path | None = None
        model_path = ""
        model_load_mode = "yaml"

        if resume_training:
            if resume_job_id and str(resume_job_id) != str(spec.run_id):
                resume_weights_path = settings.training_dir / str(resume_job_id) / "weights" / "last.pt"
                if not resume_weights_path.exists():
                    raise ValueError(f"resume weights not found: {resume_weights_path}")
                model_path = str(resume_weights_path)
                model_load_mode = "resume"
            else:
                my_weights = spec.run_dir / "weights" / "last.pt"
                if my_weights.exists():
                    model_path = str(my_weights)
                    model_load_mode = "resume"
                else:
                    resume_training = False

        if not model_path and use_pretrained:
            if pretrained_model_path:
                direct = Path(str(pretrained_model_path))
                if direct.exists():
                    resolved_pretrain = direct
                else:
                    candidate = resolve_temp_path(str(pretrained_model_path))
                    if candidate.exists():
                        resolved_pretrain = candidate
                        if settings.temp_dir.resolve() in candidate.resolve().parents:
                            cleanup_candidate = candidate
                    else:
                        candidate = resolve_pretrain_path(str(pretrained_model_path))
                        if candidate.exists():
                            resolved_pretrain = candidate

                if resolved_pretrain is None:
                    raise ValueError(f"pretrained weights not found: {pretrained_model_path}")

                model_path = str(resolved_pretrain)
                model_load_mode = "pt"
            else:
                official = resolve_pretrain_path(f"{model_variant}.pt")
                model_path = str(official) if official.exists() else f"{model_variant}.pt"
                model_load_mode = "pt"
        if not model_path:
            model_path = f"{model_variant}.yaml"
            model_load_mode = "yaml"

        logger.info(
            "Preparing training run_id=%s variant=%s loader=%s mode=%s resume=%s use_pretrained=%s",
            spec.run_id,
            model_variant,
            model_loader_cls.__name__,
            model_load_mode,
            resume_training,
            use_pretrained,
        )

        apply_torch_safe_load_patches()
        amp_probe_ready = _ensure_amp_check_weight()
        try:
            model = model_loader_cls(model_path)

            try:
                from ultralytics import settings as ultralytics_settings
                ultralytics_settings.update({"mlflow": False})
            except Exception:
                pass

            last_cancel_check = {"t": 0.0}

            def should_cancel() -> bool:
                now = time.time()
                if now - last_cancel_check["t"] < 2.0:
                    return False
                last_cancel_check["t"] = now
                return bool(callbacks.cancel_requested())

            def on_epoch_end(trainer):
                epoch = int(getattr(trainer, "epoch", 0))
                metrics = _collect_metrics(trainer)
                if metrics:
                    callbacks.upsert_epoch_metrics(epoch, metrics)
                if should_cancel():
                    raise SystemExit(0)

            def on_batch_end(_trainer):
                if should_cancel():
                    raise SystemExit(0)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.add_callback("on_train_batch_end", on_batch_end)

            run_dir = spec.run_dir
            run_dir.mkdir(parents=True, exist_ok=True)

            data_yaml = find_yolo_dataset_yaml(spec.dataset_path, dataset_name=spec.dataset_name)
            if data_yaml is None:
                raise ValueError(f"Dataset YAML not found under: {spec.dataset_path}")
            run_data_yaml = data_yaml
            try:
                with open(data_yaml, "r", encoding="utf-8", errors="replace") as file:
                    data_cfg = yaml.safe_load(file) or {}
                if not isinstance(data_cfg, dict):
                    data_cfg = {}
                data_cfg.pop("path", None)
                data_cfg["path"] = str(spec.dataset_path)
                run_data_yaml = run_dir / "data.runtime.yaml"
                with open(run_data_yaml, "w", encoding="utf-8") as file:
                    yaml.safe_dump(data_cfg, file, allow_unicode=True, sort_keys=False)
            except Exception:
                run_data_yaml = data_yaml

            batch_size = int(spec.batch_size or 16)
            requested_device_value = normalize_device_spec(spec.requested_device or "auto")
            device_value = normalize_device_spec(spec.runtime_device or requested_device_value)
            selected_gpu_ids = extract_selected_gpu_ids(device_value)
            multi_gpu = len(selected_gpu_ids) > 1

            logger.info(
                "Resolved Ultralytics device run_id=%s requested=%s runtime=%s visible=%s",
                spec.run_id,
                requested_device_value,
                device_value,
                os.getenv("CUDA_VISIBLE_DEVICES", "<inherit>"),
            )

            if selected_gpu_ids:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"GPU device(s) requested ({requested_device_value}) but CUDA is not available on this worker"
                    )
                available_gpu_count = int(torch.cuda.device_count())
                missing_gpu_ids = [idx for idx in selected_gpu_ids if idx >= available_gpu_count]
                if missing_gpu_ids:
                    raise RuntimeError(
                        f"Requested GPU device(s) {requested_device_value} but this training process only has "
                        f"{available_gpu_count} visible CUDA device(s) after runtime binding"
                    )

            if multi_gpu and batch_size == AUTO_BATCH_SIZE:
                raise RuntimeError("Ultralytics auto batch (batch_size=-1) is not supported for multi-GPU training")
            if multi_gpu and batch_size > 0 and batch_size % len(selected_gpu_ids) != 0:
                raise RuntimeError(
                    f"Ultralytics multi-GPU batch_size ({batch_size}) must be divisible by the "
                    f"selected GPU count ({len(selected_gpu_ids)})"
                )

            pin_memory_default = os.name != "nt"
            pin_memory_enabled = _coerce_bool_config(framework_config.get("pin_memory"), pin_memory_default)
            _patch_ultralytics_dataloader_pin_memory(pin_memory_enabled)
            logger.info(
                "Ultralytics dataloader pin_memory=%s run_id=%s default=%s",
                pin_memory_enabled,
                spec.run_id,
                pin_memory_default,
            )

            train_args: Dict[str, Any] = {
                "data": str(run_data_yaml),
                "epochs": int(spec.epochs),
                "batch": batch_size,
                "imgsz": int(spec.image_size),
                "workers": int(spec.workers or 8),
                "project": str(settings.training_dir),
                "name": spec.run_id,
                "device": device_value,
                "exist_ok": True,
                "save_period": int(framework_config.get("save_period", -1)),
                "amp": _coerce_bool_config(framework_config.get("amp", True), True),
            }

            if train_args["amp"] and not amp_probe_ready:
                train_args["amp"] = False
                logger.warning(
                    "AMP probe weights are not available; disable AMP for run_id=%s to avoid check_amp failure",
                    spec.run_id,
                )

            if is_rtdetr_variant:
                train_args.update(
                    {
                        "lr0": float(spec.learning_rate),
                        **_lr_scheduler_to_ultralytics_args(spec.lr_scheduler),
                        "optimizer": str(spec.optimizer or "auto"),
                        "patience": int(spec.patience or 50),
                        "weight_decay": float(spec.weight_decay if spec.weight_decay is not None else 0.0005),
                    }
                )
            else:
                train_args.update(
                    {
                        "lr0": float(spec.learning_rate),
                        **_lr_scheduler_to_ultralytics_args(spec.lr_scheduler),
                        "optimizer": str(spec.optimizer or "auto"),
                        "patience": int(spec.patience or 50),
                        "momentum": float(spec.momentum if spec.momentum is not None else 0.937),
                        "weight_decay": float(spec.weight_decay if spec.weight_decay is not None else 0.0005),
                        "warmup_epochs": float(spec.warmup_epochs if spec.warmup_epochs is not None else 3.0),
                        "warmup_momentum": float(spec.warmup_momentum if spec.warmup_momentum is not None else 0.8),
                        "warmup_bias_lr": float(spec.warmup_bias_lr if spec.warmup_bias_lr is not None else 0.1),
                    }
                )

            for key, value in spec.augmentation.items():
                key_s = str(key or "").strip()
                if key_s in ULTRALYTICS_AUGMENTATION_SPEC_BY_KEY:
                    train_args[key_s] = value

            for key, value in spec.loss_weights.items():
                key_s = str(key or "").strip()
                if key_s in ULTRALYTICS_LOSS_WEIGHT_SPEC_BY_KEY:
                    train_args[key_s] = value

            if resume_training:
                train_args["resume"] = True
            elif resolved_pretrain is not None:
                train_args["pretrained"] = str(resolved_pretrain)
            else:
                train_args["pretrained"] = bool(use_pretrained)

            logger.info(
                "Training args prepared run_id=%s variant=%s keys=%s",
                spec.run_id,
                model_variant,
                ",".join(sorted(train_args.keys())),
            )

            model.train(**train_args)
            try:
                model.val()
            except Exception:
                pass
        finally:
            if cleanup_candidate is not None:
                try:
                    if cleanup_candidate.exists() and settings.temp_dir.resolve() in cleanup_candidate.resolve().parents:
                        cleanup_candidate.unlink()
                except Exception:
                    pass


__all__ = ["UltralyticsYOLOTrainer"]
