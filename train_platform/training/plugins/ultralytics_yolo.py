from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from train_platform.core.config import settings
from train_platform.training.plugins.base import TrainContext
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.path_utils import resolve_pretrain_path, resolve_temp_path

logger = logging.getLogger("train_platform.training.ultralytics")


def _apply_torch_safe_load_patches() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.serialization
        from ultralytics.nn.modules import Bottleneck, BottleneckCSP, C2f, Conv, SPPF
        from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel

        safe_classes = [
            DetectionModel,
            SegmentationModel,
            ClassificationModel,
            nn.modules.container.Sequential,
            Conv,
            Bottleneck,
            BottleneckCSP,
            C2f,
            SPPF,
        ]
        torch.serialization.add_safe_globals(safe_classes)

        import ultralytics.nn.tasks

        def patched_torch_safe_load(weight):
            try:
                return torch.load(weight, map_location="cpu"), weight
            except Exception:
                return torch.load(weight, map_location="cpu", weights_only=False), weight

        ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
    except Exception:
        pass


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

    def run(self, ctx: TrainContext, *, config: Dict[str, Any] | None = None) -> None:
        def _coerce_bool(value, default: bool) -> bool:
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

        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyTorch not installed") from e
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Ultralytics not installed") from e

        job = ctx.job
        add = getattr(getattr(job, "parameters", None), "additional_params", None) or {}
        if isinstance(add.get("framework_config"), dict):
            add = {**add, **dict(add.get("framework_config") or {})}
        if config:
            add = {**add, **self.normalize_config(config)}

        model_variant = ""
        if getattr(job, "architecture", None) is not None:
            model_variant = str(getattr(job.architecture, "variant", "") or "")
        model_variant = (model_variant or "yolov8n").strip()
        model_variant_lower = model_variant.lower()
        is_rtdetr_variant = model_variant_lower.startswith("rtdetr")

        resume_training = _coerce_bool(add.get("resume_training", False), False)
        resume_job_id = add.get("resume_job_id")
        use_pretrained = _coerce_bool(
            add.get("use_pretrained", None),
            getattr(getattr(job, "parameters", None), "use_pretrained", True),
        )
        pretrained_model_path = add.get("pretrained_model_path")

        model_loader_cls = YOLO
        if is_rtdetr_variant:
            try:
                from ultralytics import RTDETR
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Ultralytics RT-DETR runtime not available") from e
            model_loader_cls = RTDETR

        cleanup_candidate: Path | None = None
        resolved_pretrain: Path | None = None
        model_path = ""
        model_load_mode = "yaml"

        if resume_training:
            # Case 1: Resume from another job (transfer learning / continue from checkpoint)
            if resume_job_id and str(resume_job_id) != str(ctx.job_id):
                resume_weights_path = settings.training_dir / str(resume_job_id) / "weights" / "last.pt"
                if not resume_weights_path.exists():
                    raise ValueError(f"resume weights not found: {resume_weights_path}")
                model_path = str(resume_weights_path)
                model_load_mode = "resume"
            
            # Case 2: Resume this same job (continue interrupted training)
            else:
                # For self-resume, we point to our own last.pt. 
                # Ultralytics handles 'resume=True' automatically if we pass the weights file 
                # or if we just set resume=True with the same project/name, but explicit path is safer.
                my_weights = ctx.run_dir / "weights" / "last.pt"
                if my_weights.exists():
                    model_path = str(my_weights)
                    model_load_mode = "resume"
                else:
                    # Fallback: if no weights yet, just start fresh or use pretrained
                    # This happens if user clicked 'Resume' on a job that failed before first save.
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
            ctx.job_id,
            model_variant,
            model_loader_cls.__name__,
            model_load_mode,
            resume_training,
            use_pretrained,
        )

        _apply_torch_safe_load_patches()
        amp_probe_ready = _ensure_amp_check_weight()
        try:
            model = model_loader_cls(model_path)

            # Disable Ultralytics' built-in MLflow integration to avoid URI conflicts
            # We handle MLflow logging through our own MlflowRunLogger in train_entry.py
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
                return bool(ctx.cancel_requested())

            def on_epoch_end(trainer):
                epoch = int(getattr(trainer, "epoch", 0))
                metrics = _collect_metrics(trainer)
                if metrics:
                    ctx.upsert_epoch_metrics(epoch, metrics)
                if should_cancel():
                    raise SystemExit(0)

            def on_batch_end(_trainer):
                if should_cancel():
                    raise SystemExit(0)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.add_callback("on_train_batch_end", on_batch_end)

            run_dir = ctx.run_dir
            run_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = None
            try:
                dataset_name = str(getattr(getattr(getattr(job, "project", None), "dataset", None), "name", None) or "") or None
            except Exception:
                dataset_name = None

            data_yaml = find_yolo_dataset_yaml(ctx.dataset_path, dataset_name=dataset_name)
            if data_yaml is None:
                raise ValueError(f"Dataset YAML not found under: {ctx.dataset_path}")
            run_data_yaml = data_yaml
            try:
                with open(data_yaml, "r", encoding="utf-8", errors="replace") as f:
                    data_cfg = yaml.safe_load(f) or {}
                if not isinstance(data_cfg, dict):
                    data_cfg = {}
                data_cfg.pop("path", None)
                data_cfg["path"] = str(ctx.dataset_path)
                run_data_yaml = run_dir / "data.runtime.yaml"
                with open(run_data_yaml, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)
            except Exception:
                run_data_yaml = data_yaml

            device_value = str(getattr(job.parameters, "device", "auto") or "auto").strip().lower()
            if device_value in ("", "default", "自动"):
                device_value = "auto"
            elif device_value == "gpu":
                device_value = "0"
            elif device_value.startswith("cuda:"):
                device_value = device_value.split("cuda:", 1)[1] or "0"
            if device_value not in ("auto", "cpu") and not torch.cuda.is_available():
                device_value = "cpu"

            train_args: Dict[str, Any] = {
                "data": str(run_data_yaml),
                "epochs": int(job.parameters.epochs),
                "batch": int(job.parameters.batch_size),
                "imgsz": int(job.parameters.image_size),
                "workers": int(getattr(job.parameters, "workers", 8) or 8),
                "project": str(settings.training_dir),
                "name": ctx.job_id,
                "device": device_value,
                "exist_ok": True,
                "save_period": int(add.get("save_period", -1)),
                "amp": _coerce_bool(add.get("amp", True), True),
            }

            if train_args["amp"] and not amp_probe_ready:
                train_args["amp"] = False
                logger.warning(
                    "AMP probe weights are not available; disable AMP for run_id=%s to avoid check_amp failure",
                    ctx.job_id,
                )

            if is_rtdetr_variant:
                # RT-DETR does not support some YOLO-only hyperparameters.
                train_args.update(
                    {
                        "lr0": float(job.parameters.learning_rate),
                        "optimizer": str(job.parameters.optimizer or "auto"),
                        "patience": int(getattr(job.parameters, "patience", 50) or 50),
                        "weight_decay": float(add.get("weight_decay", 0.0005)),
                    }
                )
            else:
                train_args.update(
                    {
                        "lr0": float(job.parameters.learning_rate),
                        "optimizer": str(job.parameters.optimizer or "auto"),
                        "patience": int(getattr(job.parameters, "patience", 50) or 50),
                        "momentum": float(add.get("momentum", 0.937)),
                        "weight_decay": float(add.get("weight_decay", 0.0005)),
                        "warmup_epochs": float(add.get("warmup_epochs", 3.0)),
                        "warmup_momentum": float(add.get("warmup_momentum", 0.8)),
                        "warmup_bias_lr": float(add.get("warmup_bias_lr", 0.1)),
                    }
                )

            if resume_training:
                train_args["resume"] = True
            elif resolved_pretrain is not None:
                train_args["pretrained"] = str(resolved_pretrain)
            else:
                train_args["pretrained"] = bool(use_pretrained)

            logger.info(
                "Training args prepared run_id=%s variant=%s keys=%s",
                ctx.job_id,
                model_variant,
                ",".join(sorted(train_args.keys())),
            )

            model.train(**train_args)
            
            # Run validation after training to ensure final metrics are captured and available
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
