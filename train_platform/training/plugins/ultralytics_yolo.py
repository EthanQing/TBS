from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from train_platform.core.config import settings
from train_platform.training.plugins.base import TrainContext
from train_platform.utils.path_utils import resolve_pretrain_path, resolve_temp_path


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
    Ultralytics AMP check loads YOLO('yolov8n.pt') from CWD.
    Ensure the file exists locally when pretrain models are mounted elsewhere.
    """
    try:
        cwd = Path.cwd()
        dest = (cwd / "yolov8n.pt").resolve(strict=False)
        if dest.exists():
            return True

        src = (settings.pretrain_models_dir / "yolov8n.pt").resolve(strict=False)
        if not src.exists():
            return False

        try:
            os.symlink(src, dest)
        except Exception:
            try:
                shutil.copy2(src, dest)
            except Exception:
                return False

        return dest.exists()
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
    name = "ultralytics-yolo"

    def can_handle(self, model_family: str) -> bool:
        mf = (model_family or "").strip().lower()
        return "yolo" in mf

    def run(self, ctx: TrainContext) -> None:
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

        model_variant = ""
        if getattr(job, "architecture", None) is not None:
            model_variant = str(getattr(job.architecture, "variant", "") or "")
        model_variant = (model_variant or "yolov8n").strip()

        resume_training = _coerce_bool(add.get("resume_training", False), False)
        resume_job_id = add.get("resume_job_id")
        use_pretrained = _coerce_bool(
            add.get("use_pretrained", None),
            getattr(getattr(job, "parameters", None), "use_pretrained", True),
        )
        pretrained_model_path = add.get("pretrained_model_path")

        cleanup_candidate: Path | None = None

        if resume_training and resume_job_id:
            resume_weights_path = settings.training_dir / str(resume_job_id) / "weights" / "last.pt"
            if not resume_weights_path.exists():
                raise ValueError(f"resume weights not found: {resume_weights_path}")
            model_path = str(resume_weights_path)
        elif use_pretrained:
            resolved_pretrain = None
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
            else:
                official = resolve_pretrain_path(f"{model_variant}.pt")
                model_path = str(official) if official.exists() else f"{model_variant}.pt"
        else:
            model_path = f"{model_variant}.yaml"

        _apply_torch_safe_load_patches()
        _ensure_amp_check_weight()
        try:
            model = YOLO(model_path)

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

            data_yaml = ctx.dataset_path / "data.yaml"
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
                "lr0": float(job.parameters.learning_rate),
                "imgsz": int(job.parameters.image_size),
                "optimizer": str(job.parameters.optimizer or "auto"),
                "workers": int(getattr(job.parameters, "workers", 8) or 8),
                "patience": int(getattr(job.parameters, "patience", 50) or 50),
                "momentum": float(add.get("momentum", 0.937)),
                "weight_decay": float(add.get("weight_decay", 0.0005)),
                "warmup_epochs": float(add.get("warmup_epochs", 3.0)),
                "warmup_momentum": float(add.get("warmup_momentum", 0.8)),
                "warmup_bias_lr": float(add.get("warmup_bias_lr", 0.1)),
                "project": str(settings.training_dir),
                "name": ctx.job_id,
                "device": device_value,
                "exist_ok": True,
            }
            if resume_training and resume_job_id:
                train_args["resume"] = True

            model.train(**train_args)
        finally:
            if cleanup_candidate is not None:
                try:
                    if cleanup_candidate.exists() and settings.temp_dir.resolve() in cleanup_candidate.resolve().parents:
                        cleanup_candidate.unlink()
                except Exception:
                    pass
