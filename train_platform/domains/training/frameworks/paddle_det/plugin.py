from __future__ import annotations

"""
PaddleDetection training plugin.

The execution adapter materializes ORM state into TrainingExecutionSpec.
This module owns only the framework's runtime and training flow; dataset
conversion, configuration mapping, and runtime compatibility patches live in
the neighboring cohesive modules.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from train_platform.core.config import settings
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.paddledet_paths import (
    ensure_paddledet_repo_on_syspath,
    paddledet_missing_message,
    resolve_paddledet_config_path,
)
from train_platform.utils.path_utils import resolve_pretrain_path, resolve_temp_path
from train_platform.utils.training_params import extract_selected_gpu_ids, normalize_device_spec

from ..contract import TrainingCallbacks, TrainingExecutionSpec
from .config import (
    DEFAULT_CONFIGS,
    apply_cfg_overrides,
    apply_lr_scheduler_to_cfg,
    apply_warmup_epochs_to_cfg,
    bind_ppdet_dataset_cfg,
)
from .dataset import (
    build_coco_from_yolo_list,
    load_yaml,
    normalize_yolo_names,
    read_image_list,
    summarize_ppdet_dataset_cfg,
)
from .runtime import (
    apply_download_patches,
    coerce_metric_scalar,
    patch_ppdet_assigner_label_dtype,
    patch_ppdet_training_stats,
    rebind_trainer_datasets,
    restore_download_patches,
)

def _coerce_bool(value: Any, default: bool) -> bool:
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


def _safe_float(value: Any) -> float | None:
    scalar = coerce_metric_scalar(value)
    if scalar is not None:
        return scalar
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _apply_metric_aliases(raw_metrics: Dict[str, Any] | None) -> Dict[str, float]:
    raw = raw_metrics if isinstance(raw_metrics, dict) else {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        scalar = _safe_float(value)
        if scalar is not None:
            out[str(key)] = scalar

    def _alias(destination: str, *sources: str) -> None:
        if _safe_float(out.get(destination)) is not None:
            return
        for source in sources:
            scalar = _safe_float(out.get(source))
            if scalar is not None:
                out[destination] = scalar
                return

    _alias("metrics/mAP50(B)", "AP50", "mAP50", "eval/bbox_AP50", "eval/bbox_ap50")
    _alias("metrics/mAP50-95(B)", "mAP", "eval/bbox_mAP", "eval/bbox_map")
    _alias("metrics/precision(B)", "precision", "eval/bbox_precision", "eval/precision")
    _alias("metrics/recall(B)", "recall", "eval/bbox_recall", "eval/recall")
    _alias("AP50", "metrics/mAP50(B)")
    _alias("mAP50", "metrics/mAP50(B)")
    _alias("mAP", "metrics/mAP50-95(B)")
    _alias("precision", "metrics/precision(B)")
    _alias("recall", "metrics/recall(B)")
    return out

def _pick_paddle_checkpoint(work_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (best, last) checkpoint paths from PaddleDetection output directory."""
    best: Path | None = None
    last: Path | None = None

    # PaddleDetection saves: output/<model_name>/best_model/  and  model_final.*
    try:
        # Best model
        best_dirs = list(work_dir.rglob("best_model"))
        for bd in best_dirs:
            pdparams = list(bd.glob("*.pdparams"))
            if pdparams:
                best = max(pdparams, key=lambda p: p.stat().st_mtime)
                break
    except Exception:
        pass

    try:
        # Last / final model
        final_files = list(work_dir.rglob("model_final.pdparams"))
        if final_files:
            last = max(final_files, key=lambda p: p.stat().st_mtime)
    except Exception:
        pass

    # Fallback: any .pdparams
    if last is None:
        try:
            all_pd = [p for p in work_dir.rglob("*.pdparams") if p.is_file()]
            if all_pd:
                last = max(all_pd, key=lambda p: p.stat().st_mtime)
        except Exception:
            pass

    if best is None:
        best = last

    return best, last


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class PaddleDetTrainer:
    """PaddleDetection training plugin.

    Supports PP-YOLOE(+), PicoDet and other PaddleDetection architectures.
    """

    plugin_id = "paddle-det"
    name = "paddle-det"
    display_name = "PaddleDetection"
    implemented = True

    def can_handle(self, model_family: str) -> bool:
        mf = (model_family or "").strip().lower()
        return any(kw in mf for kw in ("paddle", "ppyolo", "ppdet", "picodet"))

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "config_path": {"type": "string"},
                "resume_training": {"type": "boolean", "default": False},
                "resume_job_id": {"type": "string"},
                "use_pretrained": {"type": "boolean", "default": True},
                "pretrained_model_path": {"type": "string"},
                "metrics_source": {"type": "string", "enum": ["callback", "hybrid"], "default": "callback"},
                "eval_during_train": {"type": "boolean", "default": True},
                "eval_interval": {"type": "integer", "minimum": 1, "default": 1},
                "momentum": {"type": "number"},
                "weight_decay": {"type": "number"},
                "warmup_epochs": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": True,
        }

    def normalize_config(self, raw: Dict[str, Any] | None) -> Dict[str, Any]:
        return dict(raw or {})

    def run(self, spec: TrainingExecutionSpec, callbacks: TrainingCallbacks) -> None:  # noqa: C901
        paddle = None
        try:
            import paddle
        except Exception as exc:
            message = str(exc)
            if "Descriptors cannot be created directly" in message:
                os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
                try:
                    import paddle
                except Exception:
                    pass
            if paddle is None:
                hint = (
                    "PaddlePaddle import failed. "
                    "This is often caused by protobuf incompatibility "
                    "(install \`protobuf<=3.20.3\`) or a broken paddle installation."
                )
                raise RuntimeError(
                    f"{hint} Original error: {type(exc).__name__}: {message}"
                ) from exc

        try:
            ensure_paddledet_repo_on_syspath()
            from ppdet.core.workspace import load_config
            from ppdet.engine import Trainer as PPTrainer
        except Exception as exc:
            raise RuntimeError(paddledet_missing_message()) from exc

        framework_config = dict(spec.framework_config)
        model_variant = (str(spec.variant or "") or "ppyoloe_s").strip()

        config_path = framework_config.get("config_path") or DEFAULT_CONFIGS.get(model_variant.lower(), "")
        if not config_path:
            raise ValidationError(
                f"No PaddleDetection config found for variant '{model_variant}'. "
                "Please provide config_path in framework config or architecture defaults."
            )
        cfg_path = resolve_paddledet_config_path(str(config_path))
        if cfg_path is None or not cfg_path.exists():
            raise ValidationError(
                f"PaddleDetection config not found: {config_path}. "
                f"{paddledet_missing_message()}"
            )

        data_yaml = find_yolo_dataset_yaml(spec.dataset_path, dataset_name=spec.dataset_name)
        if data_yaml is None or not data_yaml.exists():
            raise ValidationError("Dataset YAML not found; cannot derive train/val splits and class names")
        data_cfg = load_yaml(data_yaml)
        class_names = normalize_yolo_names(data_cfg.get("names"), data_cfg.get("nc"))
        if not class_names:
            class_names = ["class_0"]
        train_spec = str(data_cfg.get("train") or "").strip()
        val_spec = str(data_cfg.get("val") or "").strip()
        if not train_spec or not val_spec:
            raise ValidationError("Dataset YAML missing train/val; please split dataset first")
        train_images = read_image_list(spec.dataset_path, train_spec)
        val_images = read_image_list(spec.dataset_path, val_spec)
        if not train_images or not val_images:
            raise ValidationError("train/val image lists empty; please verify dataset split")

        dataset_dir = str(spec.dataset_path.resolve(strict=False))
        coco_dir = (spec.run_dir / "coco").resolve(strict=False)
        coco_dir.mkdir(parents=True, exist_ok=True)
        train_json = build_coco_from_yolo_list(
            spec.dataset_path, train_images, class_names, output_json_path=coco_dir / "train.json"
        )
        val_json = build_coco_from_yolo_list(
            spec.dataset_path, val_images, class_names, output_json_path=coco_dir / "val.json"
        )

        download_patches = apply_download_patches(dataset_dir)
        try:
            patch_ppdet_training_stats()
            patch_ppdet_assigner_label_dtype()
            cfg = load_config(str(cfg_path))

            overrides: Dict[str, Any] = {
                "epoch": int(spec.epochs),
                "worker_num": int(spec.workers or 4),
                "save_dir": str(spec.run_dir),
                "TrainDataset.dataset_dir": dataset_dir,
                "TrainDataset.anno_path": str(train_json),
                "TrainDataset.image_dir": "",
                "EvalDataset.dataset_dir": dataset_dir,
                "EvalDataset.anno_path": str(val_json),
                "EvalDataset.image_dir": "",
                "TestDataset.dataset_dir": dataset_dir,
                "TestDataset.anno_path": str(val_json),
                "TestDataset.image_dir": "",
                "num_classes": len(class_names),
                "TrainReader.batch_size": int(spec.batch_size or 8),
                "LearningRate.base_lr": float(spec.learning_rate),
            }
            if "picodet" in model_variant.lower():
                image_size = int(spec.image_size or 640)
                overrides["TrainReader.inputs_def.image_shape"] = [3, image_size, image_size]
                overrides["EvalReader.inputs_def.image_shape"] = [3, image_size, image_size]

            optimizer_name = str(spec.optimizer or "auto").strip()
            if optimizer_name.lower() not in ("auto", ""):
                paddle_opt_map = {"sgd": "Momentum", "adam": "Adam", "adamw": "AdamW"}
                overrides["OptimizerBuilder.optimizer.type"] = paddle_opt_map.get(
                    optimizer_name.lower(), optimizer_name
                )
            if spec.momentum is not None:
                overrides["OptimizerBuilder.optimizer.momentum"] = float(spec.momentum)
            if spec.weight_decay is not None:
                overrides["OptimizerBuilder.regularizer.factor"] = float(spec.weight_decay)

            eval_during_train = _coerce_bool(framework_config.get("eval_during_train", True), True)
            metrics_source = str(framework_config.get("metrics_source", "callback") or "callback").strip().lower()
            eval_interval = _safe_int(
                framework_config.get("eval_interval")
                or framework_config.get("snapshot_epoch")
                or framework_config.get("save_period")
            )
            if eval_interval is None or eval_interval <= 0:
                eval_interval = 1

            apply_cfg_overrides(cfg, overrides)
            apply_lr_scheduler_to_cfg(cfg, spec.lr_scheduler, epochs=int(spec.epochs))
            if spec.warmup_epochs is not None:
                apply_warmup_epochs_to_cfg(cfg, int(spec.warmup_epochs))
            cfg["snapshot_epoch"] = int(max(1, eval_interval))
            bind_ppdet_dataset_cfg(
                cfg, dataset_dir=dataset_dir, train_json=train_json, val_json=val_json
            )
            if metrics_source == "hybrid":
                try:
                    import visualdl  # noqa: F401
                    cfg["use_vdl"] = True
                    cfg["vdl_log_dir"] = str(spec.run_dir / "vdl_log_dir")
                except Exception:
                    pass

            print(
                "[paddle_det] dataset binding "
                f"dataset_dir={dataset_dir} train_json={train_json} val_json={val_json}",
                flush=True,
            )
            print(
                "[paddle_det] final dataset cfg "
                f"train={json.dumps(summarize_ppdet_dataset_cfg(cfg, 'TrainDataset'), ensure_ascii=False)} "
                f"eval={json.dumps(summarize_ppdet_dataset_cfg(cfg, 'EvalDataset'), ensure_ascii=False)} "
                f"test={json.dumps(summarize_ppdet_dataset_cfg(cfg, 'TestDataset'), ensure_ascii=False)}",
                flush=True,
            )

            pretrain_weights: str | None = None
            resume_checkpoint: str | None = None
            if spec.resume_training and spec.resume_job_id:
                previous_dir = settings.training_dir / str(spec.resume_job_id)
                previous = previous_dir / "model_final.pdparams"
                if not previous.exists():
                    candidates = list(previous_dir.rglob("best_model/*.pdparams"))
                    if candidates:
                        previous = candidates[0]
                if not previous.exists():
                    raise ValidationError(f"Resume checkpoint not found for run_id={spec.resume_job_id}")
                resume_checkpoint = str(previous).replace(".pdparams", "")
            elif spec.use_pretrained and spec.pretrained_model_path:
                resolved = Path(str(spec.pretrained_model_path))
                if not resolved.is_absolute():
                    for resolver in (resolve_temp_path, resolve_pretrain_path):
                        candidate = resolver(str(spec.pretrained_model_path))
                        if candidate.exists():
                            resolved = candidate
                            break
                if resolved.exists():
                    pretrain_weights = str(resolved)
                    if pretrain_weights.endswith(".pdparams"):
                        pretrain_weights = pretrain_weights[: -len(".pdparams")]
                else:
                    pretrain_weights = str(spec.pretrained_model_path)

            requested_device_value = normalize_device_spec(spec.requested_device or "auto")
            device_value = normalize_device_spec(spec.runtime_device or requested_device_value)
            selected_gpu_ids = extract_selected_gpu_ids(device_value)
            use_gpu = device_value != "cpu"
            if not use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                if use_gpu and not paddle.is_compiled_with_cuda():
                    use_gpu = False
            except Exception:
                pass
            cfg["use_gpu"] = use_gpu

            try:
                paddle_device = "cpu"
                if use_gpu:
                    paddle_device = f"gpu:{selected_gpu_ids[0]}" if selected_gpu_ids else "gpu"
                paddle.set_device(paddle_device)
                print(
                    "[paddle_det] runtime device "
                    f"requested={requested_device_value} runtime={device_value} "
                    f"paddle_device={paddle_device} "
                    f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '<inherit>')}",
                    flush=True,
                )
            except Exception:
                pass

            try:
                trainer = PPTrainer(cfg, mode="train")
            finally:
                restore_download_patches(download_patches)

            rebind_trainer_datasets(
                trainer, dataset_dir=dataset_dir, train_json=train_json, val_json=val_json
            )
            if resume_checkpoint:
                trainer.resume_weights(resume_checkpoint)
            elif pretrain_weights:
                trainer.load_weights(pretrain_weights)

            last_cancel_check = {"t": 0.0}

            def check_cancel() -> bool:
                now = time.time()
                if now - last_cancel_check["t"] < 2.0:
                    return False
                last_cancel_check["t"] = now
                return bool(callbacks.cancel_requested())

            try:
                from ppdet.engine.callbacks import Callback

                class MetricsAndCancelCallback(Callback):
                    def __init__(self, pp_trainer: Any) -> None:
                        super().__init__(None)
                        self._trainer = pp_trainer

                    @staticmethod
                    def _extract_metrics(status: dict) -> Dict[str, float]:
                        metrics: Dict[str, float] = {}
                        direct_map = (
                            ("loss", "loss"), ("loss_cls", "loss_cls"), ("loss_iou", "loss_iou"),
                            ("loss_dfl", "loss_dfl"), ("loss_obj", "loss_obj"),
                            ("learning_rate", "lr"), ("lr", "lr"),
                            ("precision", "precision"), ("recall", "recall"),
                            ("mAP", "mAP"), ("AP50", "AP50"), ("AP75", "AP75"),
                        )
                        for source, destination in direct_map:
                            value = status.get(source)
                            scalar = _safe_float(value)
                            if scalar is not None:
                                metrics[destination] = scalar

                        stats_obj = (
                            status.get("training_staus")
                            or status.get("training_statis")
                            or status.get("training_stats")
                        )
                        stats_dict = stats_obj if isinstance(stats_obj, dict) else None
                        if stats_dict is None and stats_obj is not None and hasattr(stats_obj, "get"):
                            try:
                                candidate = stats_obj.get()
                                if isinstance(candidate, dict):
                                    stats_dict = candidate
                            except Exception:
                                pass
                        if isinstance(stats_dict, dict):
                            for key, value in stats_dict.items():
                                scalar = _safe_float(value)
                                if scalar is not None:
                                    metrics[str(key)] = scalar
                        if not metrics and stats_obj is not None and hasattr(stats_obj, "meters"):
                            try:
                                meters = getattr(stats_obj, "meters", None) or {}
                                if isinstance(meters, dict):
                                    for key, meter in meters.items():
                                        for attr in ("avg", "global_avg", "median", "value"):
                                            if hasattr(meter, attr):
                                                scalar = _safe_float(getattr(meter, attr))
                                                if scalar is not None:
                                                    metrics[str(key)] = scalar
                                                    break
                            except Exception:
                                pass
                        return metrics

                    @staticmethod
                    def _extract_eval_metrics_from_trainer(pp_trainer: Any) -> Dict[str, float]:
                        metrics: Dict[str, float] = {}

                        def set_metric(key: str, value: Any) -> None:
                            scalar = _safe_float(value)
                            if scalar is not None:
                                metrics[str(key)] = scalar

                        metric_objects = getattr(pp_trainer, "_metrics", None)
                        if not isinstance(metric_objects, (list, tuple)):
                            return metrics
                        for metric_object in metric_objects:
                            get_results = getattr(metric_object, "get_results", None)
                            if not callable(get_results):
                                continue
                            try:
                                results = get_results() or {}
                            except Exception:
                                continue
                            if not isinstance(results, dict):
                                continue
                            for group_key, value in results.items():
                                group = str(group_key or "metric")
                                if isinstance(value, dict):
                                    for key, item in value.items():
                                        set_metric(f"eval/{group}_{key}", item)
                                    continue
                                sequence: list[Any] | None = None
                                if isinstance(value, (list, tuple)):
                                    sequence = list(value)
                                elif hasattr(value, "tolist"):
                                    try:
                                        converted = value.tolist()
                                    except Exception:
                                        converted = None
                                    if isinstance(converted, (list, tuple)):
                                        sequence = list(converted)
                                if sequence is None:
                                    set_metric(f"eval/{group}", value)
                                    continue
                                if len(sequence) >= 1:
                                    set_metric(f"eval/{group}_mAP", sequence[0])
                                    if group == "bbox":
                                        set_metric("mAP", sequence[0])
                                        set_metric("metrics/mAP50-95(B)", sequence[0])
                                if len(sequence) >= 2:
                                    set_metric(f"eval/{group}_AP50", sequence[1])
                                    if group == "bbox":
                                        set_metric("AP50", sequence[1])
                                        set_metric("mAP50", sequence[1])
                                        set_metric("metrics/mAP50(B)", sequence[1])
                                if len(sequence) >= 3:
                                    set_metric(f"eval/{group}_AP75", sequence[2])
                                    if group == "bbox":
                                        set_metric("AP75", sequence[2])
                        return metrics

                    def on_epoch_end(self, status: dict) -> None:
                        epoch = int(status.get("epoch_id", 0))
                        mode = str(status.get("mode", "") or "").lower()
                        metrics = self._extract_metrics(status)
                        if mode == "eval":
                            metrics.update(self._extract_eval_metrics_from_trainer(self._trainer))
                        metrics = _apply_metric_aliases(metrics)
                        if metrics:
                            callbacks.upsert_epoch_metrics(epoch, metrics)
                        if check_cancel():
                            raise SystemExit(0)

                    def on_step_end(self, status: dict) -> None:
                        mode = str(status.get("mode", "") or "").lower()
                        epoch = int(status.get("epoch_id", 0))
                        step = int(status.get("step_id", -1))
                        if mode == "train" and step == 0:
                            metrics = _apply_metric_aliases(self._extract_metrics(status))
                            if metrics:
                                callbacks.upsert_epoch_metrics(epoch, metrics)
                        if check_cancel():
                            raise SystemExit(0)

                cancel_callback = MetricsAndCancelCallback(trainer)
                injected = False
                if hasattr(trainer, "_callbacks") and isinstance(trainer._callbacks, list):
                    trainer._callbacks.append(cancel_callback)
                    injected = True
                if hasattr(trainer, "_compose_callback") and hasattr(trainer._compose_callback, "_callbacks"):
                    callback_list = getattr(trainer._compose_callback, "_callbacks")
                    if isinstance(callback_list, list):
                        callback_list.append(cancel_callback)
                        injected = True
                if not injected:
                    raise RuntimeError("Paddle callback container not found")
            except Exception:
                pass

            trainer.train(validate=bool(eval_during_train))
            try:
                trainer.evaluate()
            except Exception:
                pass

            weights_dir = spec.run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint, last_checkpoint = _pick_paddle_checkpoint(spec.run_dir)
            if last_checkpoint and last_checkpoint.exists():
                shutil.copy2(last_checkpoint, weights_dir / "last.pdparams")
                optimizer_file = last_checkpoint.with_suffix(".pdopt")
                if optimizer_file.exists():
                    shutil.copy2(optimizer_file, weights_dir / "last.pdopt")
            if best_checkpoint and best_checkpoint.exists():
                shutil.copy2(best_checkpoint, weights_dir / "best.pdparams")
                optimizer_file = best_checkpoint.with_suffix(".pdopt")
                if optimizer_file.exists():
                    shutil.copy2(optimizer_file, weights_dir / "best.pdopt")
        finally:
            restore_download_patches(download_patches)
