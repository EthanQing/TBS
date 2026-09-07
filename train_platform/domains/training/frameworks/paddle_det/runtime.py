from __future__ import annotations

import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable

from train_platform.core.config import settings


def coerce_metric_scalar(value: Any) -> float | None:
    """
    Convert Paddle / NumPy metric values to plain Python floats defensively.

    PaddleDetection 2.6 may surface ndarray-like values in training meters,
    which later crash its logger on ``format(value, ".6f")``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)

    for attr in ("item", "numpy", "tolist"):
        if not hasattr(value, attr):
            continue
        try:
            nested = getattr(value, attr)()
        except Exception:
            continue
        if nested is value:
            continue
        coerced = coerce_metric_scalar(nested)
        if coerced is not None:
            return coerced

    if isinstance(value, (list, tuple)):
        numeric_values = [fv for item in value if (fv := coerce_metric_scalar(item)) is not None]
        if not numeric_values:
            return None
        if len(numeric_values) == 1:
            return numeric_values[0]
        return float(sum(numeric_values) / len(numeric_values))

    try:
        return float(value)
    except Exception:
        return None


def patch_ppdet_training_stats() -> None:
    """Patch PaddleDetection stats logging to tolerate ndarray-like values."""
    try:
        from ppdet.utils import stats as ppdet_stats
    except Exception:
        return

    if getattr(ppdet_stats, "_train_platform_safe_stats_patch", False):
        return

    smoothed_value_cls = getattr(ppdet_stats, "SmoothedValue", None)
    training_stats_cls = getattr(ppdet_stats, "TrainingStats", None)

    if smoothed_value_cls is not None and callable(getattr(smoothed_value_cls, "update", None)):
        original_smoothed_update = smoothed_value_cls.update

        def _safe_smoothed_update(self: Any, value: Any) -> Any:
            scalar = coerce_metric_scalar(value)
            if scalar is not None:
                value = scalar
            return original_smoothed_update(self, value)

        smoothed_value_cls.update = _safe_smoothed_update

    if training_stats_cls is not None and callable(getattr(training_stats_cls, "update", None)):
        def _safe_training_update(self: Any, stats: Any) -> Any:
            if not isinstance(stats, dict):
                return None

            meters = getattr(self, "meters", None)
            if not isinstance(meters, dict):
                window_size = int(getattr(self, "window_size", 20) or 20)
                meters = {
                    str(k): smoothed_value_cls(window_size)
                    for k in stats.keys()
                } if smoothed_value_cls is not None else {}
                self.meters = meters

            for key, value in stats.items():
                key_str = str(key)
                meter = meters.get(key_str)
                if meter is None and smoothed_value_cls is not None:
                    meter = smoothed_value_cls(int(getattr(self, "window_size", 20) or 20))
                    meters[key_str] = meter
                if meter is None:
                    continue

                scalar = coerce_metric_scalar(value)
                if scalar is not None:
                    meter.update(scalar)
                    continue

                try:
                    meter.update(value.numpy())
                except Exception:
                    try:
                        meter.update(value)
                    except Exception:
                        continue
            return None

        training_stats_cls.update = _safe_training_update

    if training_stats_cls is not None and callable(getattr(training_stats_cls, "get", None)):
        def _safe_training_get(self: Any, extras: Iterable[str] | None = None) -> Dict[str, str]:
            meters = getattr(self, "meters", None)
            if not isinstance(meters, dict):
                return {}

            extras_set = {str(k) for k in (extras or [])}
            stats: Dict[str, str] = {}

            def _format_meter(meter: Any, *, prefer_avg: bool) -> str:
                attr_order = ("avg", "global_avg", "median", "value") if prefer_avg else (
                    "median", "avg", "global_avg", "value"
                )
                for attr in attr_order:
                    if not hasattr(meter, attr):
                        continue
                    scalar = coerce_metric_scalar(getattr(meter, attr))
                    if scalar is not None:
                        return format(scalar, ".4f" if prefer_avg else ".6f")
                raw_value = getattr(meter, "avg" if prefer_avg else "median", None)
                return str(raw_value)

            for key, meter in meters.items():
                key_str = str(key)
                stats[key_str] = _format_meter(meter, prefer_avg=key_str in extras_set)
            return stats

        training_stats_cls.get = _safe_training_get

    ppdet_stats._train_platform_safe_stats_patch = True


def patch_ppdet_assigner_label_dtype() -> None:
    """
    Patch PaddleDetection assigners to avoid int32/int64 promotion crashes.

    Some Paddle 2.6 + PaddleDetection 2.6 combinations yield `gt_labels` as
    int32, while assigner internals produce int64 indices via `argmax()`.
    A later `assigned_gt_index + batch_ind * num_max_boxes` then fails with a
    type-promotion error. Casting `gt_labels` to int64 before calling the
    original assigner forward keeps the downstream math consistent.
    """
    try:
        import paddle
        from ppdet.modeling.assigners import (
            atss_assigner,
            fcosr_assigner,
            rotated_task_aligned_assigner,
            task_aligned_assigner,
            task_aligned_assigner_cr,
        )
    except Exception:
        return

    targets = [
        getattr(atss_assigner, "ATSSAssigner", None),
        getattr(fcosr_assigner, "FCOSRAssigner", None),
        getattr(rotated_task_aligned_assigner, "RotatedTaskAlignedAssigner", None),
        getattr(task_aligned_assigner, "TaskAlignedAssigner", None),
        getattr(task_aligned_assigner_cr, "TaskAlignedAssigner_CR", None),
    ]

    def _needs_cast(tensor: Any) -> bool:
        dtype = getattr(tensor, "dtype", None)
        return dtype is not None and str(dtype).lower().endswith("int32")

    for cls in targets:
        if cls is None or getattr(cls, "_train_platform_safe_label_dtype_patch", False):
            continue
        original_forward = getattr(cls, "forward", None)
        if not callable(original_forward):
            continue

        @wraps(original_forward)
        def _safe_forward(self: Any, *args: Any, __orig=original_forward, **kwargs: Any) -> Any:
            if "gt_labels" in kwargs and _needs_cast(kwargs.get("gt_labels")):
                kwargs = {**kwargs, "gt_labels": paddle.cast(kwargs["gt_labels"], "int64")}
                return __orig(self, *args, **kwargs)

            if len(args) >= 3 and _needs_cast(args[2]):
                new_args = list(args)
                new_args[2] = paddle.cast(new_args[2], "int64")
                return __orig(self, *tuple(new_args), **kwargs)

            return __orig(self, *args, **kwargs)

        cls.forward = _safe_forward
        cls._train_platform_safe_label_dtype_patch = True


def _rebind_runtime_dataset(dataset_obj: Any, *, dataset_dir: str, anno_path: str) -> bool:
    if dataset_obj is None:
        return False

    current_dataset_dir = str(getattr(dataset_obj, "dataset_dir", "") or "")
    current_anno_path = str(getattr(dataset_obj, "anno_path", "") or "")
    current_image_dir = str(getattr(dataset_obj, "image_dir", "") or "")
    changed = (
        current_dataset_dir != dataset_dir
        or current_anno_path != anno_path
        or current_image_dir != ""
    )

    for attr, value in (
        ("dataset_dir", dataset_dir),
        ("anno_path", anno_path),
        ("image_dir", ""),
    ):
        try:
            setattr(dataset_obj, attr, value)
        except Exception:
            pass

    for fn_name in ("check_or_download_dataset", "download_dataset"):
        if hasattr(dataset_obj, fn_name):
            try:
                setattr(dataset_obj, fn_name, lambda *a, **kw: None)
            except Exception:
                pass

    if changed and callable(getattr(dataset_obj, "parse_dataset", None)):
        try:
            dataset_obj.parse_dataset()
        except Exception as e:
            print(
                f"[paddle_det] runtime dataset rebind failed: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
    return changed


def rebind_trainer_datasets(trainer: Any, *, dataset_dir: str, train_json: Path, val_json: Path) -> None:
    targets = (
        (getattr(trainer, "dataset", None), str(train_json)),
        (getattr(getattr(trainer, "loader", None), "dataset", None), str(train_json)),
        (getattr(trainer, "_eval_dataset", None), str(val_json)),
        (getattr(getattr(trainer, "_eval_loader", None), "dataset", None), str(val_json)),
        (getattr(trainer, "_test_dataset", None), str(val_json)),
        (getattr(getattr(trainer, "_test_loader", None), "dataset", None), str(val_json)),
    )
    seen_ids: set[int] = set()
    for dataset_obj, anno_path in targets:
        if dataset_obj is None:
            continue
        obj_id = id(dataset_obj)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        _rebind_runtime_dataset(dataset_obj, dataset_dir=dataset_dir, anno_path=anno_path)


def apply_download_patches(dataset_dir: str) -> dict[tuple[Any, str], Any]:
    """Disable PaddleDetection's remote dataset download during local training."""
    patches: dict[tuple[Any, str], Any] = {}

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    def _true(*args: Any, **kwargs: Any) -> bool:
        return True

    def _bound_dataset_path(*args: Any, **kwargs: Any) -> str:
        return dataset_dir

    for module_name in ("ppdet.utils.download", "ppdet.core.workspace", "ppdet.data.source.dataset"):
        try:
            import importlib
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for function_name in (
            "_dataset_exists", "_check_download", "_check_and_download", "_download_data",
            "download_dataset", "_decompress", "get_dataset_path",
        ):
            if not hasattr(module, function_name):
                continue
            patches[(module, function_name)] = getattr(module, function_name)
            if function_name == "_dataset_exists":
                setattr(module, function_name, _true)
            elif function_name in ("get_dataset_path", "download_dataset"):
                setattr(module, function_name, _bound_dataset_path)
            else:
                setattr(module, function_name, _noop)
    return patches


def restore_download_patches(patches: dict[tuple[Any, str], Any]) -> None:
    for (module, function_name), original in patches.items():
        setattr(module, function_name, original)


from train_platform.platform.runtime.paddledetection import (
    PADDLE_DET_REQUIRED_CONFIG,
    is_paddledet_repo,
    paddledet_missing_message,
    resolve_paddledet_config_path,
    resolve_paddledet_repo,
)


def ensure_paddledet_repo_on_syspath() -> Path:
    repo = resolve_paddledet_repo()
    if repo is None:
        raise RuntimeError(paddledet_missing_message())
    repo_s = str(repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    return repo



__all__ = [
    "apply_download_patches",
    "coerce_metric_scalar",
    "patch_ppdet_assigner_label_dtype",
    "patch_ppdet_training_stats",
    "rebind_trainer_datasets",
    "restore_download_patches",
    "PADDLE_DET_REQUIRED_CONFIG",
    "ensure_paddledet_repo_on_syspath",
    "is_paddledet_repo",
    "paddledet_missing_message",
    "resolve_paddledet_config_path",
    "resolve_paddledet_repo",
]
