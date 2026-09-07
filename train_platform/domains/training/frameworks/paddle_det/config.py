from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from train_platform.domains.training.parameters import normalize_lr_scheduler


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _set_cfg_by_path(root: Any, dotted_key: str, value: Any) -> bool:
    """Set config value using dotted path (supports dict + list indices)."""
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        return False

    cur: Any = root
    for i, part in enumerate(parts[:-1]):
        next_part = parts[i + 1]
        next_is_index = _safe_int(next_part) is not None

        if isinstance(cur, list):
            idx = _safe_int(part)
            if idx is None or idx < 0:
                return False
            while idx >= len(cur):
                cur.append([] if next_is_index else {})
            if not isinstance(cur[idx], (dict, list)):
                cur[idx] = [] if next_is_index else {}
            cur = cur[idx]
            continue

        if not isinstance(cur, dict):
            return False

        if part not in cur or not isinstance(cur[part], (dict, list)):
            cur[part] = [] if next_is_index else {}
        cur = cur[part]

    last = parts[-1]
    if isinstance(cur, list):
        idx = _safe_int(last)
        if idx is None or idx < 0:
            return False
        while idx >= len(cur):
            cur.append(None)
        cur[idx] = value
        return True

    if isinstance(cur, dict):
        cur[last] = value
        return True

    return False

def apply_cfg_overrides(cfg: dict, overrides: Dict[str, Any]) -> None:
    """Apply flat dotted-path overrides directly onto ppdet global config."""
    for key, value in overrides.items():
        if not _set_cfg_by_path(cfg, key, value):
            cfg[key] = value


def apply_warmup_epochs_to_cfg(cfg: dict, warmup_epochs: int | None) -> bool:
    """
    Apply warmup epochs to PaddleDetection LearningRate schedulers safely.

    Different configs represent schedulers differently (dicts vs objects),
    so we avoid brittle dotted-path overrides and patch the warmup scheduler
    directly.
    """
    if warmup_epochs is None:
        return False

    lr_cfg = cfg.get("LearningRate")
    if not isinstance(lr_cfg, dict):
        return False

    schedulers = lr_cfg.get("schedulers")
    if not isinstance(schedulers, list):
        return False

    target = None
    for sch in schedulers:
        if isinstance(sch, dict):
            name = str(sch.get("name") or sch.get("type") or sch.get("_type_") or "").lower()
            if "warmup" in name:
                target = sch
                break
        else:
            if "warmup" in type(sch).__name__.lower():
                target = sch
                break

    if target is None:
        return False

    val = int(max(0, int(warmup_epochs)))
    if isinstance(target, dict):
        # Most configs use LinearWarmup with `epochs` or `steps`.
        if "epochs" in target or "steps" not in target:
            target["epochs"] = val
        else:
            target["steps"] = val
        target.pop("warmup_steps", None)
        return True

    # Object-based config nodes (e.g., already materialized scheduler objects).
    if hasattr(target, "epochs"):
        try:
            setattr(target, "epochs", val)
            return True
        except Exception:
            pass
    if hasattr(target, "steps"):
        try:
            setattr(target, "steps", val)
            return True
        except Exception:
            pass
    return False


def _is_warmup_scheduler(scheduler: Any) -> bool:
    if isinstance(scheduler, dict):
        name = str(scheduler.get("name") or scheduler.get("type") or scheduler.get("_type_") or "").lower()
        return "warmup" in name
    return "warmup" in type(scheduler).__name__.lower()


def apply_lr_scheduler_to_cfg(cfg: dict, scheduler: Any, *, epochs: int) -> bool:
    if normalize_lr_scheduler(scheduler) == "linear":
        return False

    lr_cfg = cfg.get("LearningRate")
    if not isinstance(lr_cfg, dict):
        lr_cfg = {}
        cfg["LearningRate"] = lr_cfg

    schedulers = lr_cfg.get("schedulers")
    if not isinstance(schedulers, list):
        schedulers = []

    cosine_scheduler = {
        "name": "CosineDecay",
        "max_epochs": int(max(1, int(epochs or 1))),
    }
    next_schedulers: list[Any] = []
    inserted = False
    for item in schedulers:
        if _is_warmup_scheduler(item):
            next_schedulers.append(item)
            continue
        if not inserted:
            next_schedulers.append(cosine_scheduler)
            inserted = True
    if not inserted:
        next_schedulers.insert(0, cosine_scheduler)
    lr_cfg["schedulers"] = next_schedulers
    return True


# Default PaddleDetection config templates for each supported variant.
# These will be used when no explicit config_path is provided.
DEFAULT_CONFIGS: Dict[str, str] = {
    "ppyoloe_s": "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml",
    "ppyoloe_m": "configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml",
    "ppyoloe_l": "configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml",
    "ppyoloe_x": "configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml",
    "picodet_s": "configs/picodet/picodet_s_320_coco_lcnet.yml",
    "picodet_l": "configs/picodet/picodet_l_640_coco_lcnet.yml",
}


def bind_ppdet_dataset_cfg(cfg: dict, *, dataset_dir: str, train_json: Path, val_json: Path) -> None:
    targets = {
        "TrainDataset": str(train_json),
        "EvalDataset": str(val_json),
        "TestDataset": str(val_json),
    }
    for dataset_key, annotation_path in targets.items():
        node = cfg.get(dataset_key)
        if not isinstance(node, dict):
            node = {}
            cfg[dataset_key] = node
        node["dataset_dir"] = dataset_dir
        node["image_dir"] = ""
        node["anno_path"] = annotation_path


__all__ = [
    "DEFAULT_CONFIGS",
    "apply_cfg_overrides",
    "apply_lr_scheduler_to_cfg",
    "apply_warmup_epochs_to_cfg",
    "bind_ppdet_dataset_cfg",
]
