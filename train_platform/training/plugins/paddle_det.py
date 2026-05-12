from __future__ import annotations

"""
PaddleDetection Trainer Plugin.

Provides training support for PaddlePaddle-based detection models such as
PP-YOLOE, PicoDet, etc.  Datasets are expected in the platform's standard
YOLO format (data.yaml + txt labels); this plugin converts them to COCO JSON
on-the-fly before feeding them to PaddleDetection's ``Trainer``.

Lazy-imports: ``paddle`` and ``ppdet`` are imported only inside ``run()`` so
that the rest of the platform works even when PaddlePaddle is not installed.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from train_platform.core.config import settings
from train_platform.training.plugins.base import TrainContext
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.path_utils import resolve_pretrain_path, resolve_temp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_bool(value: Any, default: bool) -> bool:
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


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _apply_metric_aliases(raw_metrics: Dict[str, Any] | None) -> Dict[str, float]:
    """
    Normalize metric keys across training engines while preserving originals.

    PaddleDetection and Ultralytics often use different key names for the
    same concepts; this helper adds stable aliases used by the frontend.
    """
    raw = raw_metrics if isinstance(raw_metrics, dict) else {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        fv = _safe_float(v)
        if fv is not None:
            out[str(k)] = fv

    def _alias(dst: str, *srcs: str) -> None:
        if _safe_float(out.get(dst)) is not None:
            return
        for src in srcs:
            fv = _safe_float(out.get(src))
            if fv is not None:
                out[dst] = fv
                return

    # Canonical detection metrics used by the unified frontend charts.
    _alias("metrics/mAP50(B)", "AP50", "mAP50", "eval/bbox_AP50", "eval/bbox_ap50")
    _alias("metrics/mAP50-95(B)", "mAP", "eval/bbox_mAP", "eval/bbox_map")
    _alias("metrics/precision(B)", "precision", "eval/bbox_precision", "eval/precision")
    _alias("metrics/recall(B)", "recall", "eval/bbox_recall", "eval/recall")

    # Backward compatibility for old keys consumed by existing clients.
    _alias("AP50", "metrics/mAP50(B)")
    _alias("mAP50", "metrics/mAP50(B)")
    _alias("mAP", "metrics/mAP50-95(B)")
    _alias("precision", "metrics/precision(B)")
    _alias("recall", "metrics/recall(B)")
    return out


def _ensure_local_ppdet_on_syspath() -> Path | None:
    """
    Best-effort: add a local PaddleDetection repo to sys.path.

    This enables local development where PaddleDetection is cloned but not
    installed as a wheel/package.
    """
    roots = []
    raw = settings.paddle_det_dir
    roots.append(raw)
    # Allow pointing either to repo root or to its parent directory.
    roots.append(raw / "PaddleDetection")

    for candidate in roots:
        try:
            root = candidate.resolve(strict=False)
        except Exception:
            root = candidate
        if not root.exists() or not root.is_dir():
            continue

        # Typical PaddleDetection repo root contains ./ppdet package directory.
        if not (root / "ppdet").is_dir():
            # Also allow PADDLE_DET_DIR directly pointing to ./ppdet.
            if root.name.lower() == "ppdet" and root.parent.is_dir():
                root = root.parent
            else:
                continue

        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
        return root
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


def _apply_cfg_overrides(cfg: dict, overrides: Dict[str, Any]) -> None:
    """Apply flat dotted-path overrides directly onto ppdet global config."""
    for key, value in overrides.items():
        if not _set_cfg_by_path(cfg, key, value):
            cfg[key] = value


def _load_yaml(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        obj = {}
    return obj if isinstance(obj, dict) else {}


def _apply_warmup_epochs_to_cfg(cfg: dict, warmup_epochs: int | None) -> bool:
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


def _normalize_yolo_names(names_obj: Any, nc_obj: Any) -> list[str]:
    if isinstance(names_obj, list):
        return [str(x) for x in names_obj if str(x).strip()]
    if isinstance(names_obj, dict):
        try:
            keys = sorted(int(k) for k in names_obj.keys())
            return [str(names_obj.get(i) or names_obj.get(str(i)) or f"class_{i}") for i in keys]
        except Exception:
            return [str(v) for v in names_obj.values() if str(v).strip()]
    nc = _safe_int(nc_obj)
    if nc is not None and nc > 0:
        return [f"class_{i}" for i in range(int(nc))]
    return []


def _read_image_list(dataset_root: Path, spec: str) -> list[Path]:
    """Read YOLO train/val spec (txt file or directory) → list of image paths."""
    s = str(spec or "").strip()
    if not s:
        return []
    p = Path(s)
    if not p.is_absolute():
        p = (dataset_root / p).resolve(strict=False)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    out: list[Path] = []

    if p.exists() and p.is_file() and p.suffix.lower() == ".txt":
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            ip = Path(line)
            if not ip.is_absolute():
                ip = (dataset_root / ip).resolve(strict=False)
            if ip.suffix.lower() in image_exts:
                out.append(ip)
        return out

    if p.exists() and p.is_dir():
        try:
            for img in sorted(p.rglob("*")):
                if img.is_file() and img.suffix.lower() in image_exts:
                    out.append(img)
        except Exception:
            pass
        return out

    return []


def _derive_label_path(dataset_root: Path, image_abs: Path) -> Path:
    try:
        rel = image_abs.resolve(strict=False).relative_to(dataset_root.resolve(strict=False))
    except Exception:
        rel = image_abs.name  # type: ignore[assignment]
    rel_p = Path(rel)
    parts = list(rel_p.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return (dataset_root / Path(*parts)).with_suffix(".txt")
    return (dataset_root / "labels" / rel_p).with_suffix(".txt")


def _build_coco_from_yolo_list(
    dataset_root: Path,
    image_paths: Iterable[Path],
    class_names: list[str],
    *,
    output_json_path: Path,
) -> Path:
    """Build a COCO-format annotation JSON from YOLO txt labels."""
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow (PIL) is required for COCO conversion") from e

    cats = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(class_names)]
    coco: Dict[str, Any] = {"images": [], "annotations": [], "categories": cats, "licenses": [], "info": {}}
    img_id = 1
    ann_id = 1
    ordered = sorted([Path(p) for p in image_paths], key=lambda x: x.as_posix().lower())

    for img_abs in ordered:
        img_abs = Path(img_abs).resolve(strict=False)
        if not img_abs.exists() or not img_abs.is_file():
            continue
        try:
            with Image.open(img_abs) as im:
                width, height = im.size
        except Exception:
            continue
        try:
            file_name = img_abs.relative_to(dataset_root).as_posix()
        except Exception:
            file_name = img_abs.name

        coco["images"].append({"id": img_id, "file_name": file_name, "width": int(width), "height": int(height)})
        label_path = _derive_label_path(dataset_root, img_abs)
        if label_path.exists() and label_path.is_file():
            try:
                text = label_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = _safe_int(parts[0])
                x_c, y_c, w_n, h_n = _safe_float(parts[1]), _safe_float(parts[2]), _safe_float(parts[3]), _safe_float(parts[4])
                if any(v is None for v in (cid, x_c, y_c, w_n, h_n)):
                    continue
                w_abs = max(0.0, float(w_n) * float(width))
                h_abs = max(0.0, float(h_n) * float(height))
                x_min = max(0.0, float(x_c) * float(width) - w_abs / 2.0)
                y_min = max(0.0, float(y_c) * float(height) - h_abs / 2.0)
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cid) + 1,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w_abs, 2), round(h_abs, 2)],
                    "area": round(w_abs * h_abs, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1
        img_id += 1

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return output_json_path


# ---------------------------------------------------------------------------
# PaddleDetection config helpers
# ---------------------------------------------------------------------------

# Default PaddleDetection config templates for each supported variant.
# These will be used when no explicit config_path is provided.
_DEFAULT_CONFIGS: Dict[str, str] = {
    "ppyoloe_s": "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml",
    "ppyoloe_m": "configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml",
    "ppyoloe_l": "configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml",
    "ppyoloe_x": "configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml",
    "picodet_s": "configs/picodet/picodet_s_320_coco_lcnet.yml",
    "picodet_l": "configs/picodet/picodet_l_640_coco_lcnet.yml",
}


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

    def run(self, ctx: TrainContext, *, config: Dict[str, Any] | None = None) -> None:  # noqa: C901  (complexity is inherent)
        # ---- Lazy imports ----
        paddle = None
        try:
            import paddle
        except Exception as e:
            msg = str(e)
            # Compatibility fallback for old paddle/protobuf combinations.
            if "Descriptors cannot be created directly" in msg:
                os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
                try:
                    import paddle
                except Exception:
                    pass
            if paddle is None:
                hint = (
                    "PaddlePaddle import failed. "
                    "This is often caused by protobuf incompatibility "
                    "(install `protobuf<=3.20.3`) or a broken paddle installation."
                )
                raise RuntimeError(
                    f"{hint} Original error: {type(e).__name__}: {msg}"
                ) from e

        ppdet_module = None
        try:
            import ppdet as ppdet_module  # type: ignore
            from ppdet.core.workspace import load_config
            from ppdet.engine import Trainer as PPTrainer
        except Exception as first_error:
            # Local-dev fallback: use source repo via PADDLE_DET_DIR without pip install.
            _ensure_local_ppdet_on_syspath()
            try:
                import ppdet as ppdet_module  # type: ignore
                from ppdet.core.workspace import load_config
                from ppdet.engine import Trainer as PPTrainer
            except Exception as second_error:
                root_error = second_error or first_error
                raise RuntimeError(
                    "PaddleDetection (ppdet) is not available. "
                    "Install paddledet, or set PADDLE_DET_DIR to a local PaddleDetection repo."
                ) from root_error

        job = ctx.job
        add = getattr(getattr(job, "parameters", None), "additional_params", None) or {}
        if isinstance(add.get("framework_config"), dict):
            add = {**add, **dict(add.get("framework_config") or {})}
        if config:
            add = {**add, **self.normalize_config(config)}
        arch_defaults = getattr(getattr(job, "architecture", None), "default_params", None) or {}

        # ---- Resolve variant & config ----
        model_variant = ""
        if getattr(job, "architecture", None) is not None:
            model_variant = str(getattr(job.architecture, "variant", "") or "")
        model_variant = (model_variant or "ppyoloe_s").strip()

        config_path = add.get("config_path") or arch_defaults.get("config_path") or ""
        if not config_path:
            config_path = _DEFAULT_CONFIGS.get(model_variant.lower(), "")
        if not config_path:
            raise ValidationError(
                f"No PaddleDetection config found for variant '{model_variant}'.  "
                f"Please provide config_path in additional_params or architecture default_params."
            )

        cfg_path = Path(str(config_path))
        if not cfg_path.is_absolute():
            # 1) Try PaddleDetection repo clone (settings.paddle_det_dir)
            repo_roots = [settings.paddle_det_dir, settings.paddle_det_dir / "PaddleDetection"]
            if ppdet_module is not None:
                try:
                    pkg_init = Path(ppdet_module.__file__).resolve()
                    repo_roots.extend([pkg_init.parent.parent, pkg_init.parent])
                except Exception:
                    pass

            seen_roots: set[str] = set()
            for root in repo_roots:
                key = str(root)
                if key in seen_roots:
                    continue
                seen_roots.add(key)
                cand = (root / cfg_path).resolve(strict=False)
                if cand.exists():
                    cfg_path = cand
                    break
            else:
                # 2) Try temp / pretrain_models / cwd
                for resolver in (resolve_temp_path, resolve_pretrain_path):
                    cand = resolver(str(config_path))
                    if cand.exists():
                        cfg_path = cand
                        break
                else:
                    cfg_path = (Path.cwd() / cfg_path).resolve(strict=False)

        if not cfg_path.exists():
            raise ValidationError(f"PaddleDetection config not found: {cfg_path}")

        # ---- Dataset: YOLO → COCO ----
        data_yaml = find_yolo_dataset_yaml(ctx.dataset_path)
        if data_yaml is None or not data_yaml.exists():
            raise ValidationError("Dataset YAML not found; cannot derive train/val splits and class names")

        data_cfg = _load_yaml(data_yaml)
        class_names = _normalize_yolo_names(data_cfg.get("names"), data_cfg.get("nc"))
        if not class_names:
            class_names = ["class_0"]
        num_classes = len(class_names)

        train_spec = str(data_cfg.get("train") or "").strip()
        val_spec = str(data_cfg.get("val") or "").strip()
        if not train_spec or not val_spec:
            raise ValidationError("Dataset YAML missing train/val; please split dataset first")

        train_images = _read_image_list(ctx.dataset_path, train_spec)
        val_images = _read_image_list(ctx.dataset_path, val_spec)
        if not train_images or not val_images:
            raise ValidationError("train/val image lists empty; please verify dataset split")

        coco_dir = (ctx.run_dir / "coco").resolve(strict=False)
        coco_dir.mkdir(parents=True, exist_ok=True)
        train_json = _build_coco_from_yolo_list(
            ctx.dataset_path, train_images, class_names,
            output_json_path=coco_dir / "train.json",
        )
        val_json = _build_coco_from_yolo_list(
            ctx.dataset_path, val_images, class_names,
            output_json_path=coco_dir / "val.json",
        )

        # ---- Load & merge config ----
        # PaddleDetection validates dataset paths and tries to download COCO
        # both during load_config() and Trainer.__init__().
        # We must disable ALL download/check functions across ppdet modules
        # and keep the patch active until after Trainer is fully initialized.
        _download_patches: Dict[str, Any] = {}

        def _apply_download_patches() -> None:
            """Disable all ppdet dataset-download helpers."""
            _noop = lambda *a, **kw: None
            _true = lambda *a, **kw: True
            _identity_dataset_path = lambda path, *a, **kw: path
            for mod_name in ("ppdet.utils.download", "ppdet.core.workspace", "ppdet.data.source.dataset"):
                try:
                    import importlib
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
                for fn_name in (
                    "_dataset_exists", "_check_download",
                    "_check_and_download", "_download_data",
                    "download_dataset", "_decompress",
                    "get_dataset_path",
                ):
                    if hasattr(mod, fn_name):
                        _download_patches[(mod, fn_name)] = getattr(mod, fn_name)
                        # _dataset_exists should return True; others should be no-ops
                        if fn_name == "_dataset_exists":
                            setattr(mod, fn_name, _true)
                        elif fn_name == "get_dataset_path":
                            setattr(mod, fn_name, _identity_dataset_path)
                        else:
                            setattr(mod, fn_name, _noop)

        def _restore_download_patches() -> None:
            for (mod, fn_name), fn_ref in _download_patches.items():
                setattr(mod, fn_name, fn_ref)

        _apply_download_patches()

        cfg = load_config(str(cfg_path))

        epochs = int(getattr(job.parameters, "epochs", 80) or 80)
        batch_size = int(getattr(job.parameters, "batch_size", 8) or 8)
        learning_rate = float(getattr(job.parameters, "learning_rate", 0.01) or 0.01)
        image_size = int(getattr(job.parameters, "image_size", 640) or 640)
        workers = int(getattr(job.parameters, "workers", 4) or 4)

        # Build overrides dict (flat dotted paths are applied by _apply_cfg_overrides)
        overrides: Dict[str, Any] = {
            "epoch": epochs,
            "worker_num": workers,
            "save_dir": str(ctx.run_dir),
        }

        # Dataset paths
        dataset_dir = str(ctx.dataset_path)
        overrides["TrainDataset.dataset_dir"] = dataset_dir
        overrides["TrainDataset.anno_path"] = str(train_json)
        overrides["TrainDataset.image_dir"] = ""  # file_name in COCO json is relative to dataset_dir
        overrides["EvalDataset.dataset_dir"] = dataset_dir
        overrides["EvalDataset.anno_path"] = str(val_json)
        overrides["EvalDataset.image_dir"] = ""
        overrides["TestDataset.dataset_dir"] = dataset_dir
        overrides["TestDataset.anno_path"] = str(val_json)
        overrides["TestDataset.image_dir"] = ""

        # Number of classes
        overrides["num_classes"] = num_classes

        # Batch size
        overrides["TrainReader.batch_size"] = batch_size

        # Learning rate
        overrides["LearningRate.base_lr"] = learning_rate

        # Image size (best-effort; depends on architecture config structure)
        if "picodet" in model_variant.lower():
            overrides["TrainReader.inputs_def.image_shape"] = [3, image_size, image_size]
            overrides["EvalReader.inputs_def.image_shape"] = [3, image_size, image_size]

        # Optimizer mapping
        optimizer_name = str(getattr(job.parameters, "optimizer", "auto") or "auto").strip()
        if optimizer_name.lower() not in ("auto", ""):
            paddle_opt_map = {
                "sgd": "Momentum",
                "adam": "Adam",
                "adamw": "AdamW",
            }
            mapped = paddle_opt_map.get(optimizer_name.lower(), optimizer_name)
            overrides["OptimizerBuilder.optimizer.type"] = mapped

        # Momentum / weight_decay (from additional_params)
        momentum = _safe_float(add.get("momentum"))
        if momentum is not None:
            overrides["OptimizerBuilder.optimizer.momentum"] = momentum
        weight_decay = _safe_float(add.get("weight_decay"))
        if weight_decay is not None:
            overrides["OptimizerBuilder.regularizer.factor"] = weight_decay

        # Warmup
        warmup_epochs = _safe_int(add.get("warmup_epochs"))
        eval_during_train = _coerce_bool(add.get("eval_during_train", True), True)
        metrics_source = str(add.get("metrics_source", "callback") or "callback").strip().lower()
        eval_interval = _safe_int(
            add.get("eval_interval") or add.get("snapshot_epoch") or add.get("save_period")
        )
        # For local dev UX, default to eval every epoch so mAP curves appear early.
        if eval_interval is None or eval_interval <= 0:
            eval_interval = 1

        _apply_cfg_overrides(cfg, overrides)
        _apply_warmup_epochs_to_cfg(cfg, warmup_epochs)
        cfg["snapshot_epoch"] = int(max(1, eval_interval))
        if metrics_source == "hybrid":
            try:
                import visualdl  # noqa: F401
                cfg["use_vdl"] = True
                cfg["vdl_log_dir"] = str(ctx.run_dir / "vdl_log_dir")
            except Exception:
                pass

        # Also directly set dataset paths on cfg dict (belt-and-suspenders).
        for ds_key in ("TrainDataset", "EvalDataset", "TestDataset"):
            if ds_key in cfg and isinstance(cfg[ds_key], dict):
                cfg[ds_key]["dataset_dir"] = dataset_dir
                cfg[ds_key]["image_dir"] = ""
                if ds_key == "TrainDataset":
                    cfg[ds_key]["anno_path"] = str(train_json)
                else:
                    cfg[ds_key]["anno_path"] = str(val_json)

        # ---- Pretrained / resume ----
        resume_training = _coerce_bool(add.get("resume_training", False), False)
        resume_job_id = add.get("resume_job_id")
        use_pretrained = _coerce_bool(
            add.get("use_pretrained", None),
            getattr(getattr(job, "parameters", None), "use_pretrained", True),
        )
        pretrained_model_path = add.get("pretrained_model_path") or getattr(
            getattr(job, "architecture", None), "pretrained_path", None
        )

        pretrain_weights: str | None = None
        resume_checkpoint: str | None = None

        if resume_training and resume_job_id:
            prev_dir = settings.training_dir / str(resume_job_id)
            # Look for model_final or best_model under the previous run
            prev_final = prev_dir / "model_final.pdparams"
            if not prev_final.exists():
                # Try best_model
                best_candidates = list(prev_dir.rglob("best_model/*.pdparams"))
                if best_candidates:
                    prev_final = best_candidates[0]
            if not prev_final.exists():
                raise ValidationError(f"Resume checkpoint not found for run_id={resume_job_id}")
            # PaddleDetection uses -r flag => checkpoint path without extension
            resume_checkpoint = str(prev_final).replace(".pdparams", "")
        elif use_pretrained and pretrained_model_path:
            resolved = Path(str(pretrained_model_path))
            if not resolved.is_absolute():
                for resolver in (resolve_temp_path, resolve_pretrain_path):
                    cand = resolver(str(pretrained_model_path))
                    if cand.exists():
                        resolved = cand
                        break
            if resolved.exists():
                pretrain_weights = str(resolved)
                # Strip .pdparams extension if present (PaddlePaddle convention)
                if pretrain_weights.endswith(".pdparams"):
                    pretrain_weights = pretrain_weights[: -len(".pdparams")]
            else:
                pretrain_weights = str(pretrained_model_path)

        # ---- Device ----
        device_value = str(getattr(job.parameters, "device", "auto") or "auto").strip().lower()
        use_gpu = True
        if device_value == "cpu":
            use_gpu = False
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif device_value not in ("auto", "", "default"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device_value.replace("cuda:", "")

        try:
            if use_gpu and not paddle.is_compiled_with_cuda():
                use_gpu = False
        except Exception:
            pass

        cfg["use_gpu"] = use_gpu

        # Also set device via paddle API (more reliable in newer versions)
        try:
            paddle.set_device("gpu" if use_gpu else "cpu")
        except Exception:
            pass

        # ---- Build Trainer ----
        try:
            trainer = PPTrainer(cfg, mode="train")
        finally:
            # Always restore monkey-patched functions even when trainer init fails.
            _restore_download_patches()

        # Load pretrained weights or resume
        if resume_checkpoint:
            trainer.resume_weights(resume_checkpoint)
        elif pretrain_weights:
            trainer.load_weights(pretrain_weights)

        # ---- Register callbacks for metrics & cancel ----
        last_cancel_check = {"t": 0.0}

        def _check_cancel() -> bool:
            now = time.time()
            if now - last_cancel_check["t"] < 2.0:
                return False
            last_cancel_check["t"] = now
            return bool(ctx.cancel_requested())

        # PaddleDetection Trainer supports hooks/callbacks via _callbacks dict.
        # We monkey-patch the Trainer's _compose_callback or register a custom
        # Callback that hooks into the training loop.
        try:
            from ppdet.engine.callbacks import Callback

            class _MetricsAndCancelCallback(Callback):
                """Custom callback for epoch metrics reporting and cancellation."""

                def __init__(self, pp_trainer: Any) -> None:
                    super().__init__(None)
                    self._trainer = pp_trainer

                @staticmethod
                def _extract_metrics(status: dict) -> Dict[str, float]:
                    metrics: Dict[str, float] = {}

                    # 1) Direct scalar fields from status.
                    direct_map = (
                        ("loss", "loss"),
                        ("loss_cls", "loss_cls"),
                        ("loss_iou", "loss_iou"),
                        ("loss_dfl", "loss_dfl"),
                        ("loss_obj", "loss_obj"),
                        ("learning_rate", "lr"),
                        ("lr", "lr"),
                        ("precision", "precision"),
                        ("recall", "recall"),
                        ("mAP", "mAP"),
                        ("AP50", "AP50"),
                        ("AP75", "AP75"),
                    )
                    for src, dst in direct_map:
                        val = status.get(src)
                        if val is None:
                            continue
                        fv = _safe_float(val)
                        if fv is not None:
                            metrics[dst] = fv

                    # 2) Training stats object/dict from PaddleDetection.
                    # NOTE: ppdet currently uses key "training_staus" (upstream typo).
                    ts_obj = (
                        status.get("training_staus")
                        or status.get("training_statis")
                        or status.get("training_stats")
                    )
                    ts_dict = None
                    if isinstance(ts_obj, dict):
                        ts_dict = ts_obj
                    elif ts_obj is not None and hasattr(ts_obj, "get"):
                        try:
                            got = ts_obj.get()
                            if isinstance(got, dict):
                                ts_dict = got
                        except Exception:
                            ts_dict = None

                    if isinstance(ts_dict, dict):
                        for k, v in ts_dict.items():
                            fv = _safe_float(v)
                            if fv is not None:
                                metrics[str(k)] = fv

                    # 3) Fallback: raw SmoothedValue meters.
                    if not metrics and ts_obj is not None and hasattr(ts_obj, "meters"):
                        try:
                            meters = getattr(ts_obj, "meters", None) or {}
                            if isinstance(meters, dict):
                                for k, meter in meters.items():
                                    for attr in ("avg", "global_avg", "median", "value"):
                                        if hasattr(meter, attr):
                                            fv = _safe_float(getattr(meter, attr))
                                            if fv is not None:
                                                metrics[str(k)] = fv
                                                break
                        except Exception:
                            pass

                    return metrics

                @staticmethod
                def _extract_eval_metrics_from_trainer(pp_trainer: Any) -> Dict[str, float]:
                    """
                    Read evaluation metrics from ppdet metric objects.

                    For COCO metrics, values are typically:
                    [mAP(0.50:0.95), AP50, AP75, ...].
                    """
                    metrics: Dict[str, float] = {}

                    def _set_metric(key: str, value: Any) -> None:
                        fv = _safe_float(value)
                        if fv is not None:
                            metrics[str(key)] = fv

                    metric_objs = getattr(pp_trainer, "_metrics", None)
                    if not isinstance(metric_objs, (list, tuple)):
                        return metrics

                    for metric_obj in metric_objs:
                        get_results = getattr(metric_obj, "get_results", None)
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

                            # Dict-like result: emit flattened keys.
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    _set_metric(f"eval/{group}_{k}", v)
                                continue

                            seq: list[Any] | None = None
                            if isinstance(value, (list, tuple)):
                                seq = list(value)
                            elif hasattr(value, "tolist"):
                                try:
                                    conv = value.tolist()
                                except Exception:
                                    conv = None
                                if isinstance(conv, (list, tuple)):
                                    seq = list(conv)

                            # Scalar result fallback.
                            if seq is None:
                                _set_metric(f"eval/{group}", value)
                                continue

                            # Most detector metrics provide first 3 entries as mAP/AP50/AP75.
                            if len(seq) >= 1:
                                _set_metric(f"eval/{group}_mAP", seq[0])
                                if group == "bbox":
                                    _set_metric("mAP", seq[0])
                                    _set_metric("metrics/mAP50-95(B)", seq[0])
                            if len(seq) >= 2:
                                _set_metric(f"eval/{group}_AP50", seq[1])
                                if group == "bbox":
                                    _set_metric("AP50", seq[1])
                                    _set_metric("mAP50", seq[1])
                                    _set_metric("metrics/mAP50(B)", seq[1])
                            if len(seq) >= 3:
                                _set_metric(f"eval/{group}_AP75", seq[2])
                                if group == "bbox":
                                    _set_metric("AP75", seq[2])

                    return metrics

                def on_epoch_end(self, status: dict) -> None:
                    epoch = int(status.get("epoch_id", 0))
                    mode = str(status.get("mode", "") or "").lower()
                    metrics = self._extract_metrics(status)
                    if mode == "eval":
                        metrics.update(self._extract_eval_metrics_from_trainer(self._trainer))
                    metrics = _apply_metric_aliases(metrics)
                    # Avoid overwriting existing epoch metrics with an empty payload.
                    if metrics:
                        ctx.upsert_epoch_metrics(epoch, metrics)

                    if _check_cancel():
                        raise SystemExit(0)

                def on_step_end(self, status: dict) -> None:
                    # Report once per epoch as soon as step 0 logs become available.
                    mode = str(status.get("mode", "") or "").lower()
                    epoch = int(status.get("epoch_id", 0))
                    step = int(status.get("step_id", -1))
                    if mode == "train" and step == 0:
                        metrics = _apply_metric_aliases(self._extract_metrics(status))
                        if metrics:
                            ctx.upsert_epoch_metrics(epoch, metrics)
                    if _check_cancel():
                        raise SystemExit(0)

            cancel_cb = _MetricsAndCancelCallback(trainer)

            # Inject callback into both callback containers.
            # ComposeCallback copies the callback list at construction time,
            # so appending only to trainer._callbacks is not enough.
            injected = False
            if hasattr(trainer, "_callbacks") and isinstance(trainer._callbacks, list):
                trainer._callbacks.append(cancel_cb)
                injected = True
            if hasattr(trainer, "_compose_callback") and hasattr(trainer._compose_callback, "_callbacks"):
                cb_list = getattr(trainer._compose_callback, "_callbacks")
                if isinstance(cb_list, list):
                    cb_list.append(cancel_cb)
                    injected = True
            if not injected:
                raise RuntimeError("Paddle callback container not found")
        except Exception:
            # If callback injection fails, training still proceeds
            # but without epoch metrics and cancel support
            pass

        # ---- Train ----
        trainer.train(validate=bool(eval_during_train))

        # ---- Evaluate (best-effort) ----
        try:
            trainer.evaluate()
        except Exception:
            pass

        # ---- Standardize output weights ----
        weights_dir = ctx.run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt, last_ckpt = _pick_paddle_checkpoint(ctx.run_dir)

        if last_ckpt and last_ckpt.exists():
            shutil.copy2(last_ckpt, weights_dir / "last.pdparams")
            # Also copy the companion .pdopt file if it exists
            opt_file = last_ckpt.with_suffix(".pdopt")
            if opt_file.exists():
                shutil.copy2(opt_file, weights_dir / "last.pdopt")

        if best_ckpt and best_ckpt.exists():
            shutil.copy2(best_ckpt, weights_dir / "best.pdparams")
            opt_file = best_ckpt.with_suffix(".pdopt")
            if opt_file.exists():
                shutil.copy2(opt_file, weights_dir / "best.pdopt")
