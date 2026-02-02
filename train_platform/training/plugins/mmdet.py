from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from train_platform.core.config import settings
from train_platform.training.plugins.base import TrainContext
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.path_utils import resolve_pretrain_path, resolve_temp_path


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
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _load_yaml(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        obj = {}
    return obj if isinstance(obj, dict) else {}


def _normalize_yolo_names(names_obj: Any, nc_obj: Any) -> list[str]:
    if isinstance(names_obj, list):
        out = [str(x) for x in names_obj if str(x).strip()]
        return out
    if isinstance(names_obj, dict):
        out: list[str] = []
        try:
            keys = sorted(int(k) for k in names_obj.keys())
            for i in keys:
                out.append(str(names_obj.get(i) or names_obj.get(str(i)) or f"class_{i}"))
            return out
        except Exception:
            return [str(v) for v in names_obj.values() if str(v).strip()]

    nc = _safe_int(nc_obj)
    if nc is not None and nc > 0:
        return [f"class_{i}" for i in range(int(nc))]
    return []


def _read_image_list(dataset_root: Path, spec: str) -> list[Path]:
    """
    Read YOLO train/val spec which can be:
    - a .txt file listing image paths (absolute or relative)
    - a directory containing images
    """
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
    """
    Best-effort mapping from image path -> YOLO label file.

    Common patterns:
    - images/.../foo.jpg -> labels/.../foo.txt
    - foo.jpg -> labels/foo.txt
    """
    try:
        rel = image_abs.resolve(strict=False).relative_to(dataset_root.resolve(strict=False))
    except Exception:
        rel = image_abs.name  # type: ignore[assignment]

    rel_p = Path(rel)
    parts = list(rel_p.parts)

    # (1) Replace first "images" segment with "labels"
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            cand = (dataset_root / Path(*parts)).with_suffix(".txt")
            return cand

    # (2) Fallback: labels/<same relative path>.txt
    return (dataset_root / "labels" / rel_p).with_suffix(".txt")


def _build_coco_from_yolo_list(
    dataset_root: Path,
    image_paths: Iterable[Path],
    class_names: list[str],
    *,
    output_json_path: Path,
) -> Path:
    """
    Build a COCO annotation json from a list of image paths + YOLO txt labels.

    COCO category ids are 1-based (standard).
    """
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow (PIL) is required for COCO conversion but not installed") from e

    cats = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(class_names)]
    coco: Dict[str, Any] = {"images": [], "annotations": [], "categories": cats, "licenses": [], "info": {}}

    img_id = 1
    ann_id = 1

    # Deterministic ordering
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
                x_c = _safe_float(parts[1])
                y_c = _safe_float(parts[2])
                w_n = _safe_float(parts[3])
                h_n = _safe_float(parts[4])
                if cid is None or x_c is None or y_c is None or w_n is None or h_n is None:
                    continue

                # YOLO normalized -> COCO absolute bbox
                w_abs = max(0.0, float(w_n) * float(width))
                h_abs = max(0.0, float(h_n) * float(height))
                x_min = (float(x_c) * float(width)) - (w_abs / 2.0)
                y_min = (float(y_c) * float(height)) - (h_abs / 2.0)
                x_min = max(0.0, float(x_min))
                y_min = max(0.0, float(y_min))

                coco_cid = int(cid) + 1  # 1-based category ids
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": coco_cid,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w_abs, 2), round(h_abs, 2)],
                        "area": round(w_abs * h_abs, 2),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1

        img_id += 1

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return output_json_path


def _unwrap_dataset_cfg(ds_cfg: Any) -> Any:
    """
    Many MMEngine configs wrap the dataset like RepeatDataset(dataset=dict(...)).
    Walk down a few levels to find the innermost dataset dict.
    """
    cur = ds_cfg
    for _ in range(6):
        if isinstance(cur, dict) and isinstance(cur.get("dataset"), dict):
            cur = cur["dataset"]
            continue
        break
    return cur


def _set_mmdet_dataset(
    cfg: Any,
    *,
    split: str,
    ann_file: str,
    data_root: str,
    classes: list[str],
) -> None:
    dl_key = f"{split}_dataloader"
    ev_key = f"{split}_evaluator"

    dl = cfg.get(dl_key)
    if isinstance(dl, dict):
        if "dataset" in dl:
            ds = _unwrap_dataset_cfg(dl.get("dataset"))
            if isinstance(ds, dict):
                ds["data_root"] = str(data_root)
                ds["ann_file"] = str(ann_file)
                ds.setdefault("data_prefix", {})
                if isinstance(ds["data_prefix"], dict):
                    ds["data_prefix"]["img"] = ""  # COCO file_name is relative to data_root
                ds["metainfo"] = {"classes": tuple(classes)}

    ev = cfg.get(ev_key)
    if isinstance(ev, dict):
        # Evaluator may need the same ann_file to compute metrics.
        ev["ann_file"] = str(ann_file)
    elif isinstance(ev, list):
        for item in ev:
            if isinstance(item, dict):
                item["ann_file"] = str(ann_file)


def _pick_checkpoint(work_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Return (best, last) checkpoints.

    - prefer best_*.pth as best
    - prefer latest.pth as last
    """
    best: Path | None = None
    last: Path | None = None

    try:
        cand_latest = (work_dir / "latest.pth").resolve(strict=False)
        if cand_latest.exists():
            last = cand_latest
    except Exception:
        last = None

    try:
        best_cands = list(work_dir.rglob("best_*.pth"))
        best_cands = [p for p in best_cands if p.is_file()]
        if best_cands:
            best = max(best_cands, key=lambda p: p.stat().st_mtime)
    except Exception:
        best = None

    if last is None:
        try:
            all_pths = list(work_dir.rglob("*.pth"))
            all_pths = [p for p in all_pths if p.is_file()]
            if all_pths:
                last = max(all_pths, key=lambda p: p.stat().st_mtime)
        except Exception:
            last = None

    if best is None:
        best = last

    return best, last


class MMDetTrainer:
    name = "mmdet"

    def can_handle(self, model_family: str) -> bool:
        mf = (model_family or "").strip().lower()
        return any(x in mf for x in ("mmdet", "mmdetection", "rtmdet"))

    def run(self, ctx: TrainContext) -> None:
        # Import lazily so the backend can run without MMDetection installed.
        try:
            from mmengine.config import Config
            from mmengine.runner import Runner
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "MMEngine/MMDetection not installed. Install mmengine + mmdet (and mmcv) to use mmdet training."
            ) from e

        try:
            from mmdet.utils import register_all_modules
        except Exception as e:  # pragma: no cover
            raise RuntimeError("MMDetection not installed (missing mmdet)") from e

        register_all_modules(init_default_scope=True)

        job = ctx.job
        add = getattr(getattr(job, "parameters", None), "additional_params", None) or {}
        arch_defaults = getattr(getattr(job, "architecture", None), "default_params", None) or {}

        config_path = add.get("config_path") or arch_defaults.get("config_path") or add.get("config")
        if not config_path:
            raise ValidationError(
                "mmdet requires additional_params.config_path (path to an MMDetection config .py)"
            )

        cfg_path = Path(str(config_path))
        if not cfg_path.is_absolute():
            # Allow configs to be stored under pretrain_models or temp.
            cand = resolve_temp_path(str(config_path))
            if cand.exists():
                cfg_path = cand
            else:
                cand = resolve_pretrain_path(str(config_path))
                if cand.exists():
                    cfg_path = cand
                else:
                    cfg_path = (Path.cwd() / cfg_path).resolve(strict=False)

        if not cfg_path.exists():
            raise ValidationError(f"MMDet config not found: {cfg_path}")

        # Dataset: read names + train/val from a YOLO dataset yaml.
        data_yaml = find_yolo_dataset_yaml(ctx.dataset_path)
        if data_yaml is None or not data_yaml.exists():
            raise ValidationError("Dataset YAML not found in dataset; cannot derive train/val and class names")
        data_cfg = _load_yaml(data_yaml)

        class_names = _normalize_yolo_names(data_cfg.get("names"), data_cfg.get("nc"))
        if not class_names:
            class_names = ["class_0"]

        train_spec = str(data_cfg.get("train") or "").strip()
        val_spec = str(data_cfg.get("val") or "").strip()
        if not train_spec or not val_spec:
            raise ValidationError("Dataset YAML missing train/val; please split dataset first")

        train_images = _read_image_list(ctx.dataset_path, train_spec)
        val_images = _read_image_list(ctx.dataset_path, val_spec)
        if not train_images or not val_images:
            raise ValidationError("train/val image lists are empty; please verify dataset split outputs")

        # Convert YOLO -> COCO under run_dir/coco/*.json for MMDet consumption.
        coco_dir = (ctx.run_dir / "coco").resolve(strict=False)
        coco_dir.mkdir(parents=True, exist_ok=True)

        train_json = _build_coco_from_yolo_list(
            ctx.dataset_path,
            train_images,
            class_names,
            output_json_path=coco_dir / "train.json",
        )
        val_json = _build_coco_from_yolo_list(
            ctx.dataset_path,
            val_images,
            class_names,
            output_json_path=coco_dir / "val.json",
        )

        cfg = Config.fromfile(str(cfg_path))
        cfg.work_dir = str(ctx.run_dir)

        # Basic hyper-params (best-effort; depends on config structure).
        try:
            if isinstance(cfg.get("train_cfg"), dict):
                cfg.train_cfg["max_epochs"] = int(job.parameters.epochs)
        except Exception:
            pass
        try:
            if isinstance(cfg.get("train_dataloader"), dict):
                cfg.train_dataloader["batch_size"] = int(job.parameters.batch_size)
                cfg.train_dataloader["num_workers"] = int(getattr(job.parameters, "workers", 8) or 8)
        except Exception:
            pass
        try:
            ow = cfg.get("optim_wrapper")
            if isinstance(ow, dict) and isinstance(ow.get("optimizer"), dict):
                lr = _safe_float(getattr(job.parameters, "learning_rate", None))
                if lr is not None:
                    ow["optimizer"]["lr"] = float(lr)
        except Exception:
            pass

        # Pretrained / resume
        resume_training = _coerce_bool(add.get("resume_training", False), False)
        resume_job_id = add.get("resume_job_id")
        use_pretrained = _coerce_bool(
            add.get("use_pretrained", None),
            getattr(getattr(job, "parameters", None), "use_pretrained", True),
        )
        pretrained_model_path = add.get("pretrained_model_path") or getattr(getattr(job, "architecture", None), "pretrained_path", None)

        if resume_training and resume_job_id:
            # Resume from the previous run's last checkpoint if present.
            prev_last = settings.training_dir / str(resume_job_id) / "weights" / "last.pth"
            if not prev_last.exists():
                prev_last = settings.training_dir / str(resume_job_id) / "latest.pth"
            if not prev_last.exists():
                raise ValidationError(f"resume checkpoint not found for run_id={resume_job_id}")
            cfg.resume = True
            cfg.load_from = None
            cfg.resume_from = str(prev_last)
        elif use_pretrained and pretrained_model_path:
            resolved = Path(str(pretrained_model_path))
            if not resolved.is_absolute():
                cand = resolve_temp_path(str(pretrained_model_path))
                if cand.exists():
                    resolved = cand
                else:
                    cand = resolve_pretrain_path(str(pretrained_model_path))
                    if cand.exists():
                        resolved = cand
            if resolved.exists():
                cfg.load_from = str(resolved)

        # Patch dataset paths for COCO jsons.
        _set_mmdet_dataset(cfg, split="train", ann_file=str(train_json), data_root=str(ctx.dataset_path), classes=class_names)
        _set_mmdet_dataset(cfg, split="val", ann_file=str(val_json), data_root=str(ctx.dataset_path), classes=class_names)
        _set_mmdet_dataset(cfg, split="test", ann_file=str(val_json), data_root=str(ctx.dataset_path), classes=class_names)

        # Device selection (best-effort).
        device_value = str(getattr(job.parameters, "device", "auto") or "auto").strip().lower()
        if device_value in ("", "default", "auto"):
            pass
        elif device_value == "cpu":
            cfg.device = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            # Treat as a CUDA_VISIBLE_DEVICES value (e.g. "0" / "0,1").
            os.environ["CUDA_VISIBLE_DEVICES"] = device_value.replace("cuda:", "")

        # Hook: cancel + epoch progress (best-effort metrics extraction).
        try:
            from mmengine.hooks import Hook

            class _CancelAndMetricsHook(Hook):  # type: ignore[misc]
                def __init__(self) -> None:
                    self._last_cancel_check = 0.0

                def after_train_epoch(self, runner) -> None:  # type: ignore[override]
                    epoch = int(getattr(runner, "epoch", 0))
                    metrics: Dict[str, float] = {}
                    try:
                        mh = getattr(runner, "message_hub", None)
                        log_scalars = getattr(mh, "log_scalars", None) if mh is not None else None
                        if isinstance(log_scalars, dict):
                            for k, v in log_scalars.items():
                                # Try common scalar representations.
                                fv = None
                                if isinstance(v, (int, float)):
                                    fv = float(v)
                                elif isinstance(v, (list, tuple)) and v:
                                    fv = _safe_float(v[-1])
                                else:
                                    fv = _safe_float(getattr(v, "value", None))
                                if fv is not None:
                                    metrics[str(k)] = float(fv)
                    except Exception:
                        metrics = {}

                    # Always upsert at least an empty dict so progress moves.
                    ctx.upsert_epoch_metrics(epoch, metrics)

                    now = time.time()
                    if now - self._last_cancel_check > 2.0:
                        self._last_cancel_check = now
                        if ctx.cancel_requested():
                            raise SystemExit(0)

                def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # type: ignore[override]
                    now = time.time()
                    if now - self._last_cancel_check > 2.0:
                        self._last_cancel_check = now
                        if ctx.cancel_requested():
                            raise SystemExit(0)

            cancel_hook = _CancelAndMetricsHook()
        except Exception:
            cancel_hook = None

        runner = Runner.from_cfg(cfg)
        if cancel_hook is not None:
            try:
                runner.register_hook(cancel_hook, priority="HIGHEST")
            except Exception:
                try:
                    runner.register_hook(cancel_hook)
                except Exception:
                    pass

        runner.train()

        # Standardize artifacts to run_dir/weights so the rest of the system can find them.
        best_ckpt, last_ckpt = _pick_checkpoint(ctx.run_dir)
        weights_dir = (ctx.run_dir / "weights").resolve(strict=False)
        weights_dir.mkdir(parents=True, exist_ok=True)

        if last_ckpt and last_ckpt.exists():
            shutil.copy2(last_ckpt, weights_dir / "last.pth")
        if best_ckpt and best_ckpt.exists():
            shutil.copy2(best_ckpt, weights_dir / "best.pth")
