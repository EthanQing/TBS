from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from .runtime import coerce_metric_scalar


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


def load_yaml(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        obj = {}
    return obj if isinstance(obj, dict) else {}


def normalize_yolo_names(names_obj: Any, nc_obj: Any) -> list[str]:
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


def read_image_list(dataset_root: Path, spec: str) -> list[Path]:
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


def build_coco_from_yolo_list(
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


def summarize_ppdet_dataset_cfg(cfg: dict, dataset_key: str) -> Dict[str, Any]:
    node = cfg.get(dataset_key)
    if not isinstance(node, dict):
        return {"present": False}
    return {
        "present": True,
        "name": node.get("name"),
        "dataset_dir": node.get("dataset_dir"),
        "image_dir": node.get("image_dir"),
        "anno_path": node.get("anno_path"),
    }


__all__ = [
    "build_coco_from_yolo_list",
    "load_yaml",
    "normalize_yolo_names",
    "read_image_list",
    "summarize_ppdet_dataset_cfg",
]
