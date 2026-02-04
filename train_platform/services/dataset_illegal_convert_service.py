from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm isn't available
    def tqdm(iterable, **_kwargs):
        return iterable

from train_platform.services.file_service import FileService
from train_platform.utils.exceptions import ValidationError


# NOTE: Logic below mirrors test.py with minimal refactor for service use.


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass
class SliceInfo:
    idx: int
    x: int
    y: int
    w: int
    h: int
    is_negative: bool = False
    bboxes: List[BBox] = field(default_factory=list)


def points_to_bbox(points: list, label: str) -> BBox:
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return BBox(float(x_min), float(y_min), float(x_max), float(y_max), label)


def extract_label(raw_label: str, strategy, separator: str = "%") -> str:
    parts = [p.strip() for p in raw_label.split(separator) if p.strip()]
    if not parts:
        return raw_label
    if strategy == "full":
        return raw_label
    if strategy == "leaf":
        return parts[-1]
    if strategy == "root":
        return parts[0]
    if isinstance(strategy, int):
        return separator.join(parts[:strategy])
    return raw_label


def parse_annotations(cfg: dict) -> Tuple[List[BBox], Dict[str, int]]:
    json_path = cfg["annotation_path"]
    label_map = cfg["label_map"]
    label_strategy = cfg["label_strategy"]
    label_sep = cfg["label_separator"]
    min_prob = cfg["min_probability"]
    skip_hidden = cfg["skip_hidden"]
    skip_outside = cfg["skip_outside"]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bboxes: List[BBox] = []
    auto_map: Dict[str, int] = label_map.copy() if label_map else {}

    def get_cid(name: str) -> int:
        if name not in auto_map:
            auto_map[name] = len(auto_map)
        return auto_map[name]

    shapes = []
    if isinstance(data, dict) and "shapes" in data:
        shapes = data["shapes"]
    elif isinstance(data, list):
        shapes = data
    else:
        raise ValueError(f"Unrecognized json structure: {json_path}")

    for shape in shapes:
        if skip_hidden and shape.get("hidden", False):
            continue
        if skip_outside and shape.get("outside", False):
            continue
        if shape.get("probability", 1.0) < min_prob:
            continue

        pts = shape.get("points", [])
        if not pts or len(pts) < 2:
            continue

        raw_label = shape.get("label", "unknown")
        label_name = extract_label(raw_label, label_strategy, label_sep)
        stype = shape.get("shape_type", "polygon").lower()

        if stype == "rectangle" and len(pts) == 2:
            x_min = min(pts[0][0], pts[1][0])
            y_min = min(pts[0][1], pts[1][1])
            x_max = max(pts[0][0], pts[1][0])
            y_max = max(pts[0][1], pts[1][1])
            bbox = BBox(x_min, y_min, x_max, y_max, label_name)
        elif stype == "circle" and len(pts) >= 2:
            cx, cy = pts[0]
            ex, ey = pts[1]
            r = math.hypot(cx - ex, cy - ey)
            bbox = BBox(cx - r, cy - r, cx + r, cy + r, label_name)
        else:
            bbox = points_to_bbox(pts, label_name)

        if bbox.width <= 0 or bbox.height <= 0:
            continue

        bbox.class_id = get_cid(label_name)
        bboxes.append(bbox)

    return bboxes, auto_map


def plan_slices(
    img_w: int,
    img_h: int,
    bboxes: List[BBox],
    slice_size: int,
    overlap: float,
    padding: int,
    negative_ratio: float,
) -> List[SliceInfo]:
    if not bboxes:
        return []

    stride = max(1, int(slice_size * (1 - overlap)))
    grid_cols = max(1, math.ceil((img_w - slice_size) / stride) + 1)
    grid_rows = max(1, math.ceil((img_h - slice_size) / stride) + 1)

    active_cells = set()
    for bbox in bboxes:
        bx0 = max(0, bbox.x_min - padding)
        by0 = max(0, bbox.y_min - padding)
        bx1 = min(img_w, bbox.x_max + padding)
        by1 = min(img_h, bbox.y_max + padding)

        c0 = max(0, int(bx0 // stride))
        c1 = min(grid_cols - 1, int(bx1 // stride))
        r0 = max(0, int(by0 // stride))
        r1 = min(grid_rows - 1, int(by1 // stride))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                active_cells.add((r, c))

    def make_slice(r, c, idx, is_neg=False):
        x = min(c * stride, max(0, img_w - slice_size))
        y = min(r * stride, max(0, img_h - slice_size))
        w = min(slice_size, img_w - x)
        h = min(slice_size, img_h - y)
        return SliceInfo(idx=idx, x=x, y=y, w=w, h=h, is_negative=is_neg)

    seen = set()
    slices: List[SliceInfo] = []
    for (r, c) in sorted(active_cells):
        s = make_slice(r, c, len(slices), is_neg=False)
        key = (s.x, s.y)
        if key not in seen:
            seen.add(key)
            slices.append(s)
    n_positive = len(slices)

    if negative_ratio > 0:
        all_cells = {(r, c) for r in range(grid_rows) for c in range(grid_cols)}
        inactive = list(all_cells - active_cells)
        n_neg = max(1, int(n_positive * negative_ratio))
        if inactive:
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(inactive), size=min(n_neg, len(inactive)), replace=False)
            for ci in chosen:
                r, c = inactive[ci]
                s = make_slice(r, c, len(slices), is_neg=True)
                key = (s.x, s.y)
                if key not in seen:
                    seen.add(key)
                    slices.append(s)
    return slices


def assign_labels(
    slices: List[SliceInfo],
    bboxes: List[BBox],
    min_area_ratio: float,
    min_visibility: float,
    min_pixel_size: int,
) -> List[SliceInfo]:
    BUCKET = 1024
    bbox_buckets: Dict[Tuple[int, int], List[BBox]] = {}
    for bbox in bboxes:
        r0, r1 = int(bbox.y_min) // BUCKET, int(bbox.y_max) // BUCKET
        c0, c1 = int(bbox.x_min) // BUCKET, int(bbox.x_max) // BUCKET
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bbox_buckets.setdefault((r, c), []).append(bbox)

    for sl in tqdm(slices, desc="assign_labels", unit="slice"):
        sx0, sy0 = sl.x, sl.y
        sx1, sy1 = sl.x + sl.w, sl.y + sl.h

        r0, r1 = sy0 // BUCKET, sy1 // BUCKET
        c0, c1 = sx0 // BUCKET, sx1 // BUCKET

        seen_ids = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                for bbox in bbox_buckets.get((r, c), []):
                    bid = id(bbox)
                    if bid in seen_ids:
                        continue
                    seen_ids.add(bid)

                    ix0 = max(sx0, bbox.x_min)
                    iy0 = max(sy0, bbox.y_min)
                    ix1 = min(sx1, bbox.x_max)
                    iy1 = min(sy1, bbox.y_max)
                    if ix0 >= ix1 or iy0 >= iy1:
                        continue

                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    orig_area = bbox.area
                    if orig_area <= 0:
                        continue

                    if inter_area / orig_area < min_area_ratio:
                        continue

                    w_vis = (ix1 - ix0) / bbox.width if bbox.width > 0 else 0
                    h_vis = (iy1 - iy0) / bbox.height if bbox.height > 0 else 0
                    if w_vis < min_visibility or h_vis < min_visibility:
                        continue

                    lx0, ly0 = ix0 - sx0, iy0 - sy0
                    lx1, ly1 = ix1 - sx0, iy1 - sy0
                    if (lx1 - lx0) < min_pixel_size or (ly1 - ly0) < min_pixel_size:
                        continue

                    sl.bboxes.append(
                        BBox(
                            x_min=lx0,
                            y_min=ly0,
                            x_max=lx1,
                            y_max=ly1,
                            label=bbox.label,
                            class_id=bbox.class_id,
                        )
                    )
    return slices


def post_filter_slices(slices: List[SliceInfo], action: str = "discard") -> List[SliceInfo]:
    kept: List[SliceInfo] = []
    for sl in slices:
        has_labels = len(sl.bboxes) > 0
        was_positive = not sl.is_negative

        if was_positive and not has_labels:
            if action == "negative":
                sl.is_negative = True
                kept.append(sl)
        else:
            kept.append(sl)

    for i, sl in enumerate(kept):
        sl.idx = i
    return kept


def bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> str:
    cx = np.clip((bbox.x_min + bbox.x_max) / 2.0 / img_w, 0, 1)
    cy = np.clip((bbox.y_min + bbox.y_max) / 2.0 / img_h, 0, 1)
    w = np.clip(bbox.width / img_w, 0, 1)
    h = np.clip(bbox.height / img_h, 0, 1)
    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def read_window_rgb(dataset, x: int, y: int, w: int, h: int) -> np.ndarray:
    window = Window(col_off=x, row_off=y, width=w, height=h)
    n = dataset.count
    if n >= 3:
        data = dataset.read([1, 2, 3], window=window)
    else:
        band = dataset.read(1, window=window)
        data = np.stack([band, band, band], axis=0)
    data = np.moveaxis(data, 0, -1)
    if data.dtype != np.uint8:
        d = data.astype(np.float64)
        p2, p98 = np.percentile(d, 2), np.percentile(d, 98)
        if p98 > p2:
            d = (d - p2) / (p98 - p2) * 255.0
        else:
            d = d / (d.max() + 1e-8) * 255.0
        data = np.clip(d, 0, 255).astype(np.uint8)
    return data


def save_slices(cfg: dict, slices: List[SliceInfo]) -> Dict[str, int]:
    output_dir = cfg["output_dir"]
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    ext = cfg["output_format"].lower().strip(".")
    prefix = cfg["prefix"]
    quality = cfg["jpg_quality"]
    save_negative = cfg["negative_ratio"] > 0

    stats = {"total": 0, "with_labels": 0, "empty": 0, "total_labels": 0}

    with rasterio.open(cfg["image_path"]) as dataset:
        for sl in tqdm(slices, desc="save_slices", unit="slice"):
            has_labels = len(sl.bboxes) > 0

            if not has_labels and not save_negative:
                continue

            rgb = read_window_rgb(dataset, sl.x, sl.y, sl.w, sl.h)

            name = f"{prefix}_{sl.idx:06d}"
            img_path = os.path.join(img_dir, f"{name}.{ext}")
            pil_img = Image.fromarray(rgb)
            if ext in ("jpg", "jpeg"):
                pil_img.save(img_path, quality=quality)
            else:
                pil_img.save(img_path)

            lbl_path = os.path.join(lbl_dir, f"{name}.txt")
            with open(lbl_path, "w", encoding="utf-8") as f:
                for bbox in sl.bboxes:
                    f.write(bbox_to_yolo(bbox, sl.w, sl.h) + "\n")

            stats["total"] += 1
            if has_labels:
                stats["with_labels"] += 1
                stats["total_labels"] += len(sl.bboxes)
            else:
                stats["empty"] += 1

    return stats


class DatasetIllegalConvertService:
    DEFAULT_CONFIG: dict = {
        "slice_size": 1280,
        "overlap": 0.2,
        "padding": 64,
        "min_area_ratio": 0.3,
        "min_visibility": 0.15,
        "min_pixel_size": 5,
        "min_probability": 0.0,
        "skip_hidden": True,
        "skip_outside": True,
        "label_strategy": "leaf",
        "label_separator": "%",
        "negative_ratio": 0.1,
        "empty_positive_action": "discard",
        "output_format": "png",
        "jpg_quality": 95,
        "label_map": None,
    }

    _skip_dirs = {"labels", ".versions", ".thumbnails"}

    def convert_dataset(
        self,
        dataset_root: Path,
        *,
        label_strategy: str,
        label_level: Optional[int],
        label_separator: str,
    ) -> dict:
        root = Path(dataset_root).expanduser().resolve(strict=False)
        if not root.exists() or not root.is_dir():
            raise ValidationError("Dataset root not found for conversion")

        pairs, warnings = self._collect_pairs(root)
        if not pairs:
            raise ValidationError("No image/json pairs found for conversion")

        global_label_map: Dict[str, int] = {}
        processed = 0

        for image_path, json_path in pairs:
            cfg = dict(self.DEFAULT_CONFIG)
            cfg["image_path"] = str(image_path)
            cfg["annotation_path"] = str(json_path)
            cfg["output_dir"] = str(root)
            cfg["prefix"] = f"{image_path.stem}_slice"
            cfg["label_separator"] = label_separator or cfg["label_separator"]
            if label_strategy == "level":
                lvl = int(label_level or 0)
                if lvl < 1:
                    raise ValidationError("label_level must be >= 1 when label_strategy=level")
                cfg["label_strategy"] = int(lvl)
            else:
                cfg["label_strategy"] = str(label_strategy or cfg["label_strategy"])
            cfg["label_map"] = global_label_map or None

            stats, global_label_map, slicing_meta = self._run_single(cfg)
            info_path = root / f"slicing_info_{image_path.stem}.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(slicing_meta, f, ensure_ascii=False, indent=2)
            processed += 1

        # Delete original images and JSON files after successful conversion
        for image_path, json_path in pairs:
            try:
                if image_path.exists():
                    image_path.unlink()
            except Exception:
                pass  # Ignore errors when deleting individual files
            try:
                if json_path.exists():
                    json_path.unlink()
            except Exception:
                pass

        classes_path = root / "classes.txt"
        with open(classes_path, "w", encoding="utf-8") as f:
            for name, _cid in sorted(global_label_map.items(), key=lambda x: x[1]):
                f.write(name + "\n")

        FileService()._create_yolo_data_yaml(root, root / "data.yaml")

        return {
            "pairs_total": len(pairs),
            "pairs_processed": processed,
            "warnings": warnings,
        }

    def _collect_pairs(self, root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
        image_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        image_by_stem: Dict[str, Path] = {}
        json_by_stem: Dict[str, Path] = {}

        for cur, dirnames, filenames in os.walk(root):
            cur_p = Path(cur)
            rel = cur_p.relative_to(root)
            if rel.parts and rel.parts[0].lower() in self._skip_dirs:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in self._skip_dirs]

            for fname in filenames:
                p = cur_p / fname
                ext = p.suffix.lower()
                if ext in image_exts:
                    stem = p.stem
                    prev = image_by_stem.get(stem)
                    if prev is None or image_exts.index(ext) < image_exts.index(prev.suffix.lower()):
                        image_by_stem[stem] = p
                elif ext == ".json":
                    stem = p.stem
                    if stem not in json_by_stem:
                        json_by_stem[stem] = p

        common = sorted(set(image_by_stem.keys()) & set(json_by_stem.keys()))
        pairs = [(image_by_stem[s], json_by_stem[s]) for s in common]

        warnings: list[str] = []
        extra_imgs = sorted(set(image_by_stem.keys()) - set(json_by_stem.keys()))
        extra_json = sorted(set(json_by_stem.keys()) - set(image_by_stem.keys()))
        if extra_imgs:
            warnings.append(f"Unmatched images: {len(extra_imgs)}")
        if extra_json:
            warnings.append(f"Unmatched json: {len(extra_json)}")

        return pairs, warnings

    def _run_single(self, cfg: dict) -> tuple[dict, Dict[str, int], dict]:
        with rasterio.open(cfg["image_path"]) as ds:
            img_w, img_h = ds.width, ds.height
            n_bands, dtype = ds.count, ds.dtypes[0]

        bboxes, label_map = parse_annotations(cfg)
        if not bboxes:
            raise ValidationError(f"No valid annotations in {cfg['annotation_path']}")

        oob = 0
        for b in bboxes:
            if b.x_max <= 0 or b.y_max <= 0 or b.x_min >= img_w or b.y_min >= img_h:
                oob += 1
            b.x_min = max(0.0, b.x_min)
            b.y_min = max(0.0, b.y_min)
            b.x_max = min(float(img_w), b.x_max)
            b.y_max = min(float(img_h), b.y_max)
        bboxes = [b for b in bboxes if b.width > 0 and b.height > 0]
        if not bboxes:
            raise ValidationError(f"No valid annotations after clipping for {cfg['annotation_path']}")

        slices = plan_slices(
            img_w,
            img_h,
            bboxes,
            slice_size=cfg["slice_size"],
            overlap=cfg["overlap"],
            padding=cfg["padding"],
            negative_ratio=cfg["negative_ratio"],
        )
        if not slices:
            raise ValidationError(f"No slices planned for {cfg['annotation_path']}")

        slices = assign_labels(
            slices,
            bboxes,
            min_area_ratio=cfg["min_area_ratio"],
            min_visibility=cfg["min_visibility"],
            min_pixel_size=cfg["min_pixel_size"],
        )

        slices = post_filter_slices(
            slices,
            action=cfg["empty_positive_action"],
        )

        save_stats = save_slices(cfg, slices)

        save_negative = cfg["negative_ratio"] > 0
        meta = {
            "source_image": os.path.basename(cfg["image_path"]),
            "source_size": [img_w, img_h],
            "source_bands": n_bands,
            "source_dtype": str(dtype),
            "config": {
                "slice_size": cfg["slice_size"],
                "overlap": cfg["overlap"],
                "min_area_ratio": cfg["min_area_ratio"],
                "label_strategy": str(cfg["label_strategy"]),
                "empty_positive_action": cfg["empty_positive_action"],
            },
            "label_map": label_map,
            "slices": [
                {
                    "name": f"{cfg['prefix']}_{sl.idx:06d}",
                    "x": sl.x,
                    "y": sl.y,
                    "w": sl.w,
                    "h": sl.h,
                    "num_labels": len(sl.bboxes),
                    "is_negative": sl.is_negative,
                }
                for sl in slices
                if len(sl.bboxes) > 0 or save_negative
            ],
        }

        return save_stats, label_map, meta
