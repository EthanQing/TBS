from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

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


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _emit_progress(progress_cb: ProgressCallback, **payload: Any) -> None:
    if not callable(progress_cb):
        return
    try:
        progress_cb(payload)
    except Exception:
        # Progress reporting must never break conversion flow.
        return


def points_to_bbox(points: list, label: str) -> BBox:
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return BBox(float(x_min), float(y_min), float(x_max), float(y_max), label)


def extract_label(raw_label: str, strategy, separator: str = "%") -> str:
    parts = [p.strip() for p in raw_label.split(separator) if p.strip()]
    if not parts:
        return ""
    if strategy == "full":
        return separator.join(parts)  # rejoin stripped parts to normalise whitespace
    if strategy == "leaf":
        return parts[-1]
    if strategy == "root":
        return parts[0]
    if isinstance(strategy, int):
        return separator.join(parts[:strategy])
    return raw_label


def _normalize_label_key(value: Any) -> str:
    """
    Normalize labels for stable mapping:
    - trim whitespace
    - normalize full-width percent
    - strip zero-width characters
    """
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\uFF05", "%").replace("\u3000", " ")
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(ch, "")
    return s.strip()


def parse_annotations(cfg: dict) -> Tuple[List[BBox], Dict[str, int]]:
    json_path = cfg["annotation_path"]
    label_map = cfg["label_map"]
    label_strategy = cfg["label_strategy"]
    label_sep = cfg["label_separator"]
    min_prob = cfg["min_probability"]
    skip_hidden = cfg["skip_hidden"]
    skip_outside = cfg["skip_outside"]
    label_strategy_norm = str(label_strategy or "").strip().lower()

    raw_mapping = cfg.get("label_mapping")
    label_mapping: Optional[Dict[str, str]] = None
    label_mapping_norm: Optional[Dict[str, str]] = None
    if isinstance(raw_mapping, dict) and raw_mapping:
        label_mapping = {str(k): v for k, v in raw_mapping.items()}
        label_mapping_norm = {}
        for k, v in label_mapping.items():
            nk = _normalize_label_key(k)
            if nk in label_mapping_norm and label_mapping_norm[nk] != v:
                raise ValidationError(f"Conflicting label mappings for normalized key: {nk}")
            label_mapping_norm[nk] = v
    missing_labels: set[str] = set()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with open(json_path, "r", encoding="gbk", errors="ignore") as f:
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

        raw_label = str(shape.get("label", "unknown"))
        raw_label_stripped = raw_label.strip()

        # Skip annotations with empty / whitespace-only labels
        if not raw_label_stripped:
            continue

        if label_mapping is not None:
            mapped_label = None
            if raw_label_stripped in label_mapping:
                mapped_label = label_mapping.get(raw_label_stripped)
            elif label_mapping_norm is not None:
                norm_key = _normalize_label_key(raw_label_stripped)
                if norm_key in label_mapping_norm:
                    mapped_label = label_mapping_norm.get(norm_key)

            if mapped_label == "__DISCARD__" or mapped_label == "":
                continue
            if mapped_label is not None:
                raw_label_stripped = str(mapped_label).strip()
            elif label_strategy_norm == "mapping":
                missing_labels.add(raw_label_stripped)
                continue

        label_name = extract_label(raw_label_stripped, label_strategy, label_sep).strip()
        if not label_name:
            continue
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

    if missing_labels:
        sample = ", ".join(list(missing_labels)[:10])
        raise ValidationError(
            f"Missing label mappings for {len(missing_labels)} labels in {json_path}: {sample}"
        )

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
            rng = np.random.default_rng()
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
    *,
    progress_cb: ProgressCallback = None,
    progress_context: Optional[Dict[str, Any]] = None,
) -> List[SliceInfo]:
    BUCKET = 1024
    bbox_buckets: Dict[Tuple[int, int], List[BBox]] = {}
    for bbox in bboxes:
        r0, r1 = int(bbox.y_min) // BUCKET, int(bbox.y_max) // BUCKET
        c0, c1 = int(bbox.x_min) // BUCKET, int(bbox.x_max) // BUCKET
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bbox_buckets.setdefault((r, c), []).append(bbox)

    total_slices = len(slices)
    base = dict(progress_context or {})
    for idx, sl in enumerate(slices, start=1):
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
        _emit_progress(
            progress_cb,
            **base,
            phase="assign_labels",
            current_slice_processed=int(idx),
            current_slice_total=int(total_slices),
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


# Global percentile range cache: keyed by dataset path to ensure consistent
# stretching across all slices of the same source image.
_global_stretch_cache: Dict[str, Tuple[float, float]] = {}


def read_window_rgb(
    dataset, x: int, y: int, w: int, h: int,
    global_p2: Optional[float] = None,
    global_p98: Optional[float] = None,
) -> np.ndarray:
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
        p2 = global_p2 if global_p2 is not None else float(np.percentile(d, 2))
        p98 = global_p98 if global_p98 is not None else float(np.percentile(d, 98))
        if p98 > p2:
            d = (d - p2) / (p98 - p2) * 255.0
        else:
            d = np.full_like(d, 127.0)
        data = np.clip(d, 0, 255).astype(np.uint8)
    return data


def save_slices(
    cfg: dict,
    slices: List[SliceInfo],
    *,
    progress_cb: ProgressCallback = None,
    progress_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
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

    total_slices = len(slices)
    base = dict(progress_context or {})

    with rasterio.open(cfg["image_path"]) as dataset:
        for idx, sl in enumerate(slices, start=1):
            has_labels = len(sl.bboxes) > 0

            if not has_labels and not save_negative:
                _emit_progress(
                    progress_cb,
                    **base,
                    phase="save_slices",
                    current_slice_processed=int(idx),
                    current_slice_total=int(total_slices),
                )
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

            _emit_progress(
                progress_cb,
                **base,
                phase="save_slices",
                current_slice_processed=int(idx),
                current_slice_total=int(total_slices),
            )

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
        label_mapping: Optional[dict] = None,
        slice_config: Optional[dict] = None,
        progress_cb: ProgressCallback = None,
    ) -> dict:
        root = Path(dataset_root).expanduser().resolve(strict=False)
        if not root.exists() or not root.is_dir():
            raise ValidationError("Dataset root not found for conversion")

        pairs, warnings = self._collect_pairs(root)
        if not pairs:
            raise ValidationError("No image/json pairs found for conversion")

        global_label_map: Dict[str, int] = {}
        processed = 0
        processed_images = 0
        skipped_files: List[str] = []
        processed_pairs: List[Tuple[Path, Path]] = []
        total_images = len(pairs)

        _emit_progress(
            progress_cb,
            overall_processed_images=0,
            overall_total_images=int(total_images),
            current_image_index=0,
            current_image_name="",
            phase="scanning",
            current_slice_processed=0,
            current_slice_total=0,
            message="conversion started",
        )

        for idx, (image_path, json_path) in enumerate(pairs, start=1):
            current_name = str(image_path.name)
            _emit_progress(
                progress_cb,
                overall_processed_images=int(processed_images),
                overall_total_images=int(total_images),
                current_image_index=int(idx),
                current_image_name=current_name,
                phase="scanning",
                current_slice_processed=0,
                current_slice_total=0,
                message="image started",
            )
            cfg = dict(self.DEFAULT_CONFIG)
            # Apply user-specified slice/crop overrides
            if isinstance(slice_config, dict):
                for key in ("slice_size", "overlap", "padding", "min_area_ratio",
                             "min_visibility", "min_pixel_size", "negative_ratio",
                             "empty_positive_action"):
                    if key in slice_config and slice_config[key] is not None:
                        cfg[key] = slice_config[key]
            cfg["image_path"] = str(image_path)
            cfg["annotation_path"] = str(json_path)
            cfg["output_dir"] = str(root)
            # Use relative path (with _ replacing separators) to avoid collisions
            # when multiple subdirectories contain images with the same stem.
            rel_stem = str(image_path.relative_to(root).with_suffix("")).replace(os.sep, "_").replace("/", "_")
            cfg["prefix"] = f"{rel_stem}_slice"
            cfg["label_separator"] = label_separator or cfg["label_separator"]
            if label_strategy == "level":
                lvl = int(label_level or 0)
                if lvl < 1:
                    raise ValidationError("label_level must be >= 1 when label_strategy=level")
                cfg["label_strategy"] = int(lvl)
            else:
                cfg["label_strategy"] = str(label_strategy or cfg["label_strategy"])
            cfg["label_map"] = global_label_map if global_label_map else {}
            cfg["label_mapping"] = label_mapping

            try:
                stats, global_label_map, slicing_meta = self._run_single(
                    cfg,
                    progress_cb=progress_cb,
                    progress_context={
                        "overall_processed_images": int(processed_images),
                        "overall_total_images": int(total_images),
                        "current_image_index": int(idx),
                        "current_image_name": current_name,
                    },
                )
            except ValidationError as e:
                # Skip files with no valid annotations instead of aborting
                skipped_files.append(f"{json_path.name}: {e}")
                warnings.append(f"Skipped {json_path.name}: {e}")
                processed_images += 1
                _emit_progress(
                    progress_cb,
                    overall_processed_images=int(processed_images),
                    overall_total_images=int(total_images),
                    current_image_index=int(idx),
                    current_image_name=current_name,
                    phase="skipped",
                    current_slice_processed=0,
                    current_slice_total=0,
                    message=str(e),
                )
                continue

            info_path = root / f"slicing_info_{image_path.stem}.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(slicing_meta, f, ensure_ascii=False, indent=2)
            processed += 1
            processed_images += 1
            processed_pairs.append((image_path, json_path))
            _emit_progress(
                progress_cb,
                overall_processed_images=int(processed_images),
                overall_total_images=int(total_images),
                current_image_index=int(idx),
                current_image_name=current_name,
                phase="finalizing",
                current_slice_processed=int(len(slicing_meta.get("slices") or [])),
                current_slice_total=int(len(slicing_meta.get("slices") or [])),
                message="image completed",
            )

        if processed == 0:
            skipped_summary = "; ".join(skipped_files[:5])
            raise ValidationError(
                f"All {len(pairs)} image/json pairs failed conversion. "
                f"Details: {skipped_summary}"
            )

        # Delete original images and JSON files only for successfully processed pairs
        for image_path, json_path in processed_pairs:
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

        # Filter out empty / whitespace-only class names and re-index
        sorted_classes = [
            name for name, _cid
            in sorted(global_label_map.items(), key=lambda x: x[1])
            if name and name.strip()
        ]
        classes_path = root / "classes.txt"
        with open(classes_path, "w", encoding="utf-8") as f:
            for name in sorted_classes:
                f.write(name + "\n")

        FileService()._create_yolo_data_yaml(root, root / "data.yaml")

        _emit_progress(
            progress_cb,
            overall_processed_images=int(processed_images),
            overall_total_images=int(total_images),
            current_image_index=int(total_images),
            current_image_name="",
            phase="done",
            current_slice_processed=0,
            current_slice_total=0,
            message="conversion completed",
        )

        return {
            "pairs_total": len(pairs),
            "pairs_processed": processed,
            "pairs_skipped": len(skipped_files),
            "skipped_details": skipped_files,
            "warnings": warnings,
        }

    def extract_dataset_labels(self, root: Path) -> list[str]:
        labels = set()
        for cur, dirnames, filenames in os.walk(root):
            cur_p = Path(cur)
            for fname in filenames:
                if fname.lower().endswith(".json"):
                    try:
                        with open(cur_p / fname, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        shapes = []
                        if isinstance(data, dict) and "shapes" in data:
                            shapes = data["shapes"]
                        elif isinstance(data, list):
                            shapes = data
                        for shape in shapes:
                            if isinstance(shape, dict) and "label" in shape:
                                labels.add(str(shape["label"]).strip())
                    except Exception:
                        pass
        return sorted(list(labels))

    def apply_label_mapping(
        self,
        root: Path,
        label_mapping: dict,
        *,
        strict: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Apply label mapping directly into LabelMe JSON files.

        - "__DISCARD__" or empty target removes the shape.
        - If strict=True, any label without a mapping raises ValidationError.
        """
        root = Path(root).expanduser().resolve(strict=False)
        if not root.exists() or not root.is_dir():
            raise ValidationError("Dataset root not found for label mapping")
        if not isinstance(label_mapping, dict) or not label_mapping:
            raise ValidationError("label_mapping is required")

        # Normalize mapping keys for stable matching.
        label_mapping_norm: Dict[str, str] = {}
        for k, v in label_mapping.items():
            nk = _normalize_label_key(k)
            if not nk:
                continue
            if nk in label_mapping_norm and label_mapping_norm[nk] != v:
                raise ValidationError(f"Conflicting label mappings for normalized key: {nk}")
            label_mapping_norm[nk] = v

        def _iter_json_files():
            for cur, dirnames, filenames in os.walk(root):
                cur_p = Path(cur)
                rel = cur_p.relative_to(root)
                if rel.parts and rel.parts[0].lower() in self._skip_dirs:
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames if d.lower() not in self._skip_dirs]
                for fname in filenames:
                    if fname.lower().endswith(".json"):
                        yield cur_p / fname

        missing_labels: set[str] = set()

        # Materialize file list once to avoid TOCTOU between validation and update passes.
        json_files: list[Path] = list(_iter_json_files())

        # First pass: detect missing mappings (avoid partial updates).
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            shapes = None
            if isinstance(data, dict) and isinstance(data.get("shapes"), list):
                shapes = data.get("shapes")
            elif isinstance(data, list):
                shapes = data
            if not shapes:
                continue

            for shape in shapes:
                if not isinstance(shape, dict) or "label" not in shape:
                    continue
                raw_label = str(shape.get("label", ""))
                raw_label_stripped = raw_label.strip()
                if not raw_label_stripped:
                    continue

                mapped_label = None
                if raw_label_stripped in label_mapping:
                    mapped_label = label_mapping.get(raw_label_stripped)
                else:
                    norm_key = _normalize_label_key(raw_label_stripped)
                    if norm_key in label_mapping_norm:
                        mapped_label = label_mapping_norm.get(norm_key)

                if mapped_label is None and strict:
                    missing_labels.add(raw_label_stripped)

        if missing_labels:
            sample = ", ".join(list(missing_labels)[:10])
            raise ValidationError(f"Missing label mappings for {len(missing_labels)} labels: {sample}")

        if dry_run:
            return {"files_scanned": 0, "updated_files": 0, "updated_labels": 0, "discarded": 0}

        stats = {"files_scanned": 0, "updated_files": 0, "updated_labels": 0, "discarded": 0}

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            shapes = None
            is_list = False
            if isinstance(data, dict) and isinstance(data.get("shapes"), list):
                shapes = data.get("shapes")
            elif isinstance(data, list):
                shapes = data
                is_list = True
            if shapes is None:
                continue

            stats["files_scanned"] += 1
            new_shapes = []
            changed = False

            for shape in shapes:
                if not isinstance(shape, dict) or "label" not in shape:
                    new_shapes.append(shape)
                    continue

                raw_label = str(shape.get("label", ""))
                raw_label_stripped = raw_label.strip()
                if not raw_label_stripped:
                    new_shapes.append(shape)
                    continue

                mapped_label = None
                if raw_label_stripped in label_mapping:
                    mapped_label = label_mapping.get(raw_label_stripped)
                else:
                    norm_key = _normalize_label_key(raw_label_stripped)
                    if norm_key in label_mapping_norm:
                        mapped_label = label_mapping_norm.get(norm_key)

                if mapped_label == "__DISCARD__" or mapped_label == "":
                    stats["discarded"] += 1
                    changed = True
                    continue

                if mapped_label is not None:
                    new_label = str(mapped_label).strip()
                    if new_label != raw_label:
                        shape["label"] = new_label
                        stats["updated_labels"] += 1
                        changed = True
                new_shapes.append(shape)

            if changed:
                if is_list:
                    data = new_shapes
                else:
                    data["shapes"] = new_shapes
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                stats["updated_files"] += 1

        return stats

    def _collect_pairs(self, root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
        """Collect image/json pairs by relative path (without extension) to avoid
        cross-directory stem collisions.  E.g. sub1/img.tif and sub2/img.tif
        are treated as separate pairs."""
        image_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        image_by_relkey: Dict[str, Path] = {}
        json_by_relkey: Dict[str, Path] = {}

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
                # Use relative path without extension as unique key
                rel_key = str(p.relative_to(root).with_suffix(""))
                if ext in image_exts:
                    prev = image_by_relkey.get(rel_key)
                    if prev is None or image_exts.index(ext) < image_exts.index(prev.suffix.lower()):
                        image_by_relkey[rel_key] = p
                elif ext == ".json":
                    if rel_key not in json_by_relkey:
                        json_by_relkey[rel_key] = p

        common = sorted(set(image_by_relkey.keys()) & set(json_by_relkey.keys()))
        pairs = [(image_by_relkey[s], json_by_relkey[s]) for s in common]

        warnings: list[str] = []
        extra_imgs = sorted(set(image_by_relkey.keys()) - set(json_by_relkey.keys()))
        extra_json = sorted(set(json_by_relkey.keys()) - set(image_by_relkey.keys()))
        if extra_imgs:
            warnings.append(f"Unmatched images: {len(extra_imgs)}")
        if extra_json:
            warnings.append(f"Unmatched json: {len(extra_json)}")

        return pairs, warnings

    def _run_single(
        self,
        cfg: dict,
        *,
        progress_cb: ProgressCallback = None,
        progress_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, Dict[str, int], dict]:
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

        base = dict(progress_context or {})
        slice_total = int(len(slices))
        _emit_progress(
            progress_cb,
            **base,
            phase="scanning",
            current_slice_processed=0,
            current_slice_total=slice_total,
            message="slice plan ready",
        )

        slices = assign_labels(
            slices,
            bboxes,
            min_area_ratio=cfg["min_area_ratio"],
            min_visibility=cfg["min_visibility"],
            min_pixel_size=cfg["min_pixel_size"],
            progress_cb=progress_cb,
            progress_context=base,
        )

        slices = post_filter_slices(
            slices,
            action=cfg["empty_positive_action"],
        )

        save_stats = save_slices(
            cfg,
            slices,
            progress_cb=progress_cb,
            progress_context=base,
        )

        _emit_progress(
            progress_cb,
            **base,
            phase="finalizing",
            current_slice_processed=slice_total,
            current_slice_total=slice_total,
            message="image finalized",
        )

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
