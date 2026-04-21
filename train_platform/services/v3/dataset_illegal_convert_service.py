from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from osgeo import gdal

    gdal.UseExceptions()
except Exception:  # pragma: no cover - optional dependency fallback
    gdal = None

from train_platform.services.v3.file_service import FileService
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
        total_cells = grid_rows * grid_cols
        inactive_count = max(0, total_cells - len(active_cells))
        n_neg = min(max(1, int(n_positive * negative_ratio)), inactive_count)
        if n_neg > 0:
            rng = np.random.default_rng()
            inactive_cells: List[Tuple[int, int]] = []

            if total_cells <= 50000:
                inactive_cells = [
                    (r, c)
                    for r in range(grid_rows)
                    for c in range(grid_cols)
                    if (r, c) not in active_cells
                ]
            else:
                sampled_cells = set()
                max_attempts = max(2048, n_neg * 32)
                attempts = 0
                while len(sampled_cells) < n_neg and attempts < max_attempts:
                    attempts += 1
                    cell = (int(rng.integers(grid_rows)), int(rng.integers(grid_cols)))
                    if cell in active_cells or cell in sampled_cells:
                        continue
                    sampled_cells.add(cell)

                if len(sampled_cells) < n_neg:
                    for r in range(grid_rows):
                        for c in range(grid_cols):
                            cell = (r, c)
                            if cell in active_cells or cell in sampled_cells:
                                continue
                            sampled_cells.add(cell)
                            if len(sampled_cells) >= n_neg:
                                break
                        if len(sampled_cells) >= n_neg:
                            break

                inactive_cells = list(sampled_cells)

            if inactive_cells:
                if len(inactive_cells) > n_neg:
                    chosen = rng.choice(len(inactive_cells), size=n_neg, replace=False)
                    selected_cells = [inactive_cells[int(ci)] for ci in chosen]
                else:
                    selected_cells = inactive_cells
                for r, c in selected_cells:
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
    bbox_buckets: Dict[Tuple[int, int], List[int]] = {}
    bbox_x_min = np.asarray([bbox.x_min for bbox in bboxes], dtype=np.float32)
    bbox_y_min = np.asarray([bbox.y_min for bbox in bboxes], dtype=np.float32)
    bbox_x_max = np.asarray([bbox.x_max for bbox in bboxes], dtype=np.float32)
    bbox_y_max = np.asarray([bbox.y_max for bbox in bboxes], dtype=np.float32)
    bbox_width = bbox_x_max - bbox_x_min
    bbox_height = bbox_y_max - bbox_y_min
    bbox_area = np.maximum(0.0, bbox_width) * np.maximum(0.0, bbox_height)
    bbox_labels = [bbox.label for bbox in bboxes]
    bbox_class_ids = np.asarray([bbox.class_id for bbox in bboxes], dtype=np.int32)

    for bbox_idx, bbox in enumerate(bboxes):
        r0, r1 = int(bbox.y_min) // BUCKET, int(bbox.y_max) // BUCKET
        c0, c1 = int(bbox.x_min) // BUCKET, int(bbox.x_max) // BUCKET
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bbox_buckets.setdefault((r, c), []).append(bbox_idx)

    bbox_bucket_arrays = {
        key: np.asarray(indices, dtype=np.int32)
        for key, indices in bbox_buckets.items()
    }

    total_slices = len(slices)
    base = dict(progress_context or {})
    for idx, sl in enumerate(slices, start=1):
        sx0, sy0 = sl.x, sl.y
        sx1, sy1 = sl.x + sl.w, sl.y + sl.h

        r0, r1 = sy0 // BUCKET, sy1 // BUCKET
        c0, c1 = sx0 // BUCKET, sx1 // BUCKET

        candidate_arrays: List[np.ndarray] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bucket = bbox_bucket_arrays.get((r, c))
                if bucket is not None and bucket.size > 0:
                    candidate_arrays.append(bucket)

        if candidate_arrays:
            if len(candidate_arrays) == 1:
                candidate_ids = candidate_arrays[0]
            else:
                candidate_ids = np.unique(np.concatenate(candidate_arrays))

            ix0 = np.maximum(float(sx0), bbox_x_min[candidate_ids])
            iy0 = np.maximum(float(sy0), bbox_y_min[candidate_ids])
            ix1 = np.minimum(float(sx1), bbox_x_max[candidate_ids])
            iy1 = np.minimum(float(sy1), bbox_y_max[candidate_ids])

            inter_w = ix1 - ix0
            inter_h = iy1 - iy0
            valid = (inter_w > 0.0) & (inter_h > 0.0)

            if valid.any():
                inter_area = inter_w * inter_h
                valid &= bbox_area[candidate_ids] > 0.0
                valid &= (inter_area / np.maximum(bbox_area[candidate_ids], 1e-6)) >= float(min_area_ratio)
                valid &= (inter_w / np.maximum(bbox_width[candidate_ids], 1e-6)) >= float(min_visibility)
                valid &= (inter_h / np.maximum(bbox_height[candidate_ids], 1e-6)) >= float(min_visibility)
                valid &= inter_w >= float(min_pixel_size)
                valid &= inter_h >= float(min_pixel_size)

                if valid.any():
                    kept_ids = candidate_ids[valid]
                    kept_ix0 = ix0[valid] - float(sx0)
                    kept_iy0 = iy0[valid] - float(sy0)
                    kept_ix1 = ix1[valid] - float(sx0)
                    kept_iy1 = iy1[valid] - float(sy0)

                    sl.bboxes.extend(
                        BBox(
                            x_min=float(kept_ix0[pos]),
                            y_min=float(kept_iy0[pos]),
                            x_max=float(kept_ix1[pos]),
                            y_max=float(kept_iy1[pos]),
                            label=bbox_labels[int(bbox_idx)],
                            class_id=int(bbox_class_ids[int(bbox_idx)]),
                        )
                        for pos, bbox_idx in enumerate(kept_ids)
                    )

        if _should_emit_progress(idx, total_slices):
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

_WINDOWED_RASTER_EXTS = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".j2k"}
_NUMPY_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_MAX_NUMPY_IMAGE_PIXELS = 100_000_000
_PERCENTILE_SAMPLE_GRID = 4
_PERCENTILE_SAMPLE_SIZE = 256
_PROGRESS_TARGET_UPDATES = 96


def _should_emit_progress(current: int, total: int, *, target_updates: int = _PROGRESS_TARGET_UPDATES) -> bool:
    if total <= 0:
        return False
    step = max(1, total // max(1, target_updates))
    return current == 1 or current == total or (current % step) == 0


def _write_json_file(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _ensure_rgb_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValidationError(f"Unsupported raster array shape: {arr.shape}")
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    return arr[:, :, :3]


def _stretch_to_uint8(data: np.ndarray, p2: float, p98: float) -> np.ndarray:
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)
    d = data.astype(np.float32, copy=False)
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        d = np.full(d.shape, 127.0, dtype=np.float32)
    else:
        d = (d - float(p2)) * (255.0 / float(p98 - p2))
    return np.clip(d, 0, 255).astype(np.uint8)


def _sampled_percentile_range(reader: "BaseImageReader", *, sample_grid: int = _PERCENTILE_SAMPLE_GRID) -> Tuple[float, float]:
    cache_key = getattr(reader, "image_path", "")
    if cache_key in _global_stretch_cache:
        return _global_stretch_cache[cache_key]

    sample_w = max(1, min(_PERCENTILE_SAMPLE_SIZE, int(reader.width)))
    sample_h = max(1, min(_PERCENTILE_SAMPLE_SIZE, int(reader.height)))
    grid_x = max(1, min(sample_grid, int(reader.width)))
    grid_y = max(1, min(sample_grid, int(reader.height)))
    xs = np.unique(
        np.linspace(0, max(0, int(reader.width) - sample_w), num=grid_x, dtype=np.int64)
    )
    ys = np.unique(
        np.linspace(0, max(0, int(reader.height) - sample_h), num=grid_y, dtype=np.int64)
    )

    samples: List[np.ndarray] = []
    max_values_per_patch = 65536
    for y in ys:
        for x in xs:
            patch = _ensure_rgb_array(
                reader.read_window_raw(int(x), int(y), int(sample_w), int(sample_h))
            )
            flat = np.asarray(patch).reshape(-1)
            if flat.size == 0:
                continue
            if np.issubdtype(flat.dtype, np.floating):
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    continue
            if flat.size > max_values_per_patch:
                stride = max(1, flat.size // max_values_per_patch)
                flat = flat[::stride]
            samples.append(flat.astype(np.float32, copy=False))

    if not samples:
        result = (0.0, 255.0)
    else:
        merged = np.concatenate(samples, axis=0)
        if merged.size == 0:
            result = (0.0, 255.0)
        else:
            result = (float(np.percentile(merged, 2)), float(np.percentile(merged, 98)))

    if cache_key:
        _global_stretch_cache[cache_key] = result
    return result


class BaseImageReader:
    image_path: str
    width: int
    height: int
    band_count: int
    dtype: str

    def read_window_raw(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        raise NotImplementedError

    def read_window_rgb(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return

    def __enter__(self) -> "BaseImageReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class WindowedRasterReader(BaseImageReader):
    def __init__(self, image_path: str, *, slice_size: int, is_uint8: bool) -> None:
        self.image_path = os.path.abspath(image_path)
        span = max(512, int(slice_size or 0))
        self._prefetch_w = max(span, span * 2)
        self._prefetch_h = max(span, span * 2)
        self._cache_bounds: Optional[Tuple[int, int, int, int]] = None
        self._cache_rgb: Optional[np.ndarray] = None
        self._stretch_stats: Optional[Tuple[float, float]] = None if is_uint8 else _sampled_percentile_range(self)

    def _load_cache(self, x: int, y: int, w: int, h: int) -> None:
        cache_w = min(int(self.width), max(int(w), int(self._prefetch_w)))
        cache_h = min(int(self.height), max(int(h), int(self._prefetch_h)))
        x0 = min(max(0, int(x)), max(0, int(self.width) - cache_w))
        y0 = min(max(0, int(y)), max(0, int(self.height) - cache_h))
        cache_w = min(cache_w, int(self.width) - x0)
        cache_h = min(cache_h, int(self.height) - y0)

        rgb = _ensure_rgb_array(self.read_window_raw(x0, y0, cache_w, cache_h))
        if self._stretch_stats is not None:
            rgb = _stretch_to_uint8(rgb, *self._stretch_stats)
        elif rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        self._cache_bounds = (x0, y0, cache_w, cache_h)
        self._cache_rgb = np.ascontiguousarray(rgb)

    def read_window_rgb(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        if (
            self._cache_bounds is None
            or self._cache_rgb is None
            or x < self._cache_bounds[0]
            or y < self._cache_bounds[1]
            or (x + w) > (self._cache_bounds[0] + self._cache_bounds[2])
            or (y + h) > (self._cache_bounds[1] + self._cache_bounds[3])
        ):
            self._load_cache(x, y, w, h)

        assert self._cache_bounds is not None and self._cache_rgb is not None
        x0, y0, _, _ = self._cache_bounds
        rel_x = int(x) - x0
        rel_y = int(y) - y0
        return np.ascontiguousarray(self._cache_rgb[rel_y:rel_y + int(h), rel_x:rel_x + int(w)])

    def close(self) -> None:
        self._cache_bounds = None
        self._cache_rgb = None


class GDALRasterReader(WindowedRasterReader):
    def __init__(self, image_path: str, *, slice_size: int) -> None:
        if gdal is None:
            raise ValidationError("GDAL is not available")
        self._dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        if self._dataset is None:
            raise ValidationError(f"Failed to open raster image with GDAL: {image_path}")

        self.width = int(self._dataset.RasterXSize)
        self.height = int(self._dataset.RasterYSize)
        self.band_count = max(1, int(self._dataset.RasterCount))
        band = self._dataset.GetRasterBand(1)
        self.dtype = (
            gdal.GetDataTypeName(band.DataType)
            if band is not None
            else "unknown"
        )
        self._band_list = [1, 2, 3] if self.band_count >= 3 else [1]
        super().__init__(
            image_path=image_path,
            slice_size=slice_size,
            is_uint8=bool(band is not None and band.DataType == gdal.GDT_Byte),
        )

    def read_window_raw(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        data = self._dataset.ReadAsArray(
            xoff=int(x),
            yoff=int(y),
            xsize=int(w),
            ysize=int(h),
            interleave="pixel",
            band_list=self._band_list,
        )
        if data is None:
            raise ValidationError(
                f"Failed to read raster window from {self.image_path}: x={x}, y={y}, w={w}, h={h}"
            )
        return np.asarray(data)

    def close(self) -> None:
        self._dataset = None
        super().close()


class RasterioRasterReader(WindowedRasterReader):
    def __init__(self, image_path: str, *, slice_size: int) -> None:
        self._dataset = rasterio.open(image_path)
        self.width = int(self._dataset.width)
        self.height = int(self._dataset.height)
        self.band_count = max(1, int(self._dataset.count))
        self.dtype = str(self._dataset.dtypes[0]) if self._dataset.dtypes else "unknown"
        self._band_list = [1, 2, 3] if self.band_count >= 3 else [1]
        super().__init__(
            image_path=image_path,
            slice_size=slice_size,
            is_uint8=self.dtype == "uint8",
        )

    def read_window_raw(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        window = Window(col_off=int(x), row_off=int(y), width=int(w), height=int(h))
        if self.band_count >= 3:
            data = self._dataset.read(self._band_list, window=window)
        else:
            data = self._dataset.read(1, window=window)
        return np.asarray(data)

    def close(self) -> None:
        try:
            self._dataset.close()
        finally:
            super().close()


class NumpyImageReader(BaseImageReader):
    def __init__(self, image_path: str) -> None:
        self.image_path = os.path.abspath(image_path)
        with Image.open(image_path) as img:
            self.width, self.height = map(int, img.size)
            self.band_count = len(img.getbands())
            rgb_img = img if img.mode == "RGB" else img.convert("RGB")
            self._data = np.ascontiguousarray(np.asarray(rgb_img))
        self.dtype = str(self._data.dtype)

    def read_window_raw(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return np.ascontiguousarray(self._data[int(y):int(y) + int(h), int(x):int(x) + int(w)])

    def read_window_rgb(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return self.read_window_raw(x, y, w, h)

    def close(self) -> None:
        self._data = np.empty((0, 0, 3), dtype=np.uint8)


def read_image_metadata(image_path: str) -> Tuple[int, int]:
    ext = Path(image_path).suffix.lower()
    if ext in _NUMPY_IMAGE_EXTS:
        with Image.open(image_path) as img:
            width, height = img.size
        return int(width), int(height)

    if gdal is not None:
        ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        if ds is not None:
            return int(ds.RasterXSize), int(ds.RasterYSize)

    with rasterio.open(image_path) as ds:
        return int(ds.width), int(ds.height)


def _should_use_numpy_reader(image_path: str) -> bool:
    ext = Path(image_path).suffix.lower()
    if ext not in _NUMPY_IMAGE_EXTS:
        return False
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return (int(width) * int(height)) <= _MAX_NUMPY_IMAGE_PIXELS
    except Exception:
        return False


def open_image_reader(image_path: str, *, slice_size: int) -> BaseImageReader:
    ext = Path(image_path).suffix.lower()
    if _should_use_numpy_reader(image_path):
        return NumpyImageReader(image_path)

    if gdal is not None and ext in _WINDOWED_RASTER_EXTS:
        try:
            return GDALRasterReader(image_path, slice_size=slice_size)
        except Exception:
            pass

    if gdal is not None:
        try:
            return GDALRasterReader(image_path, slice_size=slice_size)
        except Exception:
            pass

    return RasterioRasterReader(image_path, slice_size=slice_size)


def save_slices(
    cfg: dict,
    slices: List[SliceInfo],
    reader: BaseImageReader,
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
    png_compress_level = int(cfg.get("png_compress_level", 1))
    save_negative = cfg["negative_ratio"] > 0

    stats = {"total": 0, "with_labels": 0, "empty": 0, "total_labels": 0}

    slices_to_save = [
        sl for sl in slices
        if len(sl.bboxes) > 0 or save_negative
    ]
    if not slices_to_save:
        return stats

    ordered_slices = sorted(slices_to_save, key=lambda sl: (sl.y, sl.x, sl.idx))
    total_slices = len(ordered_slices)
    base = dict(progress_context or {})

    for idx, sl in enumerate(ordered_slices, start=1):
        has_labels = len(sl.bboxes) > 0
        rgb = reader.read_window_rgb(sl.x, sl.y, sl.w, sl.h)

        name = f"{prefix}_{sl.idx:06d}"
        img_path = os.path.join(img_dir, f"{name}.{ext}")
        pil_img = Image.fromarray(rgb)
        if ext in ("jpg", "jpeg"):
            pil_img.save(img_path, quality=quality, optimize=False)
        elif ext == "png":
            pil_img.save(img_path, compress_level=max(0, min(9, png_compress_level)))
        else:
            pil_img.save(img_path)

        lbl_path = os.path.join(lbl_dir, f"{name}.txt")
        lines = [bbox_to_yolo(bbox, sl.w, sl.h) for bbox in sl.bboxes]
        with open(lbl_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))
                f.write("\n")

        stats["total"] += 1
        if has_labels:
            stats["with_labels"] += 1
            stats["total_labels"] += len(sl.bboxes)
        else:
            stats["empty"] += 1

        if _should_emit_progress(idx, total_slices):
            _emit_progress(
                progress_cb,
                **base,
                phase="save_slices",
                current_slice_processed=int(idx),
                current_slice_total=int(total_slices),
            )

    return stats


def _convert_pair_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(job["cfg"])
    image_path = str(cfg["image_path"])
    json_path = str(cfg["annotation_path"])
    info_path = str(job["info_path"])
    try:
        stats, _label_map, slicing_meta = DatasetIllegalConvertService()._run_single(cfg)
        _write_json_file(info_path, slicing_meta)
        return {
            "ok": True,
            "stats": stats,
            "image_path": image_path,
            "annotation_path": json_path,
            "info_path": info_path,
        }
    except ValidationError as exc:
        return {
            "ok": False,
            "error_kind": "validation",
            "error": str(exc),
            "image_path": image_path,
            "annotation_path": json_path,
            "info_path": info_path,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_kind": "fatal",
            "error": f"{type(exc).__name__}: {exc}",
            "image_path": image_path,
            "annotation_path": json_path,
            "info_path": info_path,
        }


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
        "output_format": "jpg",
        "jpg_quality": 95,
        "png_compress_level": 1,
        "parallel_workers": None,
        "parallel_min_pairs": 2,
        "label_map": None,
    }

    _skip_dirs = {"labels", ".versions", ".thumbnails"}

    def _build_pair_cfg(
        self,
        *,
        root: Path,
        image_path: Path,
        json_path: Path,
        label_strategy: str,
        label_level: Optional[int],
        label_separator: str,
        label_mapping: Optional[dict],
        slice_config: Optional[dict],
        label_map: Optional[Dict[str, int]] = None,
    ) -> dict:
        cfg = dict(self.DEFAULT_CONFIG)
        if isinstance(slice_config, dict):
            for key in (
                "slice_size",
                "overlap",
                "padding",
                "min_area_ratio",
                "min_visibility",
                "min_pixel_size",
                "negative_ratio",
                "empty_positive_action",
                "parallel_workers",
                "parallel_min_pairs",
            ):
                if key in slice_config and slice_config[key] is not None:
                    cfg[key] = slice_config[key]

        cfg["image_path"] = str(image_path)
        cfg["annotation_path"] = str(json_path)
        cfg["output_dir"] = str(root)
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
        cfg["label_map"] = dict(label_map or {})
        cfg["label_mapping"] = label_mapping
        cfg["relative_stem"] = rel_stem
        return cfg

    def _make_progress_context(
        self,
        *,
        total_pairs: int,
        processed: int,
        pair_index: int,
        image_path: Path,
    ) -> Dict[str, Any]:
        return {
            "overall_total_images": int(total_pairs),
            "overall_processed_images": int(processed),
            "current_image_index": int(pair_index),
            "current_image_name": str(image_path.name),
        }

    def _make_slicing_info_path(self, root: Path, cfg: dict) -> Path:
        rel_stem = str(cfg.get("relative_stem") or Path(cfg["image_path"]).stem)
        safe_name = rel_stem.replace(os.sep, "_").replace("/", "_")
        return root / f"slicing_info_{safe_name}.json"

    def _preflight_pair(self, cfg: dict) -> Tuple[Dict[str, int], List[str]]:
        img_w, img_h = read_image_metadata(str(cfg["image_path"]))

        bboxes, _label_map = parse_annotations(cfg)
        if not bboxes:
            raise ValidationError(f"No valid annotations in {cfg['annotation_path']}")

        for bbox in bboxes:
            bbox.x_min = max(0.0, bbox.x_min)
            bbox.y_min = max(0.0, bbox.y_min)
            bbox.x_max = min(float(img_w), bbox.x_max)
            bbox.y_max = min(float(img_h), bbox.y_max)
        bboxes = [bbox for bbox in bboxes if bbox.width > 0 and bbox.height > 0]
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

        next_label_map = dict(cfg.get("label_map") or {})
        image_labels: List[str] = []
        seen_labels: set[str] = set()
        for bbox in bboxes:
            if bbox.label not in next_label_map:
                next_label_map[bbox.label] = len(next_label_map)
            if bbox.label not in seen_labels:
                seen_labels.add(bbox.label)
                image_labels.append(bbox.label)

        return next_label_map, image_labels

    def _resolve_parallel_workers(self, total_jobs: int, slice_config: Optional[dict]) -> int:
        min_pairs = 2
        requested_workers = None
        if isinstance(slice_config, dict):
            if slice_config.get("parallel_min_pairs") is not None:
                try:
                    min_pairs = max(1, int(slice_config["parallel_min_pairs"]))
                except Exception:
                    min_pairs = 2
            if slice_config.get("parallel_workers") is not None:
                try:
                    requested_workers = int(slice_config["parallel_workers"])
                except Exception:
                    requested_workers = None

        if total_jobs < min_pairs:
            return 1

        if requested_workers is not None:
            return max(1, min(total_jobs, requested_workers))

        cpu_total = max(1, os.cpu_count() or 1)
        auto_workers = cpu_total if cpu_total <= 2 else cpu_total - 1
        auto_workers = min(max(1, auto_workers), 8)
        return max(1, min(total_jobs, auto_workers))

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
        skipped_files: List[str] = []
        processed_pairs: List[Tuple[Path, Path]] = []
        successful_labels: set[str] = set()
        total_pairs = len(pairs)
        jobs: List[Dict[str, Any]] = []

        for pair_index, (image_path, json_path) in enumerate(pairs, start=1):
            progress_context = self._make_progress_context(
                total_pairs=total_pairs,
                processed=processed,
                pair_index=pair_index,
                image_path=image_path,
            )
            try:
                cfg = self._build_pair_cfg(
                    root=root,
                    image_path=image_path,
                    json_path=json_path,
                    label_strategy=label_strategy,
                    label_level=label_level,
                    label_separator=label_separator,
                    label_mapping=label_mapping,
                    slice_config=slice_config,
                    label_map=global_label_map,
                )
                next_label_map, image_labels = self._preflight_pair(cfg)
                global_label_map = next_label_map
                cfg["label_map"] = dict(next_label_map)
                jobs.append(
                    {
                        "pair_index": int(pair_index),
                        "image_path": image_path,
                        "json_path": json_path,
                        "cfg": cfg,
                        "info_path": str(self._make_slicing_info_path(root, cfg)),
                        "image_labels": image_labels,
                        "progress_context": progress_context,
                    }
                )
            except ValidationError as e:
                skipped_files.append(f"{json_path.name}: {e}")
                warnings.append(f"Skipped {json_path.name}: {e}")
                _emit_progress(
                    progress_cb,
                    **progress_context,
                    phase="skip_image",
                    message=f"Skipped {json_path.name}: {e}",
                    current_slice_processed=0,
                    current_slice_total=0,
                )
                continue

        worker_count = self._resolve_parallel_workers(len(jobs), slice_config)

        if worker_count <= 1 or len(jobs) <= 1:
            for job in jobs:
                progress_context = dict(job["progress_context"])
                _emit_progress(
                    progress_cb,
                    **progress_context,
                    phase="prepare_image",
                    current_slice_processed=0,
                    current_slice_total=0,
                )
                try:
                    stats, _label_map, slicing_meta = self._run_single(
                        job["cfg"],
                        progress_cb=progress_cb,
                        progress_context=progress_context,
                    )
                    _write_json_file(job["info_path"], slicing_meta)
                except ValidationError as e:
                    skipped_files.append(f"{job['json_path'].name}: {e}")
                    warnings.append(f"Skipped {job['json_path'].name}: {e}")
                    _emit_progress(
                        progress_cb,
                        **progress_context,
                        phase="skip_image",
                        message=f"Skipped {job['json_path'].name}: {e}",
                        current_slice_processed=0,
                        current_slice_total=0,
                    )
                    continue

                processed += 1
                processed_pairs.append((job["image_path"], job["json_path"]))
                successful_labels.update(job.get("image_labels") or [])
                _emit_progress(
                    progress_cb,
                    overall_total_images=int(total_pairs),
                    overall_processed_images=int(processed),
                    current_image_index=int(job["pair_index"]),
                    current_image_name=str(job["image_path"].name),
                    phase="image_completed",
                    current_slice_processed=int(stats.get("total", 0)),
                    current_slice_total=int(stats.get("total", 0)),
                )
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_job = {}
                for job in jobs:
                    _emit_progress(
                        progress_cb,
                        **job["progress_context"],
                        phase="prepare_image",
                        current_slice_processed=0,
                        current_slice_total=0,
                    )
                    future = executor.submit(_convert_pair_worker, job)
                    future_to_job[future] = job

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "ok": False,
                            "error_kind": "fatal",
                            "error": f"{type(e).__name__}: {e}",
                            "image_path": str(job["image_path"]),
                            "annotation_path": str(job["json_path"]),
                        }

                    if result.get("ok"):
                        processed += 1
                        processed_pairs.append((job["image_path"], job["json_path"]))
                        successful_labels.update(job.get("image_labels") or [])
                        stats = result.get("stats") or {}
                        _emit_progress(
                            progress_cb,
                            overall_total_images=int(total_pairs),
                            overall_processed_images=int(processed),
                            current_image_index=int(job["pair_index"]),
                            current_image_name=str(job["image_path"].name),
                            phase="image_completed",
                            current_slice_processed=int(stats.get("total", 0)),
                            current_slice_total=int(stats.get("total", 0)),
                        )
                    else:
                        if result.get("error_kind") == "fatal":
                            raise RuntimeError(
                                f"Dataset conversion failed for {job['image_path'].name}: {result.get('error')}"
                            )
                        error = str(result.get("error") or "unknown error")
                        skipped_files.append(f"{job['json_path'].name}: {error}")
                        warnings.append(f"Skipped {job['json_path'].name}: {error}")
                        _emit_progress(
                            progress_cb,
                            overall_total_images=int(total_pairs),
                            overall_processed_images=int(processed),
                            current_image_index=int(job["pair_index"]),
                            current_image_name=str(job["image_path"].name),
                            phase="skip_image",
                            message=f"Skipped {job['json_path'].name}: {error}",
                            current_slice_processed=0,
                            current_slice_total=0,
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
            if name and name.strip() and name in successful_labels
        ]
        classes_path = root / "classes.txt"
        with open(classes_path, "w", encoding="utf-8") as f:
            for name in sorted_classes:
                f.write(name + "\n")

        FileService()._create_yolo_data_yaml(root, root / "data.yaml")

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
        with open_image_reader(cfg["image_path"], slice_size=int(cfg["slice_size"])) as reader:
            img_w, img_h = int(reader.width), int(reader.height)
            n_bands, dtype = int(reader.band_count), reader.dtype

            bboxes, label_map = parse_annotations(cfg)
            if not bboxes:
                raise ValidationError(f"No valid annotations in {cfg['annotation_path']}")

            for b in bboxes:
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
                progress_cb=progress_cb,
                progress_context=progress_context,
            )

            slices = post_filter_slices(
                slices,
                action=cfg["empty_positive_action"],
            )

            save_stats = save_slices(
                cfg,
                slices,
                reader=reader,
                progress_cb=progress_cb,
                progress_context=progress_context,
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
