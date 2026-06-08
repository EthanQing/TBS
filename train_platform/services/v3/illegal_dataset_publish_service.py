from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio.windows import Window

from train_platform.utils.exceptions import ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS

Image.MAX_IMAGE_PIXELS = None

try:
    from osgeo import gdal

    gdal.UseExceptions()
except Exception:  # pragma: no cover - optional dependency
    gdal = None


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


_WINDOWED_RASTER_EXTS = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".j2k"}
_NUMPY_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_MAX_NUMPY_IMAGE_PIXELS = 100_000_000
_PERCENTILE_SAMPLE_GRID = 4
_PERCENTILE_SAMPLE_SIZE = 256


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
        return separator.join(parts)
    if strategy == "leaf":
        return parts[-1]
    if strategy == "root":
        return parts[0]
    if isinstance(strategy, int):
        return separator.join(parts[:strategy])
    return raw_label


def _normalize_label_key(value: Any) -> str:
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

    if isinstance(data, dict) and "shapes" in data:
        shapes = data["shapes"]
    elif isinstance(data, list):
        shapes = data
    else:
        raise ValidationError(f"Unrecognized json structure: {json_path}")

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
        stype = str(shape.get("shape_type", "polygon")).lower()

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
        inactive_cells = [(r, c) for r in range(grid_rows) for c in range(grid_cols) if (r, c) not in active_cells]
        n_neg = min(max(1, int(n_positive * negative_ratio)), len(inactive_cells))
        if n_neg > 0 and inactive_cells:
            rng = np.random.default_rng()
            chosen = rng.choice(len(inactive_cells), size=n_neg, replace=False) if len(inactive_cells) > n_neg else range(len(inactive_cells))
            for ci in chosen:
                r, c = inactive_cells[int(ci)]
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
    bucket = 1024
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
        r0, r1 = int(bbox.y_min) // bucket, int(bbox.y_max) // bucket
        c0, c1 = int(bbox.x_min) // bucket, int(bbox.x_max) // bucket
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bbox_buckets.setdefault((r, c), []).append(bbox_idx)

    bbox_bucket_arrays = {key: np.asarray(indices, dtype=np.int32) for key, indices in bbox_buckets.items()}

    for sl in slices:
        sx0, sy0 = sl.x, sl.y
        sx1, sy1 = sl.x + sl.w, sl.y + sl.h
        r0, r1 = sy0 // bucket, sy1 // bucket
        c0, c1 = sx0 // bucket, sx1 // bucket

        candidate_arrays: List[np.ndarray] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                candidate = bbox_bucket_arrays.get((r, c))
                if candidate is not None and candidate.size > 0:
                    candidate_arrays.append(candidate)

        if not candidate_arrays:
            continue

        candidate_ids = candidate_arrays[0] if len(candidate_arrays) == 1 else np.unique(np.concatenate(candidate_arrays))

        ix0 = np.maximum(float(sx0), bbox_x_min[candidate_ids])
        iy0 = np.maximum(float(sy0), bbox_y_min[candidate_ids])
        ix1 = np.minimum(float(sx1), bbox_x_max[candidate_ids])
        iy1 = np.minimum(float(sy1), bbox_y_max[candidate_ids])

        inter_w = ix1 - ix0
        inter_h = iy1 - iy0
        valid = (inter_w > 0.0) & (inter_h > 0.0)
        if not valid.any():
            continue

        inter_area = inter_w * inter_h
        valid &= bbox_area[candidate_ids] > 0.0
        valid &= (inter_area / np.maximum(bbox_area[candidate_ids], 1e-6)) >= float(min_area_ratio)
        valid &= (inter_w / np.maximum(bbox_width[candidate_ids], 1e-6)) >= float(min_visibility)
        valid &= (inter_h / np.maximum(bbox_height[candidate_ids], 1e-6)) >= float(min_visibility)
        valid &= inter_w >= float(min_pixel_size)
        valid &= inter_h >= float(min_pixel_size)
        if not valid.any():
            continue

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

    for idx, sl in enumerate(kept):
        sl.idx = idx
    return kept


def bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> str:
    cx = np.clip((bbox.x_min + bbox.x_max) / 2.0 / img_w, 0, 1)
    cy = np.clip((bbox.y_min + bbox.y_max) / 2.0 / img_h, 0, 1)
    w = np.clip(bbox.width / img_w, 0, 1)
    h = np.clip(bbox.height / img_h, 0, 1)
    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


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


_global_stretch_cache: Dict[str, Tuple[float, float]] = {}


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
    xs = np.unique(np.linspace(0, max(0, int(reader.width) - sample_w), num=grid_x, dtype=np.int64))
    ys = np.unique(np.linspace(0, max(0, int(reader.height) - sample_h), num=grid_y, dtype=np.int64))

    samples: List[np.ndarray] = []
    max_values_per_patch = 65536
    for y in ys:
        for x in xs:
            patch = _ensure_rgb_array(reader.read_window_raw(int(x), int(y), int(sample_w), int(sample_h)))
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
        result = (float(np.percentile(merged, 2)), float(np.percentile(merged, 98))) if merged.size else (0.0, 255.0)

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
        self.dtype = gdal.GetDataTypeName(band.DataType) if band is not None else "unknown"
        self._band_list = [1, 2, 3] if self.band_count >= 3 else [1]
        super().__init__(image_path=image_path, slice_size=slice_size, is_uint8=bool(band is not None and band.DataType == gdal.GDT_Byte))

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
        super().__init__(image_path=image_path, slice_size=slice_size, is_uint8=self.dtype == "uint8")

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


def save_slices(cfg: dict, slices: List[SliceInfo], reader: BaseImageReader) -> Dict[str, int]:
    output_dir = Path(cfg["output_dir"])
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ext = str(cfg["output_format"]).lower().strip(".")
    prefix = str(cfg["prefix"])
    quality = int(cfg["jpg_quality"])
    png_compress_level = int(cfg.get("png_compress_level", 1))
    save_negative = float(cfg["negative_ratio"]) > 0

    stats = {"total": 0, "with_labels": 0, "empty": 0, "total_labels": 0}
    slices_to_save = [sl for sl in slices if len(sl.bboxes) > 0 or save_negative]
    if not slices_to_save:
        return stats

    for sl in sorted(slices_to_save, key=lambda item: (item.y, item.x, item.idx)):
        has_labels = len(sl.bboxes) > 0
        rgb = reader.read_window_rgb(sl.x, sl.y, sl.w, sl.h)

        name = f"{prefix}_{sl.idx:06d}"
        img_path = img_dir / f"{name}.{ext}"
        pil_img = Image.fromarray(rgb)
        if ext in ("jpg", "jpeg"):
            pil_img.save(img_path, quality=quality, optimize=False)
        elif ext == "png":
            pil_img.save(img_path, compress_level=max(0, min(9, png_compress_level)))
        else:
            pil_img.save(img_path)

        lbl_path = lbl_dir / f"{name}.txt"
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

    return stats


class IllegalDatasetPublishService:
    DEFAULT_CONFIG: dict = {
        "slice_enabled": True,
        "slice_size": 1280,
        "overlap": 0.2,
        "padding": 64,
        "min_area_ratio": 0.3,
        "min_visibility": 0.15,
        "min_pixel_size": 5,
        "min_probability": 0.0,
        "skip_hidden": True,
        "skip_outside": True,
        "label_strategy": "mapping",
        "label_separator": "%",
        "negative_ratio": 0.1,
        "empty_positive_action": "discard",
        "output_format": "jpg",
        "jpg_quality": 95,
        "png_compress_level": 1,
        "label_map": None,
    }

    _skip_dirs = {"labels", ".versions", ".thumbnails", "__macosx"}
    _skip_files = {".dataset_stats.json", ".dataset_view_index.json", ".mounted_manifest.json"}
    _pair_prefix_aliases = {"image", "images", "annotation", "annotations", "json", "labels"}

    def _pair_key(self, root: Path, path: Path) -> str:
        rel = Path(path).relative_to(root).with_suffix("")
        parts = list(rel.parts)
        if parts and parts[0].lower() in self._pair_prefix_aliases:
            parts = parts[1:]
        return "/".join(parts).lower()

    def extract_dataset_labels(self, root: Path) -> list[str]:
        labels: set[str] = set()
        for cur, dirnames, filenames in os.walk(root):
            cur_p = Path(cur)
            rel = cur_p.relative_to(root)
            if rel.parts and rel.parts[0].lower() in self._skip_dirs:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in self._skip_dirs]
            for fname in filenames:
                if fname.lower() in self._skip_files:
                    continue
                if not fname.lower().endswith(".json"):
                    continue
                json_path = cur_p / fname
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    try:
                        with open(json_path, "r", encoding="gbk", errors="ignore") as f:
                            data = json.load(f)
                    except Exception:
                        continue
                shapes = []
                if isinstance(data, dict) and isinstance(data.get("shapes"), list):
                    shapes = data["shapes"]
                elif isinstance(data, list):
                    shapes = data
                for shape in shapes:
                    label = str((shape or {}).get("label") or "").strip()
                    if label:
                        labels.add(label)
        return sorted(labels)

    def _collect_pairs(self, root: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
        image_by_relkey: Dict[str, Path] = {}
        json_by_relkey: Dict[str, Path] = {}
        image_exts = {ext.lower() for ext in IMAGE_EXTS}

        for cur, dirnames, filenames in os.walk(root):
            cur_p = Path(cur)
            rel = cur_p.relative_to(root)
            if rel.parts and rel.parts[0].lower() in self._skip_dirs:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in self._skip_dirs]

            for fname in filenames:
                if fname.lower() in self._skip_files:
                    continue
                p = cur_p / fname
                rel_key = self._pair_key(root, p)
                ext = p.suffix.lower()
                if ext in image_exts:
                    image_by_relkey.setdefault(rel_key, p)
                elif ext == ".json":
                    json_by_relkey.setdefault(rel_key, p)

        common = sorted(set(image_by_relkey.keys()) & set(json_by_relkey.keys()))
        pairs = [(image_by_relkey[key], json_by_relkey[key]) for key in common]

        warnings: list[str] = []
        unmatched_details: list[str] = []
        extra_imgs = sorted(set(image_by_relkey.keys()) - set(json_by_relkey.keys()))
        extra_json = sorted(set(json_by_relkey.keys()) - set(image_by_relkey.keys()))
        if extra_imgs:
            warnings.append(f"Unmatched images: {len(extra_imgs)}")
            unmatched_details.extend(f"{image_by_relkey[key].name}: missing json annotation" for key in extra_imgs[:50])
        if extra_json:
            warnings.append(f"Unmatched json: {len(extra_json)}")
            unmatched_details.extend(f"{json_by_relkey[key].name}: missing image file" for key in extra_json[:50])
        if unmatched_details:
            warnings.extend(f"Skipped {item}" for item in unmatched_details[:50])
        return pairs, warnings, unmatched_details

    def _normalize_slice_config(self, publish_config: Optional[dict]) -> dict:
        cfg = dict(self.DEFAULT_CONFIG)
        raw = publish_config if isinstance(publish_config, dict) else {}
        conversion = raw.get("conversion") if isinstance(raw.get("conversion"), dict) else {}
        slice_cfg = conversion.get("slice") if isinstance(conversion.get("slice"), dict) else {}
        flat_cfg = raw.get("slice") if isinstance(raw.get("slice"), dict) else {}
        merged = {**flat_cfg, **slice_cfg}

        if merged.get("enabled") is not None:
            cfg["slice_enabled"] = bool(merged.get("enabled"))
        for key in (
            "slice_size",
            "overlap",
            "padding",
            "min_area_ratio",
            "min_visibility",
            "min_pixel_size",
            "negative_ratio",
            "empty_positive_action",
            "output_format",
            "jpg_quality",
            "png_compress_level",
            "label_separator",
            "label_strategy",
        ):
            if merged.get(key) is not None:
                cfg[key] = merged.get(key)

        cfg["slice_size"] = max(64, int(cfg["slice_size"]))
        cfg["padding"] = max(0, int(cfg["padding"]))
        cfg["overlap"] = max(0.0, min(0.95, float(cfg["overlap"])))
        cfg["min_area_ratio"] = max(0.0, min(1.0, float(cfg["min_area_ratio"])))
        cfg["min_visibility"] = max(0.0, min(1.0, float(cfg["min_visibility"])))
        cfg["min_pixel_size"] = max(1, int(cfg["min_pixel_size"]))
        cfg["negative_ratio"] = max(0.0, min(1.0, float(cfg["negative_ratio"])))
        cfg["empty_positive_action"] = str(cfg["empty_positive_action"] or "discard")
        cfg["label_separator"] = str(cfg["label_separator"] or "%")
        cfg["label_strategy"] = str(cfg["label_strategy"] or "mapping")
        cfg["output_format"] = str(cfg["output_format"] or "jpg").lower().strip(".") or "jpg"
        if cfg["output_format"] not in {"jpg", "jpeg", "png", "bmp", "webp"}:
            cfg["output_format"] = "jpg"
        return cfg

    def _build_effective_mapping(
        self,
        label_mapping: Optional[dict[str, str]],
        label_filters: Optional[list[str]],
    ) -> dict[str, str]:
        mapping = {str(k): str(v) for k, v in (label_mapping or {}).items() if str(k).strip() and str(v).strip()}
        filters = {str(item).strip() for item in (label_filters or []) if str(item).strip()}
        if not filters:
            return mapping
        effective: dict[str, str] = {}
        for raw_label, mapped_label in mapping.items():
            effective[raw_label] = mapped_label if mapped_label in filters else "__DISCARD__"
        return effective

    def _build_pair_cfg(
        self,
        *,
        source_root: Path,
        output_root: Path,
        image_path: Path,
        json_path: Path,
        label_mapping: Optional[dict[str, str]],
        slice_config: dict,
        label_map: Optional[Dict[str, int]],
    ) -> dict:
        cfg = dict(slice_config)
        cfg["image_path"] = str(image_path)
        cfg["annotation_path"] = str(json_path)
        cfg["output_dir"] = str(output_root)
        rel_stem = str(image_path.relative_to(source_root).with_suffix("")).replace(os.sep, "_").replace("/", "_")
        cfg["prefix"] = f"{rel_stem}_slice"
        cfg["label_map"] = dict(label_map or {})
        cfg["label_mapping"] = label_mapping
        return cfg

    def _run_single(self, cfg: dict) -> tuple[dict[str, int], Dict[str, int]]:
        with open_image_reader(cfg["image_path"], slice_size=int(cfg["slice_size"])) as reader:
            img_w, img_h = int(reader.width), int(reader.height)
            bboxes, label_map = parse_annotations(cfg)
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

            if bool(cfg.get("slice_enabled", True)):
                slices = plan_slices(
                    img_w,
                    img_h,
                    bboxes,
                    slice_size=int(cfg["slice_size"]),
                    overlap=float(cfg["overlap"]),
                    padding=int(cfg["padding"]),
                    negative_ratio=float(cfg["negative_ratio"]),
                )
                if not slices:
                    raise ValidationError(f"No slices planned for {cfg['annotation_path']}")
                slices = assign_labels(
                    slices,
                    bboxes,
                    min_area_ratio=float(cfg["min_area_ratio"]),
                    min_visibility=float(cfg["min_visibility"]),
                    min_pixel_size=int(cfg["min_pixel_size"]),
                )
                slices = post_filter_slices(slices, action=str(cfg["empty_positive_action"] or "discard"))
            else:
                slices = [
                    SliceInfo(
                        idx=0,
                        x=0,
                        y=0,
                        w=img_w,
                        h=img_h,
                        is_negative=False,
                        bboxes=[
                            BBox(
                                x_min=float(bbox.x_min),
                                y_min=float(bbox.y_min),
                                x_max=float(bbox.x_max),
                                y_max=float(bbox.y_max),
                                label=str(bbox.label),
                                class_id=int(bbox.class_id),
                            )
                            for bbox in bboxes
                        ],
                    )
                ]

            stats = save_slices(cfg, slices, reader)
            return stats, label_map

    def _write_class_files(self, output_root: Path, label_map: Dict[str, int], successful_labels: set[str], split_summary: Optional[dict]) -> list[str]:
        class_names = [
            name
            for name, _cid in sorted(label_map.items(), key=lambda item: item[1])
            if name and name.strip() and name in successful_labels
        ]
        if not class_names:
            raise ValidationError("No valid labels remained after publish conversion")

        classes_path = output_root / "classes.txt"
        with open(classes_path, "w", encoding="utf-8") as f:
            for name in class_names:
                f.write(name + "\n")

        yaml_payload: dict[str, Any]
        if split_summary and split_summary.get("total_images", 0) > 0:
            train_dir = "images/train" if (output_root / "images" / "train").exists() else "images"
            val_dir = "images/val" if (output_root / "images" / "val").exists() else train_dir
            yaml_payload = {
                "train": train_dir,
                "val": val_dir,
                "nc": len(class_names),
                "names": class_names,
            }
            if (output_root / "images" / "test").exists():
                yaml_payload["test"] = "images/test"
        else:
            yaml_payload = {
                "train": "images",
                "val": "images",
                "nc": len(class_names),
                "names": class_names,
            }

        with open(output_root / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_payload, f, allow_unicode=True, sort_keys=False)
        return class_names

    def _normalize_split_config(self, split_config: Optional[dict]) -> dict[str, Any]:
        raw = split_config if isinstance(split_config, dict) else {}
        train = float(raw.get("train") or 0)
        val = float(raw.get("val") or 0)
        test = float(raw.get("test") or 0)
        if min(train, val, test) < 0 or max(train, val, test) > 1:
            raise ValidationError("train / val / test must be between 0 and 1")
        total = train + val + test
        if total <= 0:
            return {"enabled": False, "train": 1.0, "val": 0.0, "test": 0.0, "shuffle": False, "seed": None}
        if abs(total - 1.0) > 0.001:
            raise ValidationError("train + val + test must equal 1")
        if train <= 0:
            raise ValidationError("train split must be greater than 0")
        return {
            "enabled": True,
            "train": train,
            "val": val,
            "test": test,
            "shuffle": bool(raw.get("shuffle", True)),
            "seed": int(raw["seed"]) if raw.get("seed") is not None and str(raw.get("seed")).strip() != "" else None,
        }

    def _iter_generated_pairs(self, output_root: Path) -> list[tuple[Path, Path]]:
        images_dir = output_root / "images"
        labels_dir = output_root / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            return []
        pairs: list[tuple[Path, Path]] = []
        for image_path in sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
        return pairs

    def apply_split(self, output_root: Path, *, split_config: Optional[dict]) -> Optional[dict[str, Any]]:
        cfg = self._normalize_split_config(split_config)
        if not cfg.get("enabled"):
            return None

        pairs = self._iter_generated_pairs(output_root)
        total = len(pairs)
        if total <= 0:
            raise ValidationError("No converted images available for split")

        items = list(pairs)
        if cfg["shuffle"]:
            rng = random.Random(cfg["seed"]) if cfg["seed"] is not None else random.Random()
            rng.shuffle(items)

        desired = {
            "train": total * float(cfg["train"]),
            "val": total * float(cfg["val"]),
            "test": total * float(cfg["test"]),
        }
        counts = {name: int(math.floor(value)) for name, value in desired.items()}
        remaining = total - sum(counts.values())
        priority = {"train": 0, "val": 1, "test": 2}
        if remaining > 0:
            remainders = sorted(
                ((desired[name] - counts[name], priority[name], name) for name in ("train", "val", "test")),
                key=lambda item: (-item[0], item[1]),
            )
            idx = 0
            while remaining > 0 and remainders:
                _, _, name = remainders[idx % len(remainders)]
                counts[name] += 1
                remaining -= 1
                idx += 1

        if counts["train"] <= 0:
            donor = next((name for name in ("val", "test") if counts[name] > 1), None)
            if donor is None:
                donor = next((name for name in ("val", "test") if counts[name] > 0), None)
            if donor is not None:
                counts[donor] -= 1
            counts["train"] += 1

        train_count = counts["train"]
        val_count = counts["val"]
        test_count = counts["test"]

        assignments = (
            [("train", item) for item in items[:train_count]]
            + [("val", item) for item in items[train_count: train_count + val_count]]
            + [("test", item) for item in items[train_count + val_count:]]
        )

        for split_name, (image_path, label_path) in assignments:
            image_target = output_root / "images" / split_name / image_path.name
            label_target = output_root / "labels" / split_name / label_path.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(image_target))
            shutil.move(str(label_path), str(label_target))

        return {
            "total_images": total,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "train_ratio": round((train_count / total), 6) if total else 0.0,
            "val_ratio": round((val_count / total), 6) if total else 0.0,
            "test_ratio": round((test_count / total), 6) if total else 0.0,
            "seed": cfg["seed"],
            "shuffle": bool(cfg["shuffle"]),
        }

    def convert_dataset(
        self,
        source_root: Path,
        output_root: Path,
        *,
        label_mapping: Optional[dict[str, str]] = None,
        label_filters: Optional[list[str]] = None,
        publish_config: Optional[dict] = None,
        split_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        source_root = Path(source_root).expanduser().resolve(strict=False)
        output_root = Path(output_root).expanduser().resolve(strict=False)
        if not source_root.exists() or not source_root.is_dir():
            raise ValidationError("Dataset root not found for conversion")

        output_root.mkdir(parents=True, exist_ok=True)
        effective_mapping = self._build_effective_mapping(label_mapping, label_filters)
        slice_config = self._normalize_slice_config(publish_config)

        pairs, warnings, unmatched_files = self._collect_pairs(source_root)
        if not pairs:
            detail = "; ".join(unmatched_files[:5])
            suffix = f". Skipped unmatched files: {detail}" if detail else ""
            raise ValidationError(f"No image/json pairs found for illegal dataset publish{suffix}")
        if callable(progress_callback):
            progress_callback(
                "converting",
                {
                    "message": f"已匹配 {len(pairs)} 组图片和标注，开始转换",
                    "processed": 0,
                    "completed": 0,
                    "total": len(pairs),
                    "skipped": len(unmatched_files),
                },
            )

        global_label_map: Dict[str, int] = {}
        processed = 0
        completed = 0
        skipped_files: List[str] = list(unmatched_files)
        successful_labels: set[str] = set()
        aggregate_stats = {"images": 0, "slices": 0, "labels": 0, "empty_slices": 0}

        for image_path, json_path in pairs:
            cfg = self._build_pair_cfg(
                source_root=source_root,
                output_root=output_root,
                image_path=image_path,
                json_path=json_path,
                label_mapping=effective_mapping if effective_mapping else None,
                slice_config=slice_config,
                label_map=global_label_map,
            )
            try:
                stats, global_label_map = self._run_single(cfg)
            except ValidationError as exc:
                skipped_files.append(f"{json_path.name}: {exc}")
                warnings.append(f"Skipped {json_path.name}: {exc}")
                completed += 1
                if callable(progress_callback):
                    progress_callback(
                        "converting",
                        {
                            "message": f"跳过 {json_path.name}: {exc}",
                            "processed": processed,
                            "completed": completed,
                            "total": len(pairs),
                            "skipped": len(skipped_files),
                            "current_file": json_path.name,
                        },
                    )
                continue

            processed += 1
            completed += 1
            aggregate_stats["images"] += 1
            aggregate_stats["slices"] += int(stats.get("total", 0))
            aggregate_stats["labels"] += int(stats.get("total_labels", 0))
            aggregate_stats["empty_slices"] += int(stats.get("empty", 0))
            if callable(progress_callback):
                progress_callback(
                    "converting",
                    {
                        "message": f"已转换 {processed}/{len(pairs)} 组数据",
                        "processed": processed,
                        "completed": completed,
                        "total": len(pairs),
                        "skipped": len(skipped_files),
                        "current_file": json_path.name,
                    },
                )

            try:
                raw_bboxes, _ = parse_annotations({**cfg, "label_map": dict(global_label_map)})
                successful_labels.update({str(bbox.label) for bbox in raw_bboxes if str(bbox.label).strip()})
            except Exception:
                pass

        if processed == 0:
            skipped_summary = "; ".join(skipped_files[:5])
            raise ValidationError(
                f"All {len(pairs)} image/json pairs failed conversion. Details: {skipped_summary}"
            )
        if aggregate_stats["slices"] <= 0:
            raise ValidationError("No valid YOLO samples were generated during publish conversion")

        if callable(progress_callback):
            progress_callback(
                "converting",
                {
                    "message": "转换完成，正在整理切分与类别信息",
                    "processed": processed,
                    "completed": completed,
                    "total": len(pairs),
                    "skipped": len(skipped_files),
                },
            )
        split_summary = self.apply_split(output_root, split_config=split_config)
        class_names = self._write_class_files(output_root, global_label_map, successful_labels, split_summary)

        return {
            "pairs_total": len(pairs),
            "pairs_processed": processed,
            "pairs_skipped": len(skipped_files),
            "skipped_details": skipped_files,
            "warnings": warnings,
            "class_names": class_names,
            "stats": aggregate_stats,
            "split_summary": split_summary,
            "normalized_slice_config": {
                "enabled": bool(slice_config["slice_enabled"]),
                "slice_size": int(slice_config["slice_size"]),
                "overlap": float(slice_config["overlap"]),
                "padding": int(slice_config["padding"]),
                "min_area_ratio": float(slice_config["min_area_ratio"]),
                "min_visibility": float(slice_config["min_visibility"]),
                "min_pixel_size": int(slice_config["min_pixel_size"]),
                "negative_ratio": float(slice_config["negative_ratio"]),
                "empty_positive_action": str(slice_config["empty_positive_action"]),
                "output_format": str(slice_config["output_format"]),
            },
        }
