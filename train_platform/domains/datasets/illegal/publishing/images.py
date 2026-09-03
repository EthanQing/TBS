from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.windows import Window

from train_platform.utils.exceptions import ValidationError

Image.MAX_IMAGE_PIXELS = None

SKIPPABLE_IMAGE_ERRORS = (
    OSError,
    ValueError,
    UnidentifiedImageError,
    rasterio.errors.RasterioError,
)

_WINDOWED_RASTER_EXTS = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".j2k"}
_NUMPY_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_MAX_NUMPY_IMAGE_PIXELS = 100_000_000
_PERCENTILE_SAMPLE_GRID = 4
_PERCENTILE_SAMPLE_SIZE = 256

try:
    from osgeo import gdal

    gdal.UseExceptions()
except Exception:  # pragma: no cover - optional dependency
    gdal = None


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


def _sampled_percentile_range(
    reader: "BaseImageReader", *, sample_grid: int = _PERCENTILE_SAMPLE_GRID
) -> Tuple[float, float]:
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
        result = (
            (float(np.percentile(merged, 2)), float(np.percentile(merged, 98)))
            if merged.size
            else (0.0, 255.0)
        )

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
        self._stretch_stats: Optional[Tuple[float, float]] = (
            None if is_uint8 else _sampled_percentile_range(self)
        )

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
        return np.ascontiguousarray(
            self._cache_rgb[rel_y : rel_y + int(h), rel_x : rel_x + int(w)]
        )

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
        return np.ascontiguousarray(
            self._data[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
        )

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
    errors: list[str] = []
    if _should_use_numpy_reader(image_path):
        try:
            return NumpyImageReader(image_path)
        except SKIPPABLE_IMAGE_ERRORS as exc:
            errors.append(f"PIL: {type(exc).__name__}: {exc}")

    if gdal is not None and ext in _WINDOWED_RASTER_EXTS:
        try:
            return GDALRasterReader(image_path, slice_size=slice_size)
        except SKIPPABLE_IMAGE_ERRORS as exc:
            errors.append(f"GDAL: {type(exc).__name__}: {exc}")
        except Exception as exc:
            errors.append(f"GDAL: {type(exc).__name__}: {exc}")

    if gdal is not None:
        try:
            return GDALRasterReader(image_path, slice_size=slice_size)
        except SKIPPABLE_IMAGE_ERRORS as exc:
            errors.append(f"GDAL: {type(exc).__name__}: {exc}")
        except Exception as exc:
            errors.append(f"GDAL: {type(exc).__name__}: {exc}")

    try:
        return RasterioRasterReader(image_path, slice_size=slice_size)
    except SKIPPABLE_IMAGE_ERRORS as exc:
        errors.append(f"rasterio: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors[-3:]) if errors else "unsupported or unreadable image"
    raise ValidationError(f"Unreadable image file {Path(image_path).name}: {detail}")
