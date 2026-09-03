from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from train_platform.core.config import settings
from train_platform.platform.filesystem.paths import safe_relative_path
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS


def _thumbnail_base(
    *,
    dataset_id: int,
    dataset_namespace: str | None = None,
    cache_prefix: str | None = None,
) -> Path:
    base = settings.thumbnails_dir
    namespace = str(dataset_namespace or "").strip().replace("\\", "/").strip("/\\")
    if namespace:
        namespace_path = Path(namespace)
        if namespace_path.is_absolute() or ".." in namespace_path.parts:
            raise ValidationError("Unsafe dataset_namespace")
        base = base / namespace_path

    base = base / str(int(dataset_id))
    if cache_prefix:
        prefix = str(cache_prefix).strip().replace("\\", "/").strip("/\\")
        prefix_path = Path(prefix)
        if not prefix or prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValidationError("Unsafe cache_prefix")
        base = base / prefix_path
    return base.resolve(strict=False)


def thumbnail_cache_path(
    *,
    dataset_id: int,
    relative_path: str,
    dataset_namespace: str | None = None,
    cache_prefix: str | None = None,
) -> Path:
    rel = safe_relative_path(str(relative_path or "").strip().lstrip("/"))
    base = _thumbnail_base(
        dataset_id=int(dataset_id),
        dataset_namespace=dataset_namespace,
        cache_prefix=cache_prefix,
    )
    target = (base / rel).with_suffix(".webp").resolve(strict=False)
    if base not in target.parents and target != base:
        raise ValidationError("Unsafe thumbnail path")
    return target


def ensure_thumbnail(
    *,
    dataset_id: int,
    source_path: Path,
    relative_path: str,
    size: int = 200,
    dataset_namespace: str | None = None,
    cache_prefix: str | None = None,
) -> Path:
    size_i = int(size or 0)
    if size_i <= 0:
        raise ValidationError("size must be a positive integer")
    size_i = max(16, min(size_i, 1024))

    rel = safe_relative_path(str(relative_path or "").strip().lstrip("/"))
    source = Path(source_path).resolve(strict=False)
    if not source.exists() or not source.is_file():
        raise NotFoundError("Image not found")
    if rel.suffix.lower() not in IMAGE_EXTS and source.suffix.lower() not in IMAGE_EXTS:
        raise ValidationError("Unsupported image format")

    thumbnail = thumbnail_cache_path(
        dataset_id=int(dataset_id),
        relative_path=rel.as_posix(),
        dataset_namespace=dataset_namespace,
        cache_prefix=cache_prefix,
    )
    try:
        source_mtime = float(source.stat().st_mtime)
        if thumbnail.exists():
            try:
                if float(thumbnail.stat().st_mtime) >= source_mtime:
                    return thumbnail
            except Exception:
                pass
    except Exception:
        pass

    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(thumbnail.parent),
        prefix=f"{thumbnail.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        _render_thumbnail(source, temporary, size=size_i)
        os.replace(temporary, thumbnail)
    finally:
        try:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
        except Exception:
            pass
    return thumbnail


def prewarm_dataset_thumbnails(
    *,
    dataset_id: int,
    entries: Iterable[tuple[Path, str]],
    size: int = 200,
    max_workers: int | None = None,
    dataset_namespace: str | None = None,
    cache_prefix: str | None = None,
) -> None:
    resolved_entries = list(entries)
    if not resolved_entries:
        return

    workers = max(1, int(max_workers or settings.thumbnail_max_workers or 1))

    def process(entry: tuple[Path, str]) -> None:
        source_path, relative_path = entry
        try:
            ensure_thumbnail(
                dataset_id=int(dataset_id),
                source_path=source_path,
                relative_path=relative_path,
                size=int(size),
                dataset_namespace=dataset_namespace,
                cache_prefix=cache_prefix,
            )
        except Exception:
            pass

    if workers == 1 or len(resolved_entries) == 1:
        for entry in resolved_entries:
            process(entry)
        return

    with ThreadPoolExecutor(max_workers=min(workers, len(resolved_entries))) as executor:
        list(executor.map(process, resolved_entries))


def detect_thumbnail_media_type(path: Path) -> str:
    try:
        with open(path, "rb") as stream:
            header = stream.read(16)
    except Exception:
        return "application/octet-stream"

    if header.startswith(b"\xFF\xD8\xFF"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(header) >= 12 and header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _render_thumbnail(source: Path, destination: Path, *, size: int) -> None:
    quality = int(os.getenv("THUMBNAIL_WEBP_QUALITY", "75"))
    quality = max(1, min(quality, 100))

    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            image = _to_rgb(image)
            image.thumbnail((int(size), int(size)), Image.Resampling.BILINEAR)
            try:
                image.save(destination, format="WEBP", quality=quality, method=4)
            except Exception:
                image.save(destination, format="JPEG", quality=max(50, min(quality, 95)), optimize=True)
    except UnidentifiedImageError as exc:
        raise ValidationError(f"Invalid image file: {exc}") from exc


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in (image.info or {})):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        return background.convert("RGB")
    return image.convert("RGB")


__all__ = [
    "detect_thumbnail_media_type",
    "ensure_thumbnail",
    "prewarm_dataset_thumbnails",
    "thumbnail_cache_path",
]
