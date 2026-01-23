from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from train_platform.core.config import settings
from train_platform.utils.exceptions import NotFoundError, ValidationError


class ThumbnailService:
    """
    Generate and cache thumbnails for dataset images.

    Strategy: on-demand generation + file-based cache under BASE_DATASETS_DIR/.thumbnails/.
    """

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def ensure_thumbnail(
        self,
        *,
        dataset_id: int,
        dataset_root: Path,
        file_rel_path: str,
        size: int = 200,
        cache_prefix: str | None = None,
    ) -> Path:
        size_i = int(size or 0)
        if size_i <= 0:
            raise ValidationError("size must be a positive integer")
        # Keep thumbnails reasonably small to protect server CPU/memory.
        size_i = max(16, min(size_i, 1024))

        rel = self._safe_rel_path(file_rel_path)

        dataset_root = Path(dataset_root).resolve(strict=False)
        src = (dataset_root / rel).resolve(strict=False)
        if dataset_root not in src.parents and src != dataset_root:
            raise ValidationError("Unsafe file path")
        if not src.exists() or not src.is_file():
            raise NotFoundError("Image not found")
        if src.suffix.lower() not in self._IMAGE_EXTS:
            raise ValidationError("Unsupported image format")

        thumb_base = settings.thumbnails_dir / str(int(dataset_id))
        if cache_prefix:
            cp = str(cache_prefix).strip().replace("\\", "/").strip("/\\")
            cp_path = Path(cp)
            if not cp or cp_path.is_absolute() or ".." in cp_path.parts:
                raise ValidationError("Unsafe cache_prefix")
            thumb_base = thumb_base / cp_path
        thumb_base = thumb_base.resolve(strict=False)
        thumb = (thumb_base / rel).with_suffix(".webp").resolve(strict=False)
        if thumb_base not in thumb.parents and thumb != thumb_base:
            raise ValidationError("Unsafe thumbnail path")

        try:
            # Re-generate if missing or stale.
            src_mtime = float(src.stat().st_mtime)
            if thumb.exists():
                try:
                    if float(thumb.stat().st_mtime) >= src_mtime:
                        return thumb
                except Exception:
                    # If we can't stat the cached thumb, just try to regenerate it.
                    pass
        except Exception:
            # If the source stat fails, fall through and let PIL open raise a meaningful error.
            pass

        thumb.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(dir=str(thumb.parent), prefix=f"{thumb.name}.", suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            self._render_thumbnail(src, tmp, size=size_i)
            os.replace(tmp, thumb)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

        return thumb

    def guess_media_type(self, path: Path) -> str:
        """
        Guess image media type from file signature.

        The cache is expected to be WEBP; we keep this detection as a safety net.
        """
        try:
            with open(path, "rb") as f:
                head = f.read(16)
        except Exception:
            return "application/octet-stream"

        # JPEG
        if head.startswith(b"\xFF\xD8\xFF"):
            return "image/jpeg"
        # PNG
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        # WEBP: RIFF....WEBP
        if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "image/webp"
        return "application/octet-stream"

    def _render_thumbnail(self, src: Path, dst: Path, *, size: int) -> None:
        quality = int(os.getenv("THUMBNAIL_WEBP_QUALITY", "80"))
        quality = max(1, min(quality, 100))

        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im = self._to_rgb(im)
                im.thumbnail((int(size), int(size)), Image.Resampling.LANCZOS)

                # Prefer WEBP for size; if not supported, fall back to JPEG bytes.
                try:
                    im.save(dst, format="WEBP", quality=quality, method=6)
                except Exception:
                    im.save(dst, format="JPEG", quality=max(50, min(quality, 95)), optimize=True)
        except UnidentifiedImageError as e:
            raise ValidationError(f"Invalid image file: {e}") from e

    def _to_rgb(self, im: Image.Image) -> Image.Image:
        # Normalize to RGB for consistent thumbnail output.
        if im.mode == "RGB":
            return im
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in (im.info or {})):
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im.convert("RGBA"))
            return bg.convert("RGB")
        return im.convert("RGB")

    def _safe_rel_path(self, file_rel_path: str) -> Path:
        raw = str(file_rel_path or "").strip().replace("\\", "/").lstrip("/")
        if not raw:
            raise ValidationError("file_path is required")
        rel = Path(raw)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValidationError("Unsafe file path")
        return rel
