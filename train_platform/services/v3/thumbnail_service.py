from __future__ import annotations

import os
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps, UnidentifiedImageError

from train_platform.core.config import settings
from train_platform.utils.image_exts import IMAGE_EXTS
from train_platform.utils.exceptions import NotFoundError, ValidationError


logger = logging.getLogger(__name__)


class ThumbnailService:
    """
    Generate and cache thumbnails for dataset images.

    Strategy: pre-generation on upload + file-based cache under BASE_DATASETS_DIR/.thumbnails/.
    Static files are served directly via /static/thumbnails/ for maximum performance.
    """

    _IMAGE_EXTS = IMAGE_EXTS

    def _thumbnail_base(
        self,
        *,
        dataset_id: int,
        dataset_namespace: str | None = None,
        cache_prefix: str | None = None,
    ) -> Path:
        base = settings.thumbnails_dir
        ns = str(dataset_namespace or "").strip().replace("\\", "/").strip("/\\")
        if ns:
            ns_path = Path(ns)
            if ns_path.is_absolute() or ".." in ns_path.parts:
                raise ValidationError("Unsafe dataset_namespace")
            base = base / ns_path
        base = base / str(int(dataset_id))
        if cache_prefix:
            cp = str(cache_prefix).strip().replace("\\", "/").strip("/\\")
            cp_path = Path(cp)
            if not cp or cp_path.is_absolute() or ".." in cp_path.parts:
                raise ValidationError("Unsafe cache_prefix")
            base = base / cp_path
        return base.resolve(strict=False)

    def pregenerate_for_dataset(
        self,
        *,
        dataset_id: int,
        dataset_root: Path,
        size: int = 200,
        max_workers: int = 4,
        dataset_namespace: str | None = None,
        cache_prefix: str | None = None,
        rel_paths: list[str] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        """
        Batch pre-generate thumbnails for all images in a dataset.
        
        Args:
            dataset_id: Dataset ID for cache directory
            dataset_root: Root path of the dataset
            size: Thumbnail size (max edge length)
            max_workers: Number of parallel workers
            cache_prefix: Optional cache namespace under dataset thumbnail root
            rel_paths: Optional explicit relative image paths to generate first
            on_progress: Optional callback (completed, total)
            
        Returns:
            dict with generated, skipped, failed counts
        """
        dataset_root = Path(dataset_root).resolve(strict=False)
        if not dataset_root.exists() or not dataset_root.is_dir():
            return {"generated": 0, "skipped": 0, "failed": 0, "errors": ["Dataset root not found"]}
        
        # Collect all image files
        image_files: list[Path] = []
        if rel_paths:
            seen_paths: set[Path] = set()
            for raw_rel in rel_paths:
                rel = self._safe_rel_path(raw_rel)
                src = (dataset_root / rel).resolve(strict=False)
                if dataset_root not in src.parents and src != dataset_root:
                    continue
                if not src.exists() or not src.is_file():
                    continue
                if src.suffix.lower() not in self._IMAGE_EXTS:
                    continue
                if src in seen_paths:
                    continue
                seen_paths.add(src)
                image_files.append(src)
        else:
            for ext in self._IMAGE_EXTS:
                image_files.extend(dataset_root.rglob(f"*{ext}"))
                image_files.extend(dataset_root.rglob(f"*{ext.upper()}"))
            # Remove duplicates
            image_files = list(set(image_files))
        total = len(image_files)
        
        if total == 0:
            return {"generated": 0, "skipped": 0, "failed": 0, "errors": []}
        
        generated = 0
        skipped = 0
        failed = 0
        errors = []
        completed = 0
        
        def process_one(img_path: Path) -> tuple[str, str | None]:
            """Process single image, return (status, error_msg)"""
            try:
                rel = img_path.relative_to(dataset_root)
                rel_str = rel.as_posix()
                
                # Check if thumbnail already exists and is fresh
                thumb_base = self._thumbnail_base(
                    dataset_id=int(dataset_id),
                    dataset_namespace=dataset_namespace,
                    cache_prefix=cache_prefix,
                )
                thumb = (thumb_base / rel).with_suffix(".webp").resolve(strict=False)
                
                if thumb.exists():
                    try:
                        src_mtime = float(img_path.stat().st_mtime)
                        if float(thumb.stat().st_mtime) >= src_mtime:
                            return ("skipped", None)
                    except Exception:
                        pass
                
                # Generate thumbnail
                thumb.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(dir=str(thumb.parent), prefix=f"{thumb.name}.", suffix=".tmp")
                os.close(fd)
                tmp = Path(tmp_name)
                try:
                    self._render_thumbnail(img_path, tmp, size=size)
                    os.replace(tmp, thumb)
                    return ("generated", None)
                finally:
                    try:
                        if tmp.exists():
                            tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                        
            except Exception as e:
                return ("failed", str(e))
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one, img): img for img in image_files}
            
            for future in as_completed(futures):
                completed += 1
                status, err = future.result()
                
                if status == "generated":
                    generated += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    if err:
                        errors.append(err)
                
                if on_progress and completed % 10 == 0:
                    try:
                        on_progress(completed, total)
                    except Exception:
                        pass
        
        logger.info(f"Thumbnail pre-generation for dataset {dataset_id}: {generated} generated, {skipped} skipped, {failed} failed")
        
        return {
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
            "total": total,
            "errors": errors[:10],  # Limit error list
        }

    def ensure_thumbnail(
        self,
        *,
        dataset_id: int,
        dataset_root: Path,
        file_rel_path: str,
        size: int = 200,
        dataset_namespace: str | None = None,
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

        thumb_base = self._thumbnail_base(
            dataset_id=int(dataset_id),
            dataset_namespace=dataset_namespace,
            cache_prefix=cache_prefix,
        )
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
        quality = int(os.getenv("THUMBNAIL_WEBP_QUALITY", "75"))  # Lower default for speed
        quality = max(1, min(quality, 100))

        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im = self._to_rgb(im)
                # Use BILINEAR for faster generation (good balance of speed/quality)
                im.thumbnail((int(size), int(size)), Image.Resampling.BILINEAR)

                # Prefer WEBP for size; if not supported, fall back to JPEG bytes.
                try:
                    im.save(dst, format="WEBP", quality=quality, method=4)  # method=4 is faster
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
