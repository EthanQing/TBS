"""Generic file scanning primitives for dataset storage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from train_platform.utils.image_exts import IMAGE_EXTS


_DATASET_INTERNAL_FILE_NAMES = {".dataset_stats.json", ".dataset_view_index.json"}


def iter_files(root: Path) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file() and path.name not in _DATASET_INTERNAL_FILE_NAMES)


def iter_image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in Path(root).rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def count_tree(root: Path) -> tuple[int, int]:
    total_files = 0
    total_size = 0
    for path in iter_files(root):
        total_files += 1
        try:
            total_size += int(path.stat().st_size)
        except Exception:
            pass
    return total_files, total_size


__all__ = ["count_tree", "iter_files", "iter_image_files"]
