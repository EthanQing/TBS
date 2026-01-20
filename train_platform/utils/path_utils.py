from __future__ import annotations

from pathlib import Path
from typing import Optional

from train_platform.core.config import settings


def resolve_dataset_path(raw_path: Optional[str]) -> Path:
    base_dir = settings.datasets_dir.resolve()
    if not raw_path:
        return base_dir

    p = str(raw_path).strip().replace("\\", "/")
    marker = "/static/datasets/"
    if marker in p:
        p = p.split(marker, 1)[1]

    # Treat as portable relative path under BASE_DATASETS_DIR by default.
    p = p.strip("/\\")
    if not p:
        return base_dir

    rel = Path(p)

    # If absolute and exists, return as-is (backwards compatibility).
    if rel.is_absolute() and rel.exists():
        return rel.resolve()

    # Prevent path traversal for relative paths.
    if ".." in rel.parts:
        return base_dir

    return (base_dir / rel).resolve(strict=False)


def resolve_training_path(raw_path: Optional[str]) -> Path:
    base_dir = settings.training_dir.resolve()
    if not raw_path:
        return base_dir

    p = str(raw_path).strip().replace("\\", "/")
    marker = "/static/training/"
    if marker in p:
        p = p.split(marker, 1)[1]

    p = p.strip("/\\")
    if not p:
        return base_dir

    abs_candidate = Path(p)
    if abs_candidate.is_absolute() and abs_candidate.exists():
        return abs_candidate.resolve()

    # Keep last segment for safety
    return (base_dir / p).resolve(strict=False)


def resolve_temp_path(raw_path: Optional[str]) -> Path:
    base_dir = settings.temp_dir.resolve()
    if not raw_path:
        return base_dir

    p = str(raw_path).strip().replace("\\", "/")
    marker = "/static/temp/"
    if marker in p:
        p = p.split(marker, 1)[1]

    p = p.strip("/\\")
    if not p:
        return base_dir

    abs_candidate = Path(p)
    if abs_candidate.is_absolute() and abs_candidate.exists():
        return abs_candidate.resolve()

    return (base_dir / p).resolve(strict=False)


def resolve_pretrain_path(raw_path: Optional[str]) -> Path:
    base_dir = settings.pretrain_models_dir.resolve()
    if not raw_path:
        return base_dir

    p = str(raw_path).strip().replace("\\", "/")
    marker = "/static/pretrain/"
    if marker in p:
        p = p.split(marker, 1)[1]

    p = p.strip("/\\")
    if not p:
        return base_dir

    abs_candidate = Path(p)
    if abs_candidate.is_absolute() and abs_candidate.exists():
        return abs_candidate.resolve()

    # Prevent path traversal for relative paths.
    rel = Path(p)
    if ".." in rel.parts:
        return base_dir

    return (base_dir / p).resolve(strict=False)
