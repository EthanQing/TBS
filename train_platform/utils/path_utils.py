from __future__ import annotations

from pathlib import Path
from typing import Optional

from train_platform.core.config import settings
from train_platform.utils.exceptions import ValidationError


def _resolve_under_base(
    *,
    raw_path: Optional[str],
    base_dir: Path,
    marker: str,
    label: str,
) -> Path:
    base = base_dir.resolve()
    if not raw_path:
        return base

    p = str(raw_path).strip().replace("\\", "/")
    if marker in p:
        p = p.split(marker, 1)[1]

    p = p.strip("/\\")
    if not p:
        return base

    rel = Path(p)
    if rel.is_absolute():
        raise ValidationError(f"{label} must be a relative path under {base}")
    if ".." in rel.parts:
        raise ValidationError(f"{label} cannot contain parent traversal")

    resolved = (base / rel).resolve(strict=False)
    try:
        resolved.relative_to(base)
    except Exception as e:
        raise ValidationError(f"{label} resolves outside allowed base directory") from e
    return resolved


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
    return _resolve_under_base(
        raw_path=raw_path,
        base_dir=settings.training_dir,
        marker="/static/training/",
        label="training path",
    )


def resolve_temp_path(raw_path: Optional[str]) -> Path:
    return _resolve_under_base(
        raw_path=raw_path,
        base_dir=settings.temp_dir,
        marker="/static/temp/",
        label="temp path",
    )


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
