from __future__ import annotations

from pathlib import Path

from train_platform.core.config import settings
from train_platform.platform.filesystem.paths import ensure_under, safe_relative_path
from train_platform.utils.exceptions import ValidationError


def datasets_root() -> Path:
    return settings.datasets_dir.resolve(strict=False)


def ensure_dataset_relative_path(value: str | Path | None) -> Path:
    return safe_relative_path(value)


def to_storage_token(path: Path) -> str:
    resolved = ensure_under(Path(path), datasets_root(), "dataset storage path")
    return resolved.relative_to(datasets_root()).as_posix()


def resolve_storage_token(token: str | Path) -> Path:
    rel = ensure_dataset_relative_path(token)
    return ensure_under(datasets_root() / rel, datasets_root(), "dataset storage path")


def resolve_legacy_dataset_path(raw_path: str | Path | None) -> Path:
    """Resolve legacy training references while keeping normal tokens strict."""
    base = datasets_root()
    if not raw_path:
        return base

    value = str(raw_path).strip().replace("\\", "/")
    marker = "/static/datasets/"
    if marker in value:
        value = value.split(marker, 1)[1]
    absolute = Path(value)
    if absolute.is_absolute() and absolute.exists():
        return absolute.resolve(strict=False)
    value = value.strip("/\\")
    if not value:
        return base

    path = Path(value)
    if ".." in path.parts:
        return base
    return (base / path).resolve(strict=False)


def resolve_dataset_storage_path(root: Path, rel_path: str | Path) -> Path:
    base = Path(root).resolve(strict=False)
    rel = ensure_dataset_relative_path(rel_path)
    return ensure_under(base / rel, base, "dataset file path")
