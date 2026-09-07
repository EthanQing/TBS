from __future__ import annotations

from pathlib import Path

from train_platform.utils.exceptions import ValidationError


def safe_relative_path(value: str | Path | None) -> Path:
    """Return a normalized relative path, rejecting traversal and absolute paths."""
    rel = Path(str(value or "").strip().replace("\\", "/"))
    if not str(rel) or rel.is_absolute() or ".." in rel.parts:
        raise ValidationError("Invalid relative path")
    return rel


def ensure_under(path: Path, root: Path, label: str = "path") -> Path:
    """Resolve *path* and ensure it remains below *root*."""
    resolved = Path(path).resolve(strict=False)
    base = Path(root).resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValidationError(f"Unsafe {label}") from exc
    return resolved
