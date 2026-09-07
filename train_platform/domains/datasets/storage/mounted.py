from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from train_platform.platform.filesystem import atomic_write_json, ensure_under, safe_relative_path
from train_platform.utils.exceptions import NotFoundError, ValidationError

from .paths import resolve_dataset_storage_path
from .roots import allowed_import_roots


MOUNTED_MANIFEST_NAME = ".mounted_manifest.json"


def mounted_file_entry(path: Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=False)
    if not source.exists() or not source.is_file():
        raise NotFoundError(f"Mounted source file not found: {source}")
    try:
        stat = source.stat()
    except OSError as exc:
        raise ValidationError(f"Cannot stat mounted source file: {source}") from exc
    return {
        "storage": "mounted",
        "source_path": str(source),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def validate_mounted_source_root(source_root: Path) -> Path:
    source = Path(source_root).expanduser().resolve(strict=False)
    if not any(_is_relative_to(source, allowed) for allowed in allowed_import_roots()):
        raise ValidationError("Mounted source root is not allowed")
    return source


def load_mounted_manifest(root: Path) -> dict[str, Any] | None:
    path = Path(root) / MOUNTED_MANIFEST_NAME
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_mounted_manifest(root: Path, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(Path(root) / MOUNTED_MANIFEST_NAME, dict(payload), sort_keys=True)


def resolve_mounted_file(root: Path, rel_path: str | Path) -> Path | None:
    manifest = load_mounted_manifest(root)
    if not manifest:
        return None
    rel = safe_relative_path(rel_path).as_posix()
    image_prefix = str(manifest.get("image_rel_prefix") or "images").strip().strip("/\\")
    source_image_root = str(manifest.get("source_image_root") or "").strip()
    if not source_image_root:
        return None
    if rel == image_prefix:
        suffix = ""
    elif rel.startswith(f"{image_prefix}/"):
        suffix = rel[len(image_prefix) + 1 :]
    else:
        return None
    source_root = validate_mounted_source_root(Path(source_image_root))
    source = ensure_under(source_root / suffix, source_root, "mounted source file path")
    if not source.exists() or not source.is_file():
        return None
    return source


def resolve_dataset_file(root: Path, rel_path: str | Path) -> Path:
    mounted = resolve_mounted_file(root, rel_path)
    if mounted is not None:
        return mounted
    path = resolve_dataset_storage_path(Path(root), rel_path)
    if not path.exists() or not path.is_file():
        raise NotFoundError("Dataset file not found")
    return path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(base).resolve(strict=False))
        return True
    except ValueError:
        return False
