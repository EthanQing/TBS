from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from train_platform.core.config import settings
from train_platform.platform.filesystem import (
    atomic_write_json,
    atomic_write_text,
    remove_tree,
)
from train_platform.utils.exceptions import ConflictError, NotFoundError

CUSTOM_MODELS_DIRNAME = "custom_models"
ARCHIVE_FILENAME = "source.zip"
MANIFEST_JSON_FILENAME = "manifest.json"
SHA256_FILENAME = "sha256"


def get_custom_models_base_dir() -> Path:
    """Return the base storage directory for custom model packages."""
    custom_dir = Path(os.getenv("BASE_CUSTOM_MODELS_DIR") or (settings.home_dir / CUSTOM_MODELS_DIRNAME)).resolve()
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir


def package_dir_path(package_id: int) -> Path:
    """Return the immutable directory for a specific package_id."""
    return get_custom_models_base_dir() / str(package_id)


def package_archive_path(package_id: int) -> Path:
    """Return the path to the stored immutable archive file for package_id."""
    return package_dir_path(package_id) / ARCHIVE_FILENAME


def package_manifest_path(package_id: int) -> Path:
    """Return the path to the stored normalized manifest json for package_id."""
    return package_dir_path(package_id) / MANIFEST_JSON_FILENAME


def package_sha256_path(package_id: int) -> Path:
    """Return the path to the stored sha256 checksum file for package_id."""
    return package_dir_path(package_id) / SHA256_FILENAME


def compute_file_sha256(file_path: Path, *, chunk_size: int = 65536) -> str:
    """Compute sha256 hex digest of a file in binary mode."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def store_package_archive(
    package_id: int,
    source_archive_path: Path,
    manifest_dict: dict[str, Any],
    source_sha256: str,
) -> Path:
    """Store the immutable package archive, manifest.json, and sha256.
    
    Raises ConflictError if the package storage directory already has an archive.
    """
    pkg_dir = package_dir_path(package_id)
    archive_dest = package_archive_path(package_id)
    manifest_dest = package_manifest_path(package_id)
    sha256_dest = package_sha256_path(package_id)

    if archive_dest.exists():
        raise ConflictError(f"Custom model package {package_id} archive already exists and cannot be overwritten")

    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Atomic copy of source archive
    temp_archive = pkg_dir / f".{ARCHIVE_FILENAME}.tmp"
    try:
        shutil.copy2(source_archive_path, temp_archive)
        os.replace(temp_archive, archive_dest)
    finally:
        if temp_archive.exists():
            temp_archive.unlink(missing_ok=True)

    atomic_write_json(manifest_dest, manifest_dict)
    atomic_write_text(sha256_dest, f"{source_sha256}\n")

    return archive_dest


def resolve_package_archive(package_id: int) -> Path:
    """Resolve and verify existence of the package archive file."""
    archive = package_archive_path(package_id)
    if not archive.is_file():
        raise NotFoundError(f"Custom model package archive not found for package_id={package_id}")
    return archive


def remove_staging_dir(staging_dir: Path) -> None:
    """Remove a temporary staging directory safely."""
    remove_tree(staging_dir, ignore_errors=True)


def remove_package_dir(package_id: int) -> None:
    """Remove the package directory if it was created (used for ingestion rollback compensation)."""
    remove_tree(package_dir_path(package_id), ignore_errors=True)


__all__ = [
    "ARCHIVE_FILENAME",
    "CUSTOM_MODELS_DIRNAME",
    "MANIFEST_JSON_FILENAME",
    "SHA256_FILENAME",
    "compute_file_sha256",
    "get_custom_models_base_dir",
    "package_archive_path",
    "package_dir_path",
    "package_manifest_path",
    "package_sha256_path",
    "remove_package_dir",
    "remove_staging_dir",
    "resolve_package_archive",
    "store_package_archive",
]
