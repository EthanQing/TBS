from __future__ import annotations

import concurrent.futures
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

from train_platform.utils.exceptions import ConflictError, ValidationError
from train_platform.utils.zip_encoding import safe_zip_member_relpath


ProgressCallback = Callable[[int, int, str], None]


def _located_root(destination: Path) -> Path:
    children = [item for item in destination.iterdir() if item.name.lower() != "__macosx"]
    files = [item for item in children if item.is_file()]
    directories = [item for item in children if item.is_dir()]
    if not files and len(directories) == 1:
        return directories[0]
    return destination


def _archive_key(path: Path) -> str:
    value = path.as_posix()
    return value.lower() if os.name == "nt" else value


def _register_archive_path(seen: dict[str, tuple[str, str]], rel: Path, kind: str) -> None:
    key = _archive_key(rel)
    original = rel.as_posix()
    previous = seen.get(key)
    if previous is not None and not (previous == ("dir", original) and kind == "dir"):
        raise ConflictError(f"Duplicate archive path: {original}")
    seen[key] = (kind, original)


def _ensure_archive_target(root: Path, rel: Path) -> Path:
    target = (root / rel).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValidationError("Unsafe archive extraction path") from exc
    return target


def extract_zip(
    archive_path: Path,
    destination: Path,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve(strict=False)
    try:
        with zipfile.ZipFile(Path(archive_path), "r") as archive:
            seen: dict[str, tuple[str, str]] = {}
            directories: set[Path] = set()
            files: list[tuple[zipfile.ZipInfo, Path]] = []
            for info in archive.infolist():
                rel = safe_zip_member_relpath(info)
                # ZIP marks symbolic links using the Unix mode in external_attr.
                mode = (int(info.external_attr) >> 16) & 0xFFFF
                if mode and (mode & 0o170000) == 0o120000:
                    raise ValidationError("Symlinks are not allowed in archives")
                kind = "dir" if info.is_dir() else "file"
                _register_archive_path(seen, rel, kind)
                _ensure_archive_target(root, rel)
                if info.is_dir():
                    directories.add(rel)
                else:
                    if rel.parent != Path("."):
                        for parent in rel.parents:
                            if parent == Path("."):
                                break
                            _register_archive_path(seen, parent, "dir")
                        directories.add(rel.parent)
                    files.append((info, rel))

            for rel in sorted(directories, key=lambda value: (len(value.parts), value.as_posix())):
                if rel != Path("."):
                    target = _ensure_archive_target(root, rel)
                    if target.exists() and not target.is_dir():
                        raise ConflictError(f"Duplicate path in zip: {rel.as_posix()}")
                    target.mkdir(parents=True, exist_ok=True)

            total = len(files)
            if progress_callback and total == 0:
                progress_callback(0, 0, "")
            buffer_size = max(16 * 1024, min(int(os.getenv("ARCHIVE_COPY_BUFSIZE", str(1024 * 1024))), 16 * 1024 * 1024))
            threshold = max(1, int(os.getenv("ZIP_EXTRACT_PARALLEL_THRESHOLD", "256")))
            workers = int(os.getenv("ZIP_EXTRACT_WORKERS", "8"))
            if workers <= 0:
                workers = min(8, os.cpu_count() or 4)
            archive_name = str(getattr(archive, "filename", "") or "")
            parallel = workers > 1 and total >= threshold and bool(archive_name) and Path(archive_name).exists()

            def extract_members(members: list[tuple[zipfile.ZipInfo, Path]], zip_name: str | None = None) -> int:
                handle = archive if zip_name is None else zipfile.ZipFile(zip_name, "r")
                try:
                    for info, rel in members:
                        target = _ensure_archive_target(root, rel)
                        if target.exists():
                            raise ConflictError(f"Duplicate file path in zip: {rel.as_posix()}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with handle.open(info) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output, length=buffer_size)
                    return len(members)
                finally:
                    if zip_name is not None:
                        handle.close()

            if not parallel:
                done = 0
                for info, rel in files:
                    done += extract_members([(info, rel)])
                    if progress_callback:
                        progress_callback(done, total, rel.as_posix())
            else:
                chunk_size = max(16, min(int(os.getenv("ZIP_EXTRACT_CHUNK_SIZE", "128")), 2048))
                chunks = [files[index : index + chunk_size] for index in range(0, len(files), chunk_size)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                    futures = [executor.submit(extract_members, chunk, archive_name) for chunk in chunks]
                    done = 0
                    for future in concurrent.futures.as_completed(futures):
                        done += future.result()
                        if progress_callback:
                            progress_callback(done, total, "")
    except (ValidationError, ConflictError):
        raise
    except Exception as exc:
        raise ValidationError(f"Unsupported or invalid ZIP archive: {exc}") from exc
    return _located_root(destination)


def extract_tar(
    archive_path: Path,
    destination: Path,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve(strict=False)
    try:
        with tarfile.open(Path(archive_path), "r:*") as archive:
            members = archive.getmembers()
            seen: dict[str, tuple[str, str]] = {}
            files = [member for member in members if not member.isdir()]
            if progress_callback and not files:
                progress_callback(0, 0, "")
            processed_files = 0
            for member in members:
                rel = Path(str(member.name or "").replace("\\", "/"))
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValidationError("Unsafe tar content path")
                if member.issym() or member.islnk():
                    raise ValidationError("Symlinks are not allowed in archives")
                kind = "dir" if member.isdir() else "file"
                _register_archive_path(seen, rel, kind)
                if not member.isdir():
                    for parent in rel.parents:
                        if parent == Path("."):
                            break
                        _register_archive_path(seen, parent, "dir")
                target = _ensure_archive_target(root, rel)
                if member.isdir():
                    if target.exists() and not target.is_dir():
                        raise ConflictError(f"Duplicate path in tar: {rel.as_posix()}")
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists():
                    raise ConflictError(f"Duplicate file path in tar: {rel.as_posix()}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is not None:
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                processed_files += 1
                if progress_callback:
                    progress_callback(processed_files, len(files), rel.as_posix())
    except (ValidationError, ConflictError):
        raise
    except Exception as exc:
        raise ValidationError(f"Unsupported or invalid TAR archive: {exc}") from exc
    return _located_root(destination)


def extract_archive(
    archive_path: Path,
    destination: Path,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    name = Path(archive_path).name.lower()
    if name.endswith(".zip"):
        return extract_zip(archive_path, destination, progress_callback=progress_callback)
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        return extract_tar(archive_path, destination, progress_callback=progress_callback)
    raise ValidationError("Unsupported file format")
