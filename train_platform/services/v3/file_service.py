from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from fastapi import UploadFile

from train_platform.domains.datasets import yolo
from train_platform.domains.datasets.storage.paths import datasets_root
from train_platform.models.v3.enums import DatasetType
from train_platform.platform.filesystem import ensure_under, extract_archive, merge_tree, remove_tree
from train_platform.utils.exceptions import ConflictError, ValidationError


ArchiveProgress = Callable[[int, int, str], None]


class FileService:
    """Transport adapter for dataset archive uploads.

    Archive persistence, extraction, dataset inspection, installation and rollback
    are shared by all three public upload modes. The lower-level operations only
    receive a filename and binary stream, keeping FastAPI out of storage code.
    """

    _ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tar", ".tgz")

    def upload_dataset(self, file: UploadFile, dataset_name: str, dataset_type: DatasetType) -> tuple[Path, dict[str, Any]]:
        name = self._dataset_name(dataset_name)
        filename = self._archive_filename(str(file.filename or ""))
        dataset_dir = datasets_root() / name
        if dataset_dir.exists():
            raise ConflictError(f"Dataset directory '{name}' already exists.")
        return self._install_upload(
            stream=file.file,
            filename=filename,
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            upload_key=name,
        )

    def upload_dataset_into_existing(self, file: UploadFile, dataset_dir: Path, dataset_type: DatasetType) -> tuple[Path, dict[str, Any]]:
        filename = self._archive_filename(str(file.filename or ""))
        target = self._dataset_target(dataset_dir)
        if target.exists():
            try:
                if any(target.iterdir()):
                    raise ConflictError("Dataset directory is not empty.")
            except ConflictError:
                raise
            except OSError:
                pass
            try:
                target.rmdir()
            except OSError:
                pass
        return self._install_upload(
            stream=file.file,
            filename=filename,
            dataset_dir=target,
            dataset_type=dataset_type,
            upload_key=target.name,
        )

    def append_dataset_archive(self, file: UploadFile, dataset_dir: Path, dataset_type: DatasetType) -> tuple[Path, dict[str, Any]]:
        target = self._dataset_target(dataset_dir)
        filename = self._archive_filename(str(file.filename or ""))
        target.mkdir(parents=True, exist_ok=True)
        info: dict[str, Any] = {"added_classes": [], "total_classes": 0}
        with self._prepared_archive(file.file, filename, target.name, suffix="append") as source_root:
            if dataset_type == DatasetType.DETECTION:
                source_root = yolo.find_yolo_export_root(source_root) or source_root
                info = yolo.merge_classes(target, source_root)
            merge_tree(source_root, target, skip_names=yolo.CLASS_FILE_NAMES)
            yolo.validate_dataset_structure(target, dataset_type)
            return target, info

    def _install_upload(
        self,
        *,
        stream: BinaryIO,
        filename: str,
        dataset_dir: Path,
        dataset_type: DatasetType,
        upload_key: str,
        progress_callback: ArchiveProgress | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        filename = self._archive_filename(filename)
        try:
            with self._prepared_archive(stream, filename, upload_key, progress_callback=progress_callback) as extracted_root:
                source_root, info = self._inspect_upload(extracted_root, dataset_type)
                try:
                    shutil.move(str(source_root), str(dataset_dir))
                except Exception:
                    shutil.copytree(source_root, dataset_dir)
                if info.get("format") not in {"labelme", "unknown_json"}:
                    yolo.validate_dataset_structure(dataset_dir, dataset_type)
                return dataset_dir, info
        except Exception:
            self._cleanup(dataset_dir)
            raise

    @contextmanager
    def _prepared_archive(
        self,
        stream: BinaryIO,
        filename: str,
        upload_key: str,
        *,
        suffix: str = "",
        progress_callback: ArchiveProgress | None = None,
    ) -> Iterator[Path]:
        archive_path = self._persist_archive(stream, filename, upload_key)
        staging = self._staging_path(upload_key, suffix=suffix)
        try:
            yield extract_archive(archive_path, staging, progress_callback=progress_callback)
        finally:
            self._cleanup(staging, archive_path)

    @staticmethod
    def _inspect_upload(extracted_root: Path, dataset_type: DatasetType) -> tuple[Path, dict[str, Any]]:
        source_root = Path(extracted_root)
        info: dict[str, Any] = {"format": "yolo", "illegal_reason": None}
        if dataset_type != DatasetType.DETECTION:
            return source_root, info
        detected = yolo.detect_dataset_format(source_root, dataset_type)
        fmt = str(detected.get("format") or "no_images")
        if fmt == "no_images":
            raise ValidationError("No image files found in dataset directory")
        if fmt == "images_only":
            raise ValidationError("No label files found")
        if fmt == "yolo" and detected.get("yolo_root") is not None:
            source_root = Path(detected["yolo_root"])
        elif fmt == "labelme":
            info.update(format="labelme", illegal_reason="labelme_json")
        elif fmt == "unknown_json":
            info.update(format="unknown_json", illegal_reason="unsupported_json")
        return source_root, info

    @staticmethod
    def _dataset_name(value: str) -> str:
        name = str(value or "").strip()
        if not name or any(separator in name for separator in ("/", "\\")) or name in {".", ".."}:
            raise ValidationError("Invalid dataset name.")
        return name

    @staticmethod
    def _archive_filename(value: str) -> str:
        filename = Path(str(value or "")).name
        if not filename.lower().endswith(FileService._ARCHIVE_SUFFIXES):
            raise ValidationError("Unsupported file format.")
        return filename

    @staticmethod
    def _persist_archive(stream: BinaryIO, filename: str, key: str) -> Path:
        root = datasets_root()
        root.mkdir(parents=True, exist_ok=True)
        suffix = "".join(Path(filename).suffixes) or ".zip"
        fd, name = tempfile.mkstemp(prefix=f"_tmp_upload_{key}_", suffix=suffix, dir=str(root))
        path = Path(name)
        try:
            stream.seek(0)
            with os.fdopen(fd, "wb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            try:
                stream.seek(0)
            except (AttributeError, OSError):
                pass
        return path

    @staticmethod
    def _staging_path(key: str, *, suffix: str = "") -> Path:
        root = datasets_root()
        safe_suffix = f"_{suffix}" if suffix else ""
        path = root / f"_tmp_extract_{key}{safe_suffix}"
        remove_tree(path)
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            try:
                path.chmod(0o700)
            except OSError:
                pass
        return path

    @staticmethod
    def _dataset_target(path: Path) -> Path:
        root = datasets_root()
        target = ensure_under(Path(path), root, "dataset directory")
        if target == root:
            raise ValidationError("Dataset directory must be under BASE_DATASETS_DIR")
        return target

    @staticmethod
    def _cleanup(*paths: Path) -> None:
        for path in paths:
            try:
                target = Path(path)
                if target.is_dir() and not target.is_symlink():
                    remove_tree(target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pass
