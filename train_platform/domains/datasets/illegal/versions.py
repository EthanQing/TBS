from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.datasets.illegal import labels
from train_platform.domains.datasets.illegal.cas import (
    build_manifest,
    extract_json_labels_from_manifest,
    image_rel_paths_from_manifest,
    illegal_dataset_temp_root,
    illegal_manifest_path,
    load_version_manifest,
    manifest_files,
    manifest_stats_to_dataset_statistics,
    normalize_manifest_file_entry,
    read_class_names_from_manifest,
    replace_dir_from_manifest,
    safe_manifest_rel,
    scan_tree_to_cas_files,
    write_manifest,
)
from train_platform.domains.datasets.yolo import detect_split_from_relpath
from train_platform.domains.datasets.illegal.events import add_event
from train_platform.domains.datasets.illegal.mounted import build_illegal_mounted_manifest
from train_platform.domains.datasets.storage.paths import resolve_storage_token, to_storage_token
from train_platform.models.v3.enums import DatasetVersionStatus
from train_platform.models.v3.illegal_dataset import (
    IllegalDataset,
    IllegalDatasetImage,
    IllegalDatasetVersion,
)
from train_platform.platform.filesystem import extract_archive, remove_tree, safe_relative_path
from train_platform.repositories.v3.illegal_dataset_version_repo import IllegalDatasetVersionRepository
from train_platform.services.v3.dataset_common import load_cached_json_file, write_cached_json_file
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


_VERSION_CREATE_LOCKS: dict[int, threading.RLock] = {}
_VERSION_CREATE_LOCKS_GUARD = threading.Lock()
_version_repo = IllegalDatasetVersionRepository()


def dataset_lock(illegal_dataset_id: int) -> threading.RLock:
    dataset_id = int(illegal_dataset_id)
    with _VERSION_CREATE_LOCKS_GUARD:
        lock = _VERSION_CREATE_LOCKS.get(dataset_id)
        if lock is None:
            lock = threading.RLock()
            _VERSION_CREATE_LOCKS[dataset_id] = lock
        return lock


def dataset_root(dataset: IllegalDataset) -> Path:
    return resolve_storage_token(dataset.storage_path)


def version_root(illegal_dataset_id: int, version: int) -> Path:
    return (settings.datasets_dir / "illegal" / ".versions" / str(int(illegal_dataset_id)) / f"v{int(version)}").resolve(strict=False)


def raw_labels_cache_path(dataset: IllegalDataset, version: IllegalDatasetVersion) -> Path:
    root = version_root(int(dataset.illegal_dataset_id), int(version.version))
    root.parent.mkdir(parents=True, exist_ok=True)
    return root.with_suffix(".raw_labels.json")


def active_version(db: Session, dataset: IllegalDataset) -> IllegalDatasetVersion | None:
    if dataset.active_version_id is None:
        return None
    return db.query(IllegalDatasetVersion).filter(
        IllegalDatasetVersion.version_id == int(dataset.active_version_id)
    ).first()


def selected_version(
    db: Session,
    dataset: IllegalDataset,
    version_id: int | None = None,
) -> IllegalDatasetVersion:
    version = None
    if version_id is not None:
        version = db.query(IllegalDatasetVersion).filter(
            IllegalDatasetVersion.version_id == int(version_id),
            IllegalDatasetVersion.illegal_dataset_id == int(dataset.illegal_dataset_id),
        ).first()
    else:
        version = active_version(db, dataset)
    if not version:
        raise ConflictError("Illegal dataset has no active version")
    return version


def lock_dataset_for_version_create(db: Session, dataset: IllegalDataset) -> IllegalDataset:
    row = (
        db.query(IllegalDataset)
        .filter(IllegalDataset.illegal_dataset_id == int(dataset.illegal_dataset_id))
        .with_for_update()
        .first()
    )
    if not row:
        raise NotFoundError("Illegal dataset not found")
    return row


def next_version_no(db: Session, dataset: IllegalDataset) -> int:
    latest_version = (
        db.query(func.max(IllegalDatasetVersion.version))
        .filter(IllegalDatasetVersion.illegal_dataset_id == int(dataset.illegal_dataset_id))
        .scalar()
    )
    return int(latest_version or 0) + 1


def version_files_for_inheritance(version: IllegalDatasetVersion | None) -> dict[str, dict[str, Any]]:
    if not version:
        return {}
    manifest = load_version_manifest(version)
    return {str(rel): dict(entry) for rel, entry in manifest_files(manifest).items()}


def index_version_images(
    db: Session,
    dataset: IllegalDataset,
    version: IllegalDatasetVersion,
    *,
    progress_callback: Callable[..., None] | None = None,
    progress_start: int = 92,
    progress_end: int = 96,
) -> None:
    db.query(IllegalDatasetImage).filter(
        IllegalDatasetImage.version_id == int(version.version_id)
    ).delete()
    manifest = load_version_manifest(version)
    image_paths = image_rel_paths_from_manifest(manifest)
    if not image_paths:
        db.flush()
        return
    chunk_size = 1000
    total = len(image_paths)
    for start in range(0, total, chunk_size):
        chunk = image_paths[start : start + chunk_size]
        db.add_all([
            IllegalDatasetImage(
                illegal_dataset_id=int(dataset.illegal_dataset_id),
                version_id=int(version.version_id),
                path=rel,
                split=detect_split_from_relpath(rel),
            )
            for rel in chunk
        ])
        db.flush()
        done = min(total, start + len(chunk))
        if progress_callback:
            span = max(0, int(progress_end) - int(progress_start))
            progress = int(progress_start) + int(round(span * done / max(1, total)))
            progress_callback(progress, "indexing", {
                "processed_count": done,
                "total_count": total,
                "current_item": chunk[-1] if chunk else None,
                "detail_message": f"Indexed {done}/{total} images",
            })
    db.flush()


def refresh_raw_labels_cache(
    dataset: IllegalDataset,
    version: IllegalDatasetVersion,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    found: set[str] = set()
    if manifest is not None:
        raw_labels = manifest.get("raw_labels")
        json_labels_loaded = isinstance(raw_labels, list)
        if json_labels_loaded:
            found.update(str(label).strip() for label in raw_labels if str(label).strip())
        found.update(str(label).strip() for label in read_class_names_from_manifest(manifest) if str(label).strip())
        if not json_labels_loaded:
            found.update(str(label).strip() for label in extract_json_labels_from_manifest(manifest) if str(label).strip())
    payload = {"labels": sorted(label for label in found if label)}
    write_cached_json_file(raw_labels_cache_path(dataset, version), payload)
    return list(payload["labels"])


def load_raw_labels(
    dataset: IllegalDataset,
    version: IllegalDatasetVersion,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    cached = load_cached_json_file(raw_labels_cache_path(dataset, version))
    if isinstance(cached, dict) and isinstance(cached.get("labels"), list):
        return [str(label).strip() for label in cached["labels"] if str(label).strip()]
    return refresh_raw_labels_cache(dataset, version, manifest=manifest)


def build_statistics(
    db: Session,
    dataset: IllegalDataset,
    *,
    version: IllegalDatasetVersion | None = None,
) -> dict[str, Any]:
    selected = version or active_version(db, dataset)
    if selected:
        manifest = load_version_manifest(selected)
        stats = manifest_stats_to_dataset_statistics(manifest)
        raw = load_raw_labels(dataset, selected, manifest=manifest)
    else:
        stats = empty_statistics()
        raw = []
    class_count = labels.effective_class_count(
        db,
        int(dataset.illegal_dataset_id),
        raw_labels=raw,
        fallback_count=int(stats.get("num_classes") or stats.get("class_count") or 0),
    )
    stats["num_classes"] = int(class_count)
    stats["class_count"] = int(class_count)
    return stats


def empty_statistics() -> dict[str, Any]:
    return {
        "total_files": 0,
        "total_size_bytes": 0,
        "total_size_mb": 0.0,
        "size_mb": 0.0,
        "dataset_size_mb": 0.0,
        "total_images": 0,
        "num_images": 0,
        "image_count": 0,
        "annotations_count": 0,
        "target_count": 0,
        "total_targets": 0,
        "object_count": 0,
        "total_objects": 0,
        "num_classes": 0,
        "class_count": 0,
        "declared_class_count": 0,
        "used_class_count": 0,
    }


def _finalize_version(
    db: Session,
    dataset: IllegalDataset,
    *,
    manifest: dict[str, Any],
    version_no: int,
    parent_version_id: int | None,
    message: str | None,
    created_by: str | None,
    event_type: str,
    event_message: str | None,
    event_data: dict[str, Any] | None = None,
    version_meta: dict[str, Any] | None = None,
    materialize: bool = True,
    progress_callback: Callable[..., None] | None = None,
) -> IllegalDatasetVersion:
    def progress(value: int, stage: str, detail: dict[str, Any] | None = None) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(value, stage, detail or {})
        except TypeError:
            progress_callback(value, stage)

    progress(84, "validating")
    manifest_path = illegal_manifest_path(int(dataset.illegal_dataset_id), version_no)
    write_manifest(manifest, manifest_path)
    stats = manifest_stats_to_dataset_statistics(manifest)
    row = IllegalDatasetVersion(
        illegal_dataset_id=int(dataset.illegal_dataset_id),
        version=version_no,
        parent_version_id=parent_version_id,
        status=DatasetVersionStatus.FINALIZED,
        message=message,
        snapshot_path=None,
        manifest_path=to_storage_token(manifest_path),
        file_count=int(stats.get("total_files") or 0),
        size_bytes=int(stats.get("total_size_bytes") or 0),
        meta={
            **(version_meta or event_data or {}),
            "manifest_schema_version": int(manifest.get("schema_version") or 1),
        },
        created_by=created_by,
    )
    db.add(row)
    db.flush()
    progress(88, "materializing", {"detail_message": "Switching active mounted version"} if not materialize else None)
    if materialize:
        replace_dir_from_manifest(manifest, dataset_root(dataset))
    else:
        remove_tree(dataset_root(dataset), ignore_errors=True)
        dataset_root(dataset).mkdir(parents=True, exist_ok=True)
    dataset.active_version_id = int(row.version_id)
    progress(92, "indexing")
    index_version_images(db, dataset, row, progress_callback=progress_callback)
    refresh_raw_labels_cache(dataset, row, manifest=manifest)
    progress(96, "indexing")
    progress(98, "finalizing")
    add_event(
        db,
        int(dataset.illegal_dataset_id),
        event_type,
        version_id=int(row.version_id),
        message=event_message or f"Illegal dataset version v{version_no} created",
        created_by=created_by,
        data={**(event_data or version_meta or {}), "version": version_no},
    )
    db.flush()
    return row


def _create_version_from_tree(
    db: Session,
    dataset: IllegalDataset,
    *,
    source_root: Path,
    base_files: dict[str, dict[str, Any]] | None = None,
    parent_version: IllegalDatasetVersion | None = None,
    message: str | None = None,
    created_by: str | None = None,
    event_type: str = "version_created",
    event_message: str | None = None,
    event_data: dict[str, Any] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> IllegalDatasetVersion:
    def progress(value: int, stage: str) -> None:
        if progress_callback:
            progress_callback(value, stage)

    version_no = next_version_no(db, dataset)
    inherited_files = base_files or {}
    progress(76, "validating")
    files = scan_tree_to_cas_files(Path(source_root), base_files=inherited_files)
    progress(80, "validating")
    manifest = build_manifest(
        dataset_id=int(dataset.illegal_dataset_id),
        version=version_no,
        parent_version_id=int(parent_version.version_id) if parent_version else None,
        files=files,
        parent_files=inherited_files,
    )
    return _finalize_version(
        db,
        dataset,
        manifest=manifest,
        version_no=version_no,
        parent_version_id=int(parent_version.version_id) if parent_version else None,
        message=message,
        created_by=created_by,
        event_type=event_type,
        event_message=event_message,
        event_data=event_data,
        progress_callback=progress_callback,
    )


def create_from_tree(db: Session, dataset: IllegalDataset, **kwargs) -> IllegalDatasetVersion:
    return _create_version_from_tree(db, dataset, **kwargs)


def import_archive_file(
    db: Session,
    illegal_dataset_id: int,
    archive_path: Path,
    *,
    message: str | None = None,
    created_by: str | None = None,
    append: bool = False,
    filename: str | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> IllegalDataset:
    staging = illegal_dataset_temp_root() / f"import-{int(illegal_dataset_id)}-{uuid.uuid4().hex}"
    try:
        extracted_root = extract_archive(Path(archive_path), staging / "extracted", progress_callback=progress_callback)
        return import_source_tree(
            db,
            illegal_dataset_id,
            extracted_root,
            message=message,
            created_by=created_by,
            append=append,
            filename=filename or Path(archive_path).name,
            progress_callback=progress_callback,
        )
    finally:
        remove_tree(staging, ignore_errors=True)


def import_source_tree(
    db: Session,
    illegal_dataset_id: int,
    source_root: Path,
    *,
    message: str | None = None,
    created_by: str | None = None,
    append: bool = False,
    filename: str | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> IllegalDataset:
    dataset = db.query(IllegalDataset).filter(
        IllegalDataset.illegal_dataset_id == int(illegal_dataset_id)
    ).first()
    if not dataset:
        raise NotFoundError("Illegal dataset not found")
    with dataset_lock(int(dataset.illegal_dataset_id)):
        if progress_callback:
            progress_callback(75, "validating")
        dataset = lock_dataset_for_version_create(db, dataset)
        parent = active_version(db, dataset) if append else _version_repo.get_latest(db, int(dataset.illegal_dataset_id))
        inherited = version_files_for_inheritance(parent) if append and parent else {}
        _create_version_from_tree(
            db,
            dataset,
            source_root=Path(source_root),
            base_files=inherited,
            parent_version=parent,
            message=message,
            created_by=created_by,
            event_type="appended" if append else "uploaded",
            event_message="Illegal dataset archive appended" if append else "Illegal dataset archive uploaded",
            event_data={"filename": str(filename or ""), "append": bool(append)},
            progress_callback=progress_callback,
        )
        db.commit()
        db.refresh(dataset)
        return dataset


def create_from_mounted_source(
    db: Session,
    dataset: IllegalDataset,
    source_root: Path,
    *,
    message: str | None = None,
    created_by: str | None = None,
    append: bool = False,
    filename: str | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> IllegalDatasetVersion:
    def progress(value: int, stage: str, detail: dict[str, Any] | None = None) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(value, stage, detail or {})
        except TypeError:
            progress_callback(value, stage)

    def manifest_progress(value: int, stage: str, detail: dict[str, Any] | None = None) -> None:
        progress(60 + int(round(15 * max(0, min(100, int(value))) / 100)), stage, detail)

    with dataset_lock(int(dataset.illegal_dataset_id)):
        progress(60, "linking", {"detail_message": "Preparing mounted import"})
        dataset = lock_dataset_for_version_create(db, dataset)
        latest = _version_repo.get_latest(db, int(dataset.illegal_dataset_id))
        version_no = next_version_no(db, dataset)
        parent_version_id = int(latest.version_id) if latest and append else None
        mounted = build_illegal_mounted_manifest(
            Path(source_root),
            prefer_yolo=True,
            progress_callback=manifest_progress,
            max_workers=settings.dataset_import_max_workers,
        )
        progress(75, "validating", {"detail_message": "Writing mounted manifest"})
        files = {
            safe_manifest_rel(rel): normalize_manifest_file_entry(entry)
            for rel, entry in (mounted.get("files") or {}).items()
            if isinstance(entry, dict)
        }
        manifest = build_manifest(
            dataset_id=int(dataset.illegal_dataset_id),
            version=version_no,
            parent_version_id=parent_version_id,
            files=files,
        )
        raw_labels = [str(label).strip() for label in (mounted.get("raw_labels") or []) if str(label).strip()]
        image_count = int(mounted.get("image_count") or 0)
        object_count = int(mounted.get("object_count") or 0)
        manifest.update({
            "source_type": "mounted_dir_link",
            "format": mounted.get("format"),
            "source_root": mounted.get("source_root"),
            "source_image_root": mounted.get("source_image_root"),
            "image_rel_prefix": mounted.get("image_rel_prefix"),
            "image_paths": mounted.get("image_paths") or [],
            "raw_labels": raw_labels,
            "warnings": mounted.get("warnings") or [],
        })
        manifest["stats"] = {
            **(manifest.get("stats") or {}),
            "image_count": image_count,
            "total_images": image_count,
            "num_images": image_count,
            "json_count": int(mounted.get("json_count") or 0),
            "target_count": object_count,
            "annotations_count": object_count,
            "total_targets": object_count,
            "object_count": object_count,
            "total_objects": object_count,
            "class_count": len(raw_labels),
            "num_classes": len(raw_labels),
            "declared_class_count": len(raw_labels),
        }
        version_meta = {
            "source_type": "mounted_dir_link",
            "source_root": str(Path(source_root).resolve(strict=False)),
            "format": mounted.get("format"),
            "link_type": "manifest",
            "image_count": image_count,
            "json_count": int(mounted.get("json_count") or 0),
            "filename": str(filename or ""),
            "append": bool(append),
            "append_semantics": "new_mounted_version",
            "lightweight_import": True,
        }
        return _finalize_version(
            db,
            dataset,
            manifest=manifest,
            version_no=version_no,
            parent_version_id=parent_version_id,
            message=message,
            created_by=created_by,
            event_type="mounted_appended" if append else "mounted_imported",
            event_message="Illegal dataset imported from mounted directory",
            event_data=version_meta,
            version_meta=version_meta,
            materialize=False,
            progress_callback=progress_callback,
        )


def import_mounted_source_tree(
    db: Session,
    illegal_dataset_id: int,
    source_root: Path,
    **kwargs,
) -> IllegalDataset:
    dataset = db.query(IllegalDataset).filter(
        IllegalDataset.illegal_dataset_id == int(illegal_dataset_id)
    ).first()
    if not dataset:
        raise NotFoundError("Illegal dataset not found")
    create_from_mounted_source(db, dataset, source_root, **kwargs)
    db.commit()
    db.refresh(dataset)
    return dataset


def activate(db: Session, dataset: IllegalDataset, version_id: int) -> IllegalDataset:
    with dataset_lock(int(dataset.illegal_dataset_id)):
        version = db.query(IllegalDatasetVersion).filter(
            IllegalDatasetVersion.version_id == int(version_id),
            IllegalDatasetVersion.illegal_dataset_id == int(dataset.illegal_dataset_id),
        ).first()
        if not version:
            raise NotFoundError("Illegal dataset version not found")
        manifest = load_version_manifest(version)
        meta = version.meta if isinstance(version.meta, dict) else {}
        if str(meta.get("source_type") or "") == "mounted_dir_link":
            remove_tree(dataset_root(dataset), ignore_errors=True)
            dataset_root(dataset).mkdir(parents=True, exist_ok=True)
        else:
            replace_dir_from_manifest(manifest, dataset_root(dataset))
        dataset.active_version_id = int(version.version_id)
        add_event(
            db,
            int(dataset.illegal_dataset_id),
            "activated",
            version_id=int(version.version_id),
            message=f"Activated illegal dataset version v{int(version.version)}",
        )
        db.commit()
        db.refresh(dataset)
        return dataset


def list_versions(db: Session, illegal_dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[IllegalDatasetVersion]:
    return _version_repo.list_by_dataset(db, int(illegal_dataset_id), skip=skip, limit=limit)


def upload_images(
    db: Session,
    dataset: IllegalDataset,
    *,
    files: list,
    relative_dir: str = "images/uploads",
    message: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    with dataset_lock(int(dataset.illegal_dataset_id)):
        dataset = lock_dataset_for_version_create(db, dataset)
        temp_dir = Path(tempfile.mkdtemp(dir=illegal_dataset_temp_root()))
        try:
            rel_dir = safe_relative_path(relative_dir)
            target_dir = temp_dir / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            saved_files: list[str] = []
            total_bytes = 0
            for upload in files:
                filename = Path(str(getattr(upload, "filename", "") or "")).name
                if not filename:
                    continue
                out = target_dir / filename
                with out.open("wb") as target:
                    upload.file.seek(0)
                    shutil.copyfileobj(upload.file, target)
                upload.file.seek(0)
                saved_files.append((rel_dir / filename).as_posix())
                total_bytes += int(out.stat().st_size)
            parent = active_version(db, dataset)
            inherited = version_files_for_inheritance(parent) if parent else {}
            version = _create_version_from_tree(
                db,
                dataset,
                source_root=temp_dir,
                base_files=inherited,
                parent_version=parent,
                message=message,
                created_by=created_by,
                event_type="images_uploaded",
                event_message="Illegal dataset images uploaded",
                event_data={"saved_count": len(saved_files)},
            )
            db.commit()
            db.refresh(dataset)
            return {
                "saved_count": len(saved_files),
                "saved_files": saved_files,
                "total_bytes": total_bytes,
                "created_at": version.created_at,
                "version_id": int(version.version_id),
                "version": int(version.version),
                "active_version_id": int(dataset.active_version_id) if dataset.active_version_id is not None else None,
            }
        finally:
            remove_tree(temp_dir, ignore_errors=True)
