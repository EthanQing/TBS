from __future__ import annotations

import shutil
import tempfile
import math
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.illegal_dataset import (
    IllegalDataset,
    IllegalDatasetEvent,
    IllegalDatasetImage,
    IllegalDatasetLabelMapping,
    IllegalDatasetVersion,
)
from train_platform.models.v3.enums import DatasetVersionStatus
from train_platform.repositories.v3.illegal_dataset_repo import IllegalDatasetRepository
from train_platform.repositories.v3.illegal_dataset_version_repo import IllegalDatasetVersionRepository
from train_platform.services.v3.dataset_common import (
    build_annotations_payload,
    build_file_listing,
    build_statistics,
    build_view_payload_from_index,
    build_yolo_view_index,
    detect_split_from_relpath,
    iter_image_files,
    load_cached_json_file,
    read_class_names,
    resolve_storage_token,
    safe_extract_zip,
    dataset_thumbnail_url,
    to_storage_token,
    unpack_uploaded_archive,
    write_cached_json_file,
)
from train_platform.services.v3.illegal_dataset_cas import (
    build_manifest,
    cas_path_for_hash,
    extract_json_labels_from_manifest,
    image_rel_paths_from_manifest,
    illegal_dataset_file_url,
    illegal_dataset_temp_root,
    illegal_manifest_path,
    legacy_snapshot_file_path,
    load_version_manifest,
    manifest_cas_file_path,
    manifest_entry,
    manifest_files,
    manifest_stats_to_dataset_statistics,
    materialize_manifest_to_dir,
    materialize_snapshot_to_dir,
    read_class_names_from_manifest,
    read_yolo_box_summary_from_manifest,
    read_yolo_boxes_from_manifest,
    remove_tree,
    replace_dir_from_manifest,
    replace_dir_from_snapshot,
    scan_tree_to_cas_files,
    write_manifest,
)
from train_platform.services.v3.illegal_dataset_publish_service import IllegalDatasetPublishService
from train_platform.services.v3.thumbnail_service import ThumbnailService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS


LABEL_MAPPING_STATUS_KEEP = "keep"
LABEL_MAPPING_STATUS_DELETE = "delete"
LABEL_MAPPING_DELETE_SENTINEL = "__DISCARD__"


_VERSION_CREATE_LOCKS: dict[int, threading.RLock] = {}
_VERSION_CREATE_LOCKS_GUARD = threading.Lock()


class IllegalDatasetService:
    def __init__(self) -> None:
        self.repo = IllegalDatasetRepository()
        self.version_repo = IllegalDatasetVersionRepository()

    def _dataset_lock(self, illegal_dataset_id: int) -> threading.RLock:
        dataset_id = int(illegal_dataset_id)
        with _VERSION_CREATE_LOCKS_GUARD:
            lock = _VERSION_CREATE_LOCKS.get(dataset_id)
            if lock is None:
                lock = threading.RLock()
                _VERSION_CREATE_LOCKS[dataset_id] = lock
            return lock

    def _normalize_label_mapping_status(self, status: Any, mapped_label: str | None = None) -> str:
        if str(mapped_label or "").strip() == LABEL_MAPPING_DELETE_SENTINEL:
            return LABEL_MAPPING_STATUS_DELETE
        normalized = str(status or LABEL_MAPPING_STATUS_KEEP).strip().lower()
        if normalized in {LABEL_MAPPING_STATUS_DELETE, "discard", "drop", "remove", "删除", "丢弃", "忽略"}:
            return LABEL_MAPPING_STATUS_DELETE
        return LABEL_MAPPING_STATUS_KEEP

    def _effective_label_mapping_value(self, mapping: IllegalDatasetLabelMapping) -> str:
        status = self._normalize_label_mapping_status(
            getattr(mapping, "status", LABEL_MAPPING_STATUS_KEEP),
            str(mapping.mapped_label or ""),
        )
        if status == LABEL_MAPPING_STATUS_DELETE:
            return LABEL_MAPPING_DELETE_SENTINEL
        return str(mapping.mapped_label or "").strip()

    def _normalize_label_mapping_override_value(self, value: Any) -> str | None:
        """Normalize publish-time mapping overrides.

        Accepts the legacy shape {"raw": "mapped"} and the status-aware shape
        {"raw": {"mapped_label": "...", "status": "delete"}}.
        """
        if isinstance(value, dict):
            mapped_label = str(
                value.get("mapped_label")
                or value.get("target_label")
                or value.get("target")
                or value.get("mapped")
                or ""
            ).strip()
            status = self._normalize_label_mapping_status(value.get("status"), mapped_label)
        else:
            mapped_label = str(value or "").strip()
            status = self._normalize_label_mapping_status(None, mapped_label)
        if status == LABEL_MAPPING_STATUS_DELETE:
            return LABEL_MAPPING_DELETE_SENTINEL
        if not mapped_label:
            return None
        return mapped_label

    def _next_dataset_id(self, db: Session) -> int:
        current_max = db.query(func.max(IllegalDataset.illegal_dataset_id)).scalar()
        start = int(settings.illegal_dataset_id_start)
        if current_max is None:
            return start
        return max(start, int(current_max) + 1)

    def _root_path(self, dataset: IllegalDataset) -> Path:
        return resolve_storage_token(dataset.storage_path)

    def _version_root(self, illegal_dataset_id: int, version: int) -> Path:
        return settings.datasets_dir / "illegal" / ".versions" / str(int(illegal_dataset_id)) / f"v{int(version)}"

    def _version_view_index_cache_path(self, dataset: IllegalDataset, version: IllegalDatasetVersion) -> Path:
        root = self._version_root(int(dataset.illegal_dataset_id), int(version.version))
        root.parent.mkdir(parents=True, exist_ok=True)
        return root.with_suffix(".view_index.json")

    def _version_raw_labels_cache_path(self, dataset: IllegalDataset, version: IllegalDatasetVersion) -> Path:
        root = self._version_root(int(dataset.illegal_dataset_id), int(version.version))
        root.parent.mkdir(parents=True, exist_ok=True)
        return root.with_suffix(".raw_labels.json")

    def _version_statistics_cache_path(self, dataset: IllegalDataset, version: IllegalDatasetVersion) -> Path:
        root = self._version_root(int(dataset.illegal_dataset_id), int(version.version))
        root.parent.mkdir(parents=True, exist_ok=True)
        return root.with_suffix(".stats.json")

    def _ensure_name_available(self, db: Session, name: str, *, exclude_id: int | None = None) -> None:
        row = self.repo.get_by_name(db, str(name).strip())
        if row and (exclude_id is None or int(row.illegal_dataset_id) != int(exclude_id)):
            raise ConflictError(f"Illegal dataset '{name}' already exists")

    def _active_version(self, db: Session, dataset: IllegalDataset) -> IllegalDatasetVersion | None:
        if dataset.active_version_id is None:
            return None
        return db.query(IllegalDatasetVersion).filter(IllegalDatasetVersion.version_id == int(dataset.active_version_id)).first()

    def _legacy_snapshot_root(self, version: IllegalDatasetVersion) -> Path:
        token = str(version.snapshot_path or "").strip()
        if not token:
            raise NotFoundError("Illegal dataset version has no manifest or snapshot path")
        root = resolve_storage_token(token)
        if not root.exists() or not root.is_dir():
            raise NotFoundError("Illegal dataset snapshot path not found")
        return root

    def _add_event(
        self,
        db: Session,
        dataset_id: int,
        event_type: str,
        *,
        version_id: int | None = None,
        message: str | None = None,
        created_by: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> IllegalDatasetEvent:
        row = IllegalDatasetEvent(
            illegal_dataset_id=int(dataset_id),
            version_id=int(version_id) if version_id is not None else None,
            event_type=str(event_type),
            message=message,
            created_by=created_by,
            data=data,
        )
        db.add(row)
        db.flush()
        return row

    def _index_version_images(self, db: Session, dataset: IllegalDataset, version: IllegalDatasetVersion) -> None:
        db.query(IllegalDatasetImage).filter(IllegalDatasetImage.version_id == int(version.version_id)).delete()
        manifest = load_version_manifest(version)
        if manifest:
            image_paths = image_rel_paths_from_manifest(manifest)
        else:
            root = resolve_storage_token(str(version.snapshot_path or dataset.storage_path))
            image_paths = [image_path.relative_to(root).as_posix() for image_path in iter_image_files(root)]
        for rel in image_paths:
            db.add(
                IllegalDatasetImage(
                    illegal_dataset_id=int(dataset.illegal_dataset_id),
                    version_id=int(version.version_id),
                    path=rel,
                    split=detect_split_from_relpath(rel),
                )
            )
        db.flush()

    def _refresh_version_raw_labels_cache(
        self,
        dataset: IllegalDataset,
        version: IllegalDatasetVersion,
        *,
        manifest: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> list[str]:
        labels: set[str] = set()
        if manifest is not None:
            labels.update(str(label).strip() for label in read_class_names_from_manifest(manifest) if str(label).strip())
            labels.update(str(label).strip() for label in extract_json_labels_from_manifest(manifest) if str(label).strip())
        elif root is not None:
            labels.update(str(label).strip() for label in read_class_names(root) if str(label).strip())
            labels.update(
                str(label).strip()
                for label in IllegalDatasetPublishService().extract_dataset_labels(root)
                if str(label).strip()
            )
        payload = {"labels": sorted(label for label in labels if label)}
        write_cached_json_file(self._version_raw_labels_cache_path(dataset, version), payload)
        return list(payload["labels"])

    def _load_version_raw_labels(
        self,
        dataset: IllegalDataset,
        version: IllegalDatasetVersion,
        *,
        manifest: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> list[str]:
        cached = load_cached_json_file(self._version_raw_labels_cache_path(dataset, version))
        if isinstance(cached, dict):
            labels = cached.get("labels")
            if isinstance(labels, list):
                return [str(label).strip() for label in labels if str(label).strip()]
        return self._refresh_version_raw_labels_cache(dataset, version, manifest=manifest, root=root)

    def _refresh_version_statistics_cache(
        self,
        db: Session,
        dataset: IllegalDataset,
        version: IllegalDatasetVersion,
        *,
        root: Path,
    ) -> dict[str, Any]:
        image_count = (
            db.query(IllegalDatasetImage)
            .filter(IllegalDatasetImage.version_id == int(version.version_id))
            .count()
        )
        total_files = int(version.file_count) if version.file_count is not None else None
        total_size_bytes = int(version.size_bytes) if version.size_bytes is not None else None
        stats = build_statistics(
            root,
            image_count=image_count,
            total_files=total_files,
            total_size_bytes=total_size_bytes,
        )
        write_cached_json_file(self._version_statistics_cache_path(dataset, version), stats)
        return stats

    def _load_version_statistics(
        self,
        db: Session,
        dataset: IllegalDataset,
        version: IllegalDatasetVersion,
        *,
        root: Path,
    ) -> dict[str, Any]:
        cached = load_cached_json_file(self._version_statistics_cache_path(dataset, version))
        if isinstance(cached, dict):
            return cached
        return self._refresh_version_statistics_cache(db, dataset, version, root=root)

    def _refresh_version_view_index_cache(
        self,
        db: Session,
        dataset: IllegalDataset,
        version: IllegalDatasetVersion,
        *,
        manifest: dict[str, Any] | None = None,
        snapshot_root: Path | None = None,
    ) -> dict[str, Any]:
        if manifest is not None:
            class_names = read_class_names_from_manifest(manifest)
            image_paths = image_rel_paths_from_manifest(manifest)
            entries = [
                {"id": int(idx), "path": rel_path, "name": Path(rel_path).name}
                for idx, rel_path in enumerate(image_paths, start=1)
            ]

            def _process(entry: dict[str, Any]) -> dict[str, Any]:
                width, height, object_count, classes = read_yolo_box_summary_from_manifest(
                    manifest,
                    entry["path"],
                    class_names,
                )
                return {
                    **entry,
                    "width": width,
                    "height": height,
                    "object_count": int(object_count),
                    "classes": [int(x) for x in classes],
                }

            workers = max(1, int(settings.view_index_max_workers or 1))
            if len(entries) <= 1 or workers <= 1:
                items = [_process(entry) for entry in entries]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(workers, max(1, len(entries)))) as executor:
                    items = list(executor.map(_process, entries))

            category_counts: dict[int, int] = {}
            for item in items:
                for class_id in item.get("classes", []):
                    category_counts[int(class_id)] = int(category_counts.get(int(class_id), 0)) + 1
            view_index = {
                "schema_version": 1,
                "categories": [
                    {
                        "class_id": int(class_id),
                        "name": class_names[class_id] if 0 <= int(class_id) < len(class_names) else str(class_id),
                        "count": int(count),
                    }
                    for class_id, count in sorted(category_counts.items())
                ],
                "items": items,
                "total_items": len(items),
            }
        else:
            if snapshot_root is None:
                snapshot_root = self._legacy_snapshot_root(version)
            image_rows = (
                db.query(IllegalDatasetImage)
                .filter(IllegalDatasetImage.version_id == int(version.version_id))
                .order_by(IllegalDatasetImage.path.asc())
                .all()
            )
            if not image_rows:
                image_rows = [
                    {"id": int(idx), "path": image_path.relative_to(snapshot_root).as_posix()}
                    for idx, image_path in enumerate(iter_image_files(snapshot_root), start=1)
                ]
            view_index = build_yolo_view_index(snapshot_root, image_rows, max_workers=settings.view_index_max_workers)

        write_cached_json_file(self._version_view_index_cache_path(dataset, version), view_index)
        self._prewarm_version_thumbnails(dataset, version, view_index)
        return view_index

    def _load_version_view_index(
        self,
        db: Session,
        dataset: IllegalDataset,
        version: IllegalDatasetVersion,
        *,
        manifest: dict[str, Any] | None = None,
        snapshot_root: Path | None = None,
    ) -> dict[str, Any]:
        cached = load_cached_json_file(self._version_view_index_cache_path(dataset, version))
        if isinstance(cached, dict):
            return cached
        return self._refresh_version_view_index_cache(
            db,
            dataset,
            version,
            manifest=manifest,
            snapshot_root=snapshot_root,
        )

    def _prewarm_version_thumbnails(self, dataset: IllegalDataset, version: IllegalDatasetVersion, view_index: dict[str, Any]) -> None:
        if int(dataset.active_version_id or 0) != int(version.version_id):
            return
        limit = max(0, int(settings.thumbnail_first_page_prewarm or 0))
        if limit <= 0:
            return
        items = view_index.get("items") if isinstance(view_index, dict) else []
        if not isinstance(items, list) or not items:
            return
        rel_paths = [str(item.get("path") or "") for item in items[:limit] if str(item.get("path") or "")]
        if not rel_paths:
            return
        try:
            ThumbnailService().pregenerate_for_dataset(
                dataset_id=int(dataset.illegal_dataset_id),
                dataset_root=self._root_path(dataset),
                size=int(settings.thumbnail_size or 200),
                max_workers=int(settings.thumbnail_max_workers or 4),
                dataset_namespace="illegal",
                cache_prefix=f"v{int(version.version_id)}",
                rel_paths=rel_paths,
            )
        except Exception:
            pass

    def _version_files_for_inheritance(
        self,
        version: IllegalDatasetVersion | None,
    ) -> dict[str, dict[str, Any]]:
        if not version:
            return {}
        manifest = load_version_manifest(version)
        if manifest:
            return {
                str(rel): {
                    "hash": str(entry.get("hash") or ""),
                    "size": int(entry.get("size") or entry.get("size_bytes") or 0),
                    "mtime": float(entry.get("mtime") or 0.0),
                }
                for rel, entry in manifest_files(manifest).items()
            }
        snapshot_root = self._legacy_snapshot_root(version)
        return scan_tree_to_cas_files(snapshot_root)

    def _create_version_from_tree(
        self,
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
    ) -> IllegalDatasetVersion:
        latest = self.version_repo.get_latest(db, int(dataset.illegal_dataset_id))
        version_no = int(latest.version) + 1 if latest else 1
        parent_version_id = int(parent_version.version_id) if parent_version else None
        inherited_files = base_files or {}
        files = scan_tree_to_cas_files(source_root, base_files=inherited_files)
        manifest = build_manifest(
            dataset_id=int(dataset.illegal_dataset_id),
            version=version_no,
            parent_version_id=parent_version_id,
            files=files,
            parent_files=inherited_files,
        )
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
            meta={**(event_data or {}), "manifest_schema_version": int(manifest.get("schema_version") or 1)},
            created_by=created_by,
        )
        db.add(row)
        db.flush()
        replace_dir_from_manifest(manifest, self._root_path(dataset))
        dataset.active_version_id = int(row.version_id)
        self._index_version_images(db, dataset, row)
        self._refresh_version_raw_labels_cache(dataset, row, manifest=manifest)
        self._refresh_version_view_index_cache(db, dataset, row, manifest=manifest)
        self._add_event(
            db,
            int(dataset.illegal_dataset_id),
            event_type,
            version_id=int(row.version_id),
            message=event_message or f"Illegal dataset version v{version_no} created",
            created_by=created_by,
            data={**(event_data or {}), "version": version_no},
        )
        db.flush()
        return row

    def _effective_illegal_class_count(
        self,
        db: Session,
        dataset: IllegalDataset,
        *,
        raw_labels: list[str] | None = None,
        fallback_count: int = 0,
    ) -> int:
        labels: set[str] = {str(label).strip() for label in (raw_labels or []) if str(label).strip()}

        mapping: dict[str, str] = {}
        rows = (
            db.query(IllegalDatasetLabelMapping)
            .filter(IllegalDatasetLabelMapping.illegal_dataset_id == int(dataset.illegal_dataset_id))
            .all()
        )
        for row in rows:
            raw_label = str(row.raw_label or "").strip()
            mapped_label = self._effective_label_mapping_value(row)
            if not raw_label:
                continue
            labels.add(raw_label)
            if mapped_label:
                mapping[raw_label] = mapped_label

        effective_labels: set[str] = set()
        for label in labels:
            mapped_label = str(mapping.get(label, label) or "").strip()
            if mapped_label and mapped_label != "__DISCARD__":
                effective_labels.add(mapped_label)

        return max(int(fallback_count or 0), len(effective_labels))

    def _build_dataset_statistics(
        self,
        db: Session,
        dataset: IllegalDataset,
        *,
        version: IllegalDatasetVersion | None = None,
    ) -> dict[str, Any]:
        active_version = version or self._active_version(db, dataset)
        raw_labels: list[str] = []
        if active_version:
            manifest = load_version_manifest(active_version)
            if manifest:
                root = None
                stats = manifest_stats_to_dataset_statistics(manifest)
                raw_labels = self._load_version_raw_labels(dataset, active_version, manifest=manifest)
            else:
                root = self._legacy_snapshot_root(active_version)
                stats = self._load_version_statistics(db, dataset, active_version, root=root)
                raw_labels = self._load_version_raw_labels(dataset, active_version, root=root)
        else:
            root = self._root_path(dataset)
            manifest = None
            stats = build_statistics(root, image_count=0)
            raw_labels = []

        class_count = self._effective_illegal_class_count(
            db,
            dataset,
            raw_labels=raw_labels,
            fallback_count=int(stats.get("num_classes") or stats.get("class_count") or 0),
        )
        stats["num_classes"] = int(class_count)
        stats["class_count"] = int(class_count)
        return stats

    def _dataset_with_statistics(self, db: Session, dataset: IllegalDataset) -> dict[str, Any]:
        statistics = self._build_dataset_statistics(db, dataset)
        return {
            "illegal_dataset_id": int(dataset.illegal_dataset_id),
            "name": dataset.name,
            "dataset_type": dataset.dataset_type,
            "format": dataset.format,
            "storage_path": dataset.storage_path,
            "description": dataset.description,
            "active_version_id": dataset.active_version_id,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "statistics": statistics,
            "preview_image_url": self._first_image_preview_url(db, dataset, statistics=statistics),
        }

    def _first_image_preview_url(
        self,
        db: Session,
        dataset: IllegalDataset,
        *,
        statistics: dict[str, Any] | None = None,
    ) -> str | None:
        if statistics is not None and int(
            statistics.get("num_images") or statistics.get("total_images") or statistics.get("image_count") or 0
        ) <= 0:
            return None
        version = self._active_version(db, dataset)
        if not version:
            return None

        row = (
            db.query(IllegalDatasetImage)
            .filter(IllegalDatasetImage.version_id == int(version.version_id))
            .order_by(IllegalDatasetImage.image_id.asc())
            .first()
        )
        rel_path = str(getattr(row, "path", "") or "").strip() if row else ""

        if not rel_path:
            manifest = load_version_manifest(version)
            if manifest:
                paths = image_rel_paths_from_manifest(manifest)
                rel_path = str(paths[0]).strip() if paths else ""

        if not rel_path:
            try:
                snapshot_root = self._legacy_snapshot_root(version)
                images = iter_image_files(snapshot_root)
                first_image = images[0] if images else None
                if first_image:
                    rel_path = first_image.relative_to(snapshot_root).as_posix()
            except Exception:
                rel_path = ""

        if not rel_path:
            return None
        return dataset_thumbnail_url(
            "illegal",
            int(dataset.illegal_dataset_id),
            rel_path,
            version_id=int(version.version_id),
            size=int(settings.thumbnail_size or 200),
        )

    def list_datasets(self, db: Session, *, skip: int = 0, limit: int = 100, format: str | None = None) -> list[dict[str, Any]]:
        q = db.query(IllegalDataset)
        if format:
            q = q.filter(IllegalDataset.format == str(format))
        rows = q.order_by(IllegalDataset.updated_at.desc()).offset(skip).limit(limit).all()
        return [self._dataset_with_statistics(db, row) for row in rows]

    def create_dataset(self, db: Session, *, obj: dict) -> IllegalDataset:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        fmt = str(obj.get("format") or "yolo").strip().lower() or "yolo"
        if fmt != "yolo":
            raise ValidationError("Only YOLO dataset format is supported")
        self._ensure_name_available(db, name)
        row = IllegalDataset(
            illegal_dataset_id=self._next_dataset_id(db),
            name=name,
            dataset_type=obj["dataset_type"],
            format=fmt,
            storage_path="pending/illegal",
            description=obj.get("description"),
        )
        db.add(row)
        db.flush()
        row.storage_path = f"illegal/{int(row.illegal_dataset_id)}"
        self._root_path(row).mkdir(parents=True, exist_ok=True)
        self._add_event(db, int(row.illegal_dataset_id), "created", message="Illegal dataset created")
        db.commit()
        db.refresh(row)
        return row

    def get_dataset(self, db: Session, illegal_dataset_id: int) -> IllegalDataset:
        row = self.repo.get(db, int(illegal_dataset_id))
        if not row:
            raise NotFoundError("Illegal dataset not found")
        return row

    def update_dataset(self, db: Session, illegal_dataset_id: int, *, patch: dict) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        if "name" in patch and patch["name"] is not None:
            new_name = str(patch["name"]).strip()
            if not new_name:
                raise ValidationError("name cannot be empty")
            self._ensure_name_available(db, new_name, exclude_id=int(row.illegal_dataset_id))
            row.name = new_name
        if "description" in patch:
            row.description = patch["description"]
        db.commit()
        db.refresh(row)
        return row

    def delete_dataset(self, db: Session, illegal_dataset_id: int, *, delete_files: bool = False, force: bool = False) -> None:
        row = self.get_dataset(db, illegal_dataset_id)
        root = self._root_path(row)
        version_root = settings.datasets_dir / "illegal" / ".versions" / str(int(row.illegal_dataset_id))
        db.delete(row)
        db.commit()
        if delete_files:
            try:
                remove_tree(root)
            except Exception:
                pass
            try:
                remove_tree(version_root)
            except Exception:
                pass

    def upload_archive(
        self,
        db: Session,
        illegal_dataset_id: int,
        upload,
        *,
        message: str | None = None,
        created_by: str | None = None,
        append: bool = False,
    ) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        with self._dataset_lock(int(row.illegal_dataset_id)):
            temp_dir = Path(tempfile.mkdtemp(dir=illegal_dataset_temp_root()))
            try:
                extracted_root = unpack_uploaded_archive(upload, temp_dir)
                parent_version = self._active_version(db, row) if append else self.version_repo.get_latest(db, int(row.illegal_dataset_id))
                inherited_files = self._version_files_for_inheritance(parent_version) if append and parent_version else {}
                self._create_version_from_tree(
                    db,
                    row,
                    source_root=extracted_root,
                    base_files=inherited_files,
                    parent_version=parent_version,
                    message=message,
                    created_by=created_by,
                    event_type="appended" if append else "uploaded",
                    event_message="Illegal dataset archive appended" if append else "Illegal dataset archive uploaded",
                    event_data={"filename": str(getattr(upload, 'filename', '') or ''), "append": bool(append)},
                )
                db.commit()
                db.refresh(row)
                return row
            finally:
                try:
                    remove_tree(temp_dir)
                except Exception:
                    pass

    def import_archive_file(
        self,
        db: Session,
        illegal_dataset_id: int,
        archive_path: Path,
        *,
        message: str | None = None,
        created_by: str | None = None,
        append: bool = False,
        filename: str | None = None,
    ) -> IllegalDataset:
        staging = illegal_dataset_temp_root() / f"import-{int(illegal_dataset_id)}-{uuid.uuid4().hex}"
        extracted_dir = staging / "extracted"
        try:
            extracted_root = safe_extract_zip(Path(archive_path), extracted_dir)
            return self.import_source_tree(
                db,
                illegal_dataset_id,
                extracted_root,
                message=message,
                created_by=created_by,
                append=append,
                filename=filename or Path(archive_path).name,
            )
        finally:
            try:
                remove_tree(staging)
            except Exception:
                pass

    def import_source_tree(
        self,
        db: Session,
        illegal_dataset_id: int,
        source_root: Path,
        *,
        message: str | None = None,
        created_by: str | None = None,
        append: bool = False,
        filename: str | None = None,
    ) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        with self._dataset_lock(int(row.illegal_dataset_id)):
            parent_version = self._active_version(db, row) if append else self.version_repo.get_latest(db, int(row.illegal_dataset_id))
            inherited_files = self._version_files_for_inheritance(parent_version) if append and parent_version else {}
            self._create_version_from_tree(
                db,
                row,
                source_root=Path(source_root),
                base_files=inherited_files,
                parent_version=parent_version,
                message=message,
                created_by=created_by,
                event_type="appended" if append else "uploaded",
                event_message="Illegal dataset archive appended" if append else "Illegal dataset archive uploaded",
                event_data={"filename": str(filename or ""), "append": bool(append)},
            )
            db.commit()
            db.refresh(row)
            return row

    def activate_version(self, db: Session, illegal_dataset_id: int, version_id: int) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        with self._dataset_lock(int(row.illegal_dataset_id)):
            version = db.query(IllegalDatasetVersion).filter(
                IllegalDatasetVersion.version_id == int(version_id),
                IllegalDatasetVersion.illegal_dataset_id == int(row.illegal_dataset_id),
            ).first()
            if not version:
                raise NotFoundError("Illegal dataset version not found")
            manifest = load_version_manifest(version)
            if manifest:
                replace_dir_from_manifest(manifest, self._root_path(row))
            else:
                snapshot_root = self._legacy_snapshot_root(version)
                replace_dir_from_snapshot(snapshot_root, self._root_path(row))
            row.active_version_id = int(version.version_id)
            self._add_event(
                db,
                int(row.illegal_dataset_id),
                "activated",
                version_id=int(version.version_id),
                message=f"Activated illegal dataset version v{int(version.version)}",
            )
            db.commit()
            db.refresh(row)
            return row

    def list_versions(self, db: Session, illegal_dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[IllegalDatasetVersion]:
        self.get_dataset(db, illegal_dataset_id)
        return self.version_repo.list_by_dataset(db, int(illegal_dataset_id), skip=skip, limit=limit)

    def list_events(self, db: Session, illegal_dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[IllegalDatasetEvent]:
        self.get_dataset(db, illegal_dataset_id)
        return (
            db.query(IllegalDatasetEvent)
            .filter(IllegalDatasetEvent.illegal_dataset_id == int(illegal_dataset_id))
            .order_by(IllegalDatasetEvent.created_at.desc(), IllegalDatasetEvent.event_id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_detail(self, db: Session, illegal_dataset_id: int, *, versions_limit: int = 20, events_limit: int = 20) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        active_version = self._active_version(db, row)
        return {
            "dataset": row,
            "statistics": self._build_dataset_statistics(db, row, version=active_version),
            "active_version": active_version,
            "versions": self.list_versions(db, int(row.illegal_dataset_id), skip=0, limit=versions_limit),
            "events": self.list_events(db, int(row.illegal_dataset_id), skip=0, limit=events_limit),
        }

    def _selected_version(self, db: Session, row: IllegalDataset, version_id: int | None = None) -> IllegalDatasetVersion:
        version = None
        if version_id is not None:
            version = db.query(IllegalDatasetVersion).filter(
                IllegalDatasetVersion.version_id == int(version_id),
                IllegalDatasetVersion.illegal_dataset_id == int(row.illegal_dataset_id),
            ).first()
        else:
            version = self._active_version(db, row)
        if not version:
            raise ConflictError("Illegal dataset has no active version")
        return version

    def get_statistics(self, db: Session, illegal_dataset_id: int, *, version_id: int | None = None) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        return self._build_dataset_statistics(db, row, version=version)

    def _build_manifest_view_payload(
        self,
        manifest: dict[str, Any],
        *,
        illegal_dataset_id: int,
        version_id: int,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        class_names = read_class_names_from_manifest(manifest)
        image_paths = image_rel_paths_from_manifest(manifest)
        category_image_ids: dict[int, set[int]] = {}
        total_items = len(image_paths)
        start = max(0, (int(page) - 1) * int(page_size))
        end = start + int(page_size)

        boxes_by_rel: dict[str, tuple[int | None, int | None, list[dict[str, Any]]]] = {}
        for idx, rel_path in enumerate(image_paths, start=1):
            width, height, boxes = read_yolo_boxes_from_manifest(manifest, rel_path, class_names)
            boxes_by_rel[rel_path] = (width, height, boxes)
            for box in boxes:
                category_image_ids.setdefault(int(box["class_id"]), set()).add(idx)

        items: list[dict[str, Any]] = []
        for idx, rel_path in enumerate(image_paths[start:end], start=start + 1):
            width, height, boxes = boxes_by_rel.get(rel_path) or read_yolo_boxes_from_manifest(manifest, rel_path, class_names)
            classes = sorted({int(box["class_id"]) for box in boxes})
            items.append(
                {
                    "id": int(idx),
                    "name": Path(rel_path).name,
                    "path": rel_path,
                    "url": illegal_dataset_file_url(illegal_dataset_id, version_id, rel_path),
                    "thumbnail_url": dataset_thumbnail_url(
                        "illegal",
                        int(illegal_dataset_id),
                        rel_path,
                        version_id=int(version_id),
                        size=320,
                    ),
                    "width": width,
                    "height": height,
                    "object_count": len(boxes),
                    "classes": classes,
                }
            )

        categories = [
            {
                "class_id": class_id,
                "name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                "count": len(image_ids),
            }
            for class_id, image_ids in sorted(category_image_ids.items())
        ]
        total_pages = math.ceil(total_items / int(page_size)) if int(page_size) else 1
        return {
            "categories": categories,
            "items": items,
            "meta": {
                "page": int(page),
                "page_size": int(page_size),
                "total_items": int(total_items),
                "total_pages": int(total_pages or 1),
            },
        }

    def _build_manifest_annotations_payload(
        self,
        manifest: dict[str, Any],
        *,
        illegal_dataset_id: int,
        version_id: int,
        image_path: str,
    ) -> dict[str, Any]:
        rel_path = str(image_path or "")
        manifest_entry(manifest, rel_path, required=True)
        class_names = read_class_names_from_manifest(manifest)
        width, height, boxes = read_yolo_boxes_from_manifest(manifest, rel_path, class_names)
        return {
            "image_path": rel_path,
            "image_url": illegal_dataset_file_url(illegal_dataset_id, version_id, rel_path),
            "width": width,
            "height": height,
            "object_count": len(boxes),
            "boxes": boxes,
        }

    def _list_manifest_files(
        self,
        manifest: dict[str, Any],
        *,
        illegal_dataset_id: int,
        version_id: int,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        rel_paths = sorted(manifest_files(manifest))
        total = len(rel_paths)
        start = max(0, (int(page) - 1) * int(page_size))
        end = start + int(page_size)
        items: list[dict[str, Any]] = []
        for rel in rel_paths[start:end]:
            entry = manifest_files(manifest).get(rel) or {}
            exists = False
            try:
                exists = cas_path_for_hash(str(entry.get("hash") or ""), require_exists=False).exists()
            except Exception:
                exists = False
            url = illegal_dataset_file_url(illegal_dataset_id, version_id, rel) if Path(rel).suffix.lower() in IMAGE_EXTS else None
            items.append(
                {
                    "path": rel,
                    "size_bytes": int(entry.get("size") or entry.get("size_bytes") or 0),
                    "mtime": float(entry.get("mtime") or 0.0),
                    "url": url,
                    "exists": bool(exists),
                }
            )
        return items, total

    def get_view(
        self,
        db: Session,
        illegal_dataset_id: int,
        *,
        version_id: int | None = None,
        class_id: int | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        manifest = load_version_manifest(version)
        if manifest:
            view_index = self._load_version_view_index(
                db,
                row,
                version,
                manifest=manifest,
            )
        else:
            snapshot_root = self._legacy_snapshot_root(version)
            view_index = self._load_version_view_index(
                db,
                row,
                version,
                snapshot_root=snapshot_root,
            )
        return build_view_payload_from_index(
            view_index,
            page=page,
            page_size=page_size,
            class_id=class_id,
            file_url_builder=lambda rel_path: illegal_dataset_file_url(
                int(row.illegal_dataset_id),
                int(version.version_id),
                rel_path,
            ),
            thumbnail_url_builder=lambda rel_path: dataset_thumbnail_url(
                "illegal",
                int(row.illegal_dataset_id),
                rel_path,
                version_id=int(version.version_id),
                size=320,
            ),
        )

    def get_image_annotations(self, db: Session, illegal_dataset_id: int, *, image_path: str, version_id: int | None = None) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        manifest = load_version_manifest(version)
        if manifest:
            return self._build_manifest_annotations_payload(
                manifest,
                illegal_dataset_id=int(row.illegal_dataset_id),
                version_id=int(version.version_id),
                image_path=image_path,
            )
        snapshot_root = self._legacy_snapshot_root(version)
        return build_annotations_payload(snapshot_root, str(version.snapshot_path), image_path)

    def list_files(self, db: Session, illegal_dataset_id: int, *, version_id: int | None = None, page: int = 1, page_size: int = 100) -> tuple[list[dict[str, Any]], int]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        manifest = load_version_manifest(version)
        if manifest:
            return self._list_manifest_files(
                manifest,
                illegal_dataset_id=int(row.illegal_dataset_id),
                version_id=int(version.version_id),
                page=page,
                page_size=page_size,
            )
        snapshot_root = self._legacy_snapshot_root(version)
        return build_file_listing(snapshot_root, str(version.snapshot_path), page=page, page_size=page_size)

    def get_version_file_path(self, db: Session, illegal_dataset_id: int, version_id: int, file_path: str) -> Path:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        manifest = load_version_manifest(version)
        if manifest:
            return manifest_cas_file_path(manifest, file_path, required=True)
        snapshot_root = self._legacy_snapshot_root(version)
        return legacy_snapshot_file_path(snapshot_root, file_path)

    def upload_images(
        self,
        db: Session,
        illegal_dataset_id: int,
        *,
        files: list,
        relative_dir: str = "images/uploads",
        message: str | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        with self._dataset_lock(int(row.illegal_dataset_id)):
            temp_dir = Path(tempfile.mkdtemp(dir=illegal_dataset_temp_root()))
            try:
                from train_platform.services.v3.dataset_common import ensure_safe_relative_path

                rel_dir = ensure_safe_relative_path(relative_dir)
                target_dir = temp_dir / rel_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                saved_files: list[str] = []
                total_bytes = 0
                for upload in files:
                    filename = Path(str(getattr(upload, "filename", "") or "")).name
                    if not filename:
                        continue
                    out = target_dir / filename
                    with out.open("wb") as f:
                        upload.file.seek(0)
                        shutil.copyfileobj(upload.file, f)
                    upload.file.seek(0)
                    saved_files.append((rel_dir / filename).as_posix())
                    try:
                        total_bytes += int(out.stat().st_size)
                    except Exception:
                        pass
                parent_version = self._active_version(db, row)
                inherited_files = self._version_files_for_inheritance(parent_version) if parent_version else {}
                version = self._create_version_from_tree(
                    db,
                    row,
                    source_root=temp_dir,
                    base_files=inherited_files,
                    parent_version=parent_version,
                    message=message,
                    created_by=created_by,
                    event_type="images_uploaded",
                    event_message="Illegal dataset images uploaded",
                    event_data={"saved_count": len(saved_files)},
                )
                db.commit()
                db.refresh(row)
                return {
                    "saved_count": len(saved_files),
                    "saved_files": saved_files,
                    "total_bytes": total_bytes,
                    "created_at": version.created_at,
                    "version_id": int(version.version_id),
                    "version": int(version.version),
                    "active_version_id": int(row.active_version_id) if row.active_version_id is not None else None,
                }
            finally:
                try:
                    remove_tree(temp_dir)
                except Exception:
                    pass

    def get_raw_labels(self, db: Session, illegal_dataset_id: int) -> list[str]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row)
        manifest = load_version_manifest(version)
        if manifest:
            labels = set(self._load_version_raw_labels(row, version, manifest=manifest))
        else:
            snapshot_root = self._legacy_snapshot_root(version)
            labels = set(self._load_version_raw_labels(row, version, root=snapshot_root))
        for mapping in db.query(IllegalDatasetLabelMapping).filter(IllegalDatasetLabelMapping.illegal_dataset_id == int(row.illegal_dataset_id)).all():
            labels.add(str(mapping.raw_label))
        return sorted(label for label in labels if label)

    def get_label_mappings(self, db: Session, illegal_dataset_id: int) -> list[IllegalDatasetLabelMapping]:
        self.get_dataset(db, illegal_dataset_id)
        return (
            db.query(IllegalDatasetLabelMapping)
            .filter(IllegalDatasetLabelMapping.illegal_dataset_id == int(illegal_dataset_id))
            .order_by(IllegalDatasetLabelMapping.raw_label.asc())
            .all()
        )

    def update_label_mappings(self, db: Session, illegal_dataset_id: int, *, items: list[dict[str, Any]]) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        existing = {
            str(item.raw_label): item
            for item in db.query(IllegalDatasetLabelMapping).filter(IllegalDatasetLabelMapping.illegal_dataset_id == int(row.illegal_dataset_id)).all()
        }
        seen: set[str] = set()
        delete_count = 0
        for item in items:
            raw_label = str(item.get("raw_label") or "").strip()
            mapped_label = str(item.get("mapped_label") or "").strip()
            status = self._normalize_label_mapping_status(item.get("status"), mapped_label)
            if not raw_label:
                continue
            if status == LABEL_MAPPING_STATUS_DELETE:
                delete_count += 1
                mapped_label = ""
            elif not mapped_label:
                continue
            seen.add(raw_label)
            if raw_label in existing:
                existing[raw_label].mapped_label = mapped_label
                existing[raw_label].status = status
            else:
                db.add(
                    IllegalDatasetLabelMapping(
                        illegal_dataset_id=int(row.illegal_dataset_id),
                        raw_label=raw_label,
                        mapped_label=mapped_label,
                        status=status,
                    )
                )
        for raw_label, record in existing.items():
            if raw_label not in seen:
                db.delete(record)
        self._add_event(
            db,
            int(row.illegal_dataset_id),
            "label_mappings_updated",
            message="Illegal dataset label mappings updated",
            data={"count": len(seen), "delete_count": delete_count},
        )
        db.commit()
        db.refresh(row)
        return row

    def publish_standard_dataset(
        self,
        db: Session,
        illegal_dataset_id: int,
        *,
        obj: dict,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=obj.get("version_id"))
        mapping_rows = self.get_label_mappings(db, int(row.illegal_dataset_id))
        mapping_snapshot = {
            raw_label: mapped_label
            for item in mapping_rows
            for raw_label, mapped_label in [
                (str(item.raw_label or "").strip(), self._effective_label_mapping_value(item))
            ]
            if raw_label and mapped_label
        }
        overrides = obj.get("label_mapping_overrides") or {}
        if isinstance(overrides, dict):
            for raw_label, raw_value in overrides.items():
                raw_label_s = str(raw_label or "").strip()
                mapped_label = self._normalize_label_mapping_override_value(raw_value)
                if raw_label_s and mapped_label:
                    mapping_snapshot[raw_label_s] = mapped_label
        label_filters = [str(x) for x in (obj.get("label_filters") or []) if str(x).strip()]
        publish_config = {
            "source_illegal_dataset_id": int(row.illegal_dataset_id),
            "source_illegal_dataset_name": str(row.name),
            "source_illegal_version_id": int(version.version_id),
            "source_version": int(version.version),
            "label_mappings": mapping_snapshot,
            "label_filters": label_filters,
            "split": obj.get("split") or {},
            **(obj.get("publish_config") or {}),
        }
        from train_platform.services.v3.standard_dataset_service import StandardDatasetService

        temp_dir = Path(tempfile.mkdtemp(dir=illegal_dataset_temp_root()))
        source_root = temp_dir / "illegal_source"
        processed_root = temp_dir / "standard_publish"
        try:
            if callable(progress_callback):
                progress_callback(
                    "materializing",
                    {
                        "message": f"正在准备原始数据集版本 v{int(version.version)}",
                    },
                )
            manifest = load_version_manifest(version)
            if manifest:
                materialize_manifest_to_dir(manifest, source_root, replace=True)
            else:
                snapshot_root = self._legacy_snapshot_root(version)
                materialize_snapshot_to_dir(snapshot_root, source_root, replace=True)
            if callable(progress_callback):
                progress_callback(
                    "converting",
                    {
                        "message": "原始数据已准备完成，开始执行格式转换",
                    },
                )
            publish_result = IllegalDatasetPublishService().convert_dataset(
                source_root,
                processed_root,
                label_mapping=mapping_snapshot,
                label_filters=label_filters,
                publish_config=obj.get("publish_config") or {},
                split_config=obj.get("split") or {},
                progress_callback=progress_callback,
            )
            publish_config["conversion_result"] = {
                "pairs_total": int(publish_result.get("pairs_total", 0)),
                "pairs_processed": int(publish_result.get("pairs_processed", 0)),
                "pairs_skipped": int(publish_result.get("pairs_skipped", 0)),
                "class_names": publish_result.get("class_names") or [],
                "stats": publish_result.get("stats") or {},
                "split_summary": publish_result.get("split_summary"),
                "normalized_slice_config": publish_result.get("normalized_slice_config") or {},
            }

            if callable(progress_callback):
                progress_callback(
                    "publishing",
                    {
                        "message": "转换完成，正在生成标准数据集",
                        "processed": int(publish_result.get("pairs_processed", 0)),
                        "completed": int(publish_result.get("pairs_total", 0)),
                        "total": int(publish_result.get("pairs_total", 0)),
                        "skipped": int(publish_result.get("pairs_skipped", 0)),
                    },
                )
            standard = StandardDatasetService().materialize_from_source_tree(
                db,
                name=str(obj.get("name") or "").strip(),
                dataset_type=row.dataset_type,
                source_root=processed_root,
                description=obj.get("description"),
                source_type="illegal_publish",
                publish_config=publish_config,
            )
            self._add_event(
                db,
                int(row.illegal_dataset_id),
                "published",
                version_id=int(version.version_id),
                message=f"Published standard dataset {standard.name}",
                data={
                    "standard_dataset_id": int(standard.standard_dataset_id),
                    "pairs_processed": int(publish_result.get("pairs_processed", 0)),
                    "pairs_skipped": int(publish_result.get("pairs_skipped", 0)),
                },
            )
            db.commit()
            return {
                "standard_dataset_id": int(standard.standard_dataset_id),
                "name": standard.name,
                "source_illegal_dataset_id": int(row.illegal_dataset_id),
                "source_illegal_version_id": int(version.version_id),
                "publish_config": publish_config,
            }
        finally:
            try:
                remove_tree(temp_dir)
            except Exception:
                pass
