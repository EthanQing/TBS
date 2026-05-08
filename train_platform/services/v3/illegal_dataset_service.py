from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

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
    build_view_payload,
    clear_directory,
    commit_refresh,
    copy_tree,
    count_tree,
    detect_split_from_relpath,
    iter_image_files,
    overlay_tree,
    read_class_names,
    resolve_storage_token,
    static_dataset_url,
    dataset_thumbnail_url,
    to_storage_token,
    unpack_uploaded_archive,
)
from train_platform.services.v3.illegal_dataset_publish_service import IllegalDatasetPublishService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class IllegalDatasetService:
    def __init__(self) -> None:
        self.repo = IllegalDatasetRepository()
        self.version_repo = IllegalDatasetVersionRepository()

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

    def _ensure_name_available(self, db: Session, name: str, *, exclude_id: int | None = None) -> None:
        row = self.repo.get_by_name(db, str(name).strip())
        if row and (exclude_id is None or int(row.illegal_dataset_id) != int(exclude_id)):
            raise ConflictError(f"Illegal dataset '{name}' already exists")

    def _active_version(self, db: Session, dataset: IllegalDataset) -> IllegalDatasetVersion | None:
        if dataset.active_version_id is None:
            return None
        return db.query(IllegalDatasetVersion).filter(IllegalDatasetVersion.version_id == int(dataset.active_version_id)).first()

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
        root = resolve_storage_token(str(version.snapshot_path or dataset.storage_path))
        db.query(IllegalDatasetImage).filter(IllegalDatasetImage.version_id == int(version.version_id)).delete()
        for image_path in iter_image_files(root):
            rel = image_path.relative_to(root).as_posix()
            db.add(
                IllegalDatasetImage(
                    illegal_dataset_id=int(dataset.illegal_dataset_id),
                    version_id=int(version.version_id),
                    path=rel,
                    split=detect_split_from_relpath(rel),
                )
            )
        db.flush()

    def _create_version_from_root(
        self,
        db: Session,
        dataset: IllegalDataset,
        *,
        message: str | None = None,
        created_by: str | None = None,
        event_type: str = "version_created",
        event_message: str | None = None,
        event_data: dict[str, Any] | None = None,
    ) -> IllegalDatasetVersion:
        root = self._root_path(dataset)
        latest = self.version_repo.get_latest(db, int(dataset.illegal_dataset_id))
        version_no = int(latest.version) + 1 if latest else 1
        snapshot_root = self._version_root(int(dataset.illegal_dataset_id), version_no)
        copy_tree(root, snapshot_root)
        total_files, total_size = count_tree(snapshot_root)
        row = IllegalDatasetVersion(
            illegal_dataset_id=int(dataset.illegal_dataset_id),
            version=version_no,
            parent_version_id=int(latest.version_id) if latest else None,
            status=DatasetVersionStatus.FINALIZED,
            message=message,
            snapshot_path=to_storage_token(snapshot_root),
            manifest_path=None,
            file_count=total_files,
            size_bytes=total_size,
            meta=event_data or {},
            created_by=created_by,
        )
        db.add(row)
        db.flush()
        dataset.active_version_id = int(row.version_id)
        self._index_version_images(db, dataset, row)
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

    def _build_dataset_statistics(
        self,
        db: Session,
        dataset: IllegalDataset,
        *,
        version: IllegalDatasetVersion | None = None,
    ) -> dict[str, Any]:
        active_version = version or self._active_version(db, dataset)
        if active_version:
            root = resolve_storage_token(str(active_version.snapshot_path or dataset.storage_path))
            image_count = (
                db.query(IllegalDatasetImage)
                .filter(IllegalDatasetImage.version_id == int(active_version.version_id))
                .count()
            )
            total_files = int(active_version.file_count) if active_version.file_count is not None else None
            total_size_bytes = int(active_version.size_bytes) if active_version.size_bytes is not None else None
            return build_statistics(
                root,
                image_count=image_count,
                total_files=total_files,
                total_size_bytes=total_size_bytes,
            )
        return build_statistics(self._root_path(dataset), image_count=0)

    def _dataset_with_statistics(self, db: Session, dataset: IllegalDataset) -> dict[str, Any]:
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
            "statistics": self._build_dataset_statistics(db, dataset),
        }

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
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(version_root, ignore_errors=True)

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
        root = self._root_path(row)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=settings.temp_dir))
        try:
            extracted_root = unpack_uploaded_archive(upload, temp_dir)
            if append:
                root.mkdir(parents=True, exist_ok=True)
                overlay_tree(extracted_root, root)
            else:
                copy_tree(extracted_root, root)
            version = self._create_version_from_root(
                db,
                row,
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
            shutil.rmtree(temp_dir, ignore_errors=True)

    def activate_version(self, db: Session, illegal_dataset_id: int, version_id: int) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        version = db.query(IllegalDatasetVersion).filter(
            IllegalDatasetVersion.version_id == int(version_id),
            IllegalDatasetVersion.illegal_dataset_id == int(row.illegal_dataset_id),
        ).first()
        if not version:
            raise NotFoundError("Illegal dataset version not found")
        snapshot_root = resolve_storage_token(str(version.snapshot_path or ""))
        if not snapshot_root.exists():
            raise NotFoundError("Illegal dataset snapshot path not found")
        copy_tree(snapshot_root, self._root_path(row))
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
        images = (
            db.query(IllegalDatasetImage)
            .filter(IllegalDatasetImage.version_id == int(version.version_id))
            .order_by(IllegalDatasetImage.path.asc())
            .all()
        )
        payload = build_view_payload(
            resolve_storage_token(str(version.snapshot_path)),
            str(version.snapshot_path),
            images,
            page=page,
            page_size=page_size,
            thumbnail_url_builder=lambda rel_path: dataset_thumbnail_url(
                "illegal",
                int(row.illegal_dataset_id),
                rel_path,
                version_id=int(version.version_id),
                size=320,
            ),
        )
        if class_id is not None:
            payload["items"] = [item for item in payload["items"] if int(class_id) in item.get("classes", [])]
            payload["meta"]["total_items"] = len(payload["items"])
            payload["meta"]["total_pages"] = 1
        return payload

    def get_image_annotations(self, db: Session, illegal_dataset_id: int, *, image_path: str, version_id: int | None = None) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        return build_annotations_payload(resolve_storage_token(str(version.snapshot_path)), str(version.snapshot_path), image_path)

    def list_files(self, db: Session, illegal_dataset_id: int, *, version_id: int | None = None, page: int = 1, page_size: int = 100) -> tuple[list[dict[str, Any]], int]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=version_id)
        return build_file_listing(resolve_storage_token(str(version.snapshot_path)), str(version.snapshot_path), page=page, page_size=page_size)

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
        root = self._root_path(row)
        target_dir = root / relative_dir
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
            saved_files.append((Path(relative_dir) / filename).as_posix())
            try:
                total_bytes += int(out.stat().st_size)
            except Exception:
                pass
        version = self._create_version_from_root(
            db,
            row,
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

    def get_raw_labels(self, db: Session, illegal_dataset_id: int) -> list[str]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row)
        snapshot_root = resolve_storage_token(str(version.snapshot_path))
        labels = set(read_class_names(snapshot_root))
        labels.update(IllegalDatasetPublishService().extract_dataset_labels(snapshot_root))
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

    def update_label_mappings(self, db: Session, illegal_dataset_id: int, *, items: list[dict[str, str]]) -> IllegalDataset:
        row = self.get_dataset(db, illegal_dataset_id)
        existing = {
            str(item.raw_label): item
            for item in db.query(IllegalDatasetLabelMapping).filter(IllegalDatasetLabelMapping.illegal_dataset_id == int(row.illegal_dataset_id)).all()
        }
        seen: set[str] = set()
        for item in items:
            raw_label = str(item.get("raw_label") or "").strip()
            mapped_label = str(item.get("mapped_label") or "").strip()
            if not raw_label or not mapped_label:
                continue
            seen.add(raw_label)
            if raw_label in existing:
                existing[raw_label].mapped_label = mapped_label
            else:
                db.add(
                    IllegalDatasetLabelMapping(
                        illegal_dataset_id=int(row.illegal_dataset_id),
                        raw_label=raw_label,
                        mapped_label=mapped_label,
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
            data={"count": len(seen)},
        )
        db.commit()
        db.refresh(row)
        return row

    def publish_standard_dataset(self, db: Session, illegal_dataset_id: int, *, obj: dict) -> dict[str, Any]:
        row = self.get_dataset(db, illegal_dataset_id)
        version = self._selected_version(db, row, version_id=obj.get("version_id"))
        snapshot_root = resolve_storage_token(str(version.snapshot_path))
        mapping_rows = self.get_label_mappings(db, int(row.illegal_dataset_id))
        mapping_snapshot = {str(item.raw_label): str(item.mapped_label) for item in mapping_rows}
        overrides = obj.get("label_mapping_overrides") or {}
        if isinstance(overrides, dict):
            mapping_snapshot.update({str(k): str(v) for k, v in overrides.items() if str(k).strip() and str(v).strip()})
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

        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=settings.temp_dir))
        processed_root = temp_dir / "standard_publish"
        try:
            publish_result = IllegalDatasetPublishService().convert_dataset(
                snapshot_root,
                processed_root,
                label_mapping=mapping_snapshot,
                label_filters=label_filters,
                publish_config=obj.get("publish_config") or {},
                split_config=obj.get("split") or {},
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
            shutil.rmtree(temp_dir, ignore_errors=True)
