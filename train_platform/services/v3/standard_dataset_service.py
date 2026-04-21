from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetEvent, StandardDatasetImage
from train_platform.models.v3.training_run import TrainingRun
from train_platform.repositories.v3.standard_dataset_repo import StandardDatasetRepository
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
    resolve_storage_token,
    static_dataset_url,
    to_storage_token,
    unpack_uploaded_archive,
    utcnow,
)
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class StandardDatasetService:
    def __init__(self) -> None:
        self.repo = StandardDatasetRepository()

    def _root_path(self, dataset: StandardDataset) -> Path:
        return resolve_storage_token(dataset.storage_path)

    def _ensure_name_available(self, db: Session, name: str, *, exclude_id: int | None = None) -> None:
        row = self.repo.get_by_name(db, str(name).strip())
        if row and (exclude_id is None or int(row.standard_dataset_id) != int(exclude_id)):
            raise ConflictError(f"Standard dataset '{name}' already exists")

    def _index_images(self, db: Session, dataset: StandardDataset) -> None:
        root = self._root_path(dataset)
        db.query(StandardDatasetImage).filter(StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)).delete()
        for image_path in iter_image_files(root):
            rel = image_path.relative_to(root).as_posix()
            db.add(
                StandardDatasetImage(
                    standard_dataset_id=int(dataset.standard_dataset_id),
                    path=rel,
                    split=detect_split_from_relpath(rel),
                )
            )
        db.flush()

    def _add_event(
        self,
        db: Session,
        dataset_id: int,
        event_type: str,
        *,
        message: str | None = None,
        created_by: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        db.add(
            StandardDatasetEvent(
                standard_dataset_id=int(dataset_id),
                event_type=str(event_type),
                message=message,
                created_by=created_by,
                data=data,
            )
        )
        db.flush()

    def list_datasets(self, db: Session, *, skip: int = 0, limit: int = 100, format: str | None = None) -> list[StandardDataset]:
        q = db.query(StandardDataset)
        if format:
            q = q.filter(StandardDataset.format == str(format))
        return q.order_by(StandardDataset.updated_at.desc()).offset(skip).limit(limit).all()

    def create_dataset(self, db: Session, *, obj: dict) -> StandardDataset:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        self._ensure_name_available(db, name)
        row = StandardDataset(
            name=name,
            dataset_type=obj["dataset_type"],
            format=str(obj.get("format") or "yolo").strip() or "yolo",
            storage_path="pending/standard",
            description=obj.get("description"),
            source_type=obj.get("source_type"),
            source_illegal_dataset_id=obj.get("source_illegal_dataset_id"),
            source_illegal_version_id=obj.get("source_illegal_version_id"),
            publish_config=obj.get("publish_config"),
        )
        db.add(row)
        db.flush()
        row.storage_path = f"standard/{int(row.standard_dataset_id)}"
        root = self._root_path(row)
        root.mkdir(parents=True, exist_ok=True)
        self._add_event(db, int(row.standard_dataset_id), "created", message="Standard dataset created")
        db.commit()
        db.refresh(row)
        return row

    def get_dataset(self, db: Session, standard_dataset_id: int) -> StandardDataset:
        row = self.repo.get(db, int(standard_dataset_id))
        if not row:
            raise NotFoundError("Standard dataset not found")
        return row

    def update_dataset(self, db: Session, standard_dataset_id: int, *, patch: dict) -> StandardDataset:
        row = self.get_dataset(db, standard_dataset_id)
        if "name" in patch and patch["name"] is not None:
            new_name = str(patch["name"]).strip()
            if not new_name:
                raise ValidationError("name cannot be empty")
            self._ensure_name_available(db, new_name, exclude_id=int(row.standard_dataset_id))
            row.name = new_name
        if "description" in patch:
            row.description = patch["description"]
        db.commit()
        db.refresh(row)
        return row

    def delete_dataset(self, db: Session, standard_dataset_id: int, *, delete_files: bool = False, force: bool = False) -> None:
        row = self.get_dataset(db, standard_dataset_id)
        projects = db.query(Project).filter(Project.standard_dataset_id == int(row.standard_dataset_id)).all()
        runs = db.query(TrainingRun).filter(TrainingRun.standard_dataset_id == int(row.standard_dataset_id)).all()
        if (projects or runs) and not force:
            raise ConflictError("Standard dataset is still referenced by projects or training runs")
        for run in runs:
            db.delete(run)
        for project in projects:
            db.delete(project)
        root = self._root_path(row)
        db.delete(row)
        db.commit()
        if delete_files:
            shutil.rmtree(root, ignore_errors=True)

    def upload_archive(self, db: Session, standard_dataset_id: int, upload, *, created_by: str | None = None) -> StandardDataset:
        row = self.get_dataset(db, standard_dataset_id)
        root = self._root_path(row)
        existing_files, _ = count_tree(root)
        if existing_files > 0:
            raise ConflictError("Standard dataset content is immutable after upload")
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=settings.temp_dir))
        try:
            extracted_root = unpack_uploaded_archive(upload, temp_dir)
            copy_tree(extracted_root, root)
            self._index_images(db, row)
            self._add_event(
                db,
                int(row.standard_dataset_id),
                "uploaded",
                message="Standard dataset archive uploaded",
                created_by=created_by,
                data={"filename": str(getattr(upload, 'filename', '') or '')},
            )
            db.commit()
            db.refresh(row)
            return row
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def materialize_from_source_tree(
        self,
        db: Session,
        *,
        name: str,
        dataset_type,
        source_root: Path,
        description: str | None = None,
        source_type: str | None = None,
        source_illegal_dataset_id: int | None = None,
        source_illegal_version_id: int | None = None,
        publish_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> StandardDataset:
        row = self.create_dataset(
            db,
            obj={
                "name": name,
                "dataset_type": dataset_type,
                "format": "yolo",
                "description": description,
                "source_type": source_type,
                "source_illegal_dataset_id": source_illegal_dataset_id,
                "source_illegal_version_id": source_illegal_version_id,
                "publish_config": publish_config,
            },
        )
        root = self._root_path(row)
        copy_tree(source_root, root)
        self._index_images(db, row)
        self._add_event(
            db,
            int(row.standard_dataset_id),
            "published",
            message="Standard dataset materialized from source tree",
            created_by=created_by,
            data={"source_type": source_type},
        )
        db.commit()
        db.refresh(row)
        return row

    def list_events(self, db: Session, standard_dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[StandardDatasetEvent]:
        self.get_dataset(db, standard_dataset_id)
        return (
            db.query(StandardDatasetEvent)
            .filter(StandardDatasetEvent.standard_dataset_id == int(standard_dataset_id))
            .order_by(StandardDatasetEvent.created_at.desc(), StandardDatasetEvent.event_id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_detail(self, db: Session, standard_dataset_id: int, *, events_limit: int = 20) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        root = self._root_path(row)
        image_count = db.query(StandardDatasetImage).filter(StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id)).count()
        return {
            "dataset": row,
            "statistics": build_statistics(root, image_count=image_count),
            "events": self.list_events(db, int(row.standard_dataset_id), skip=0, limit=events_limit),
        }

    def get_statistics(self, db: Session, standard_dataset_id: int) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        root = self._root_path(row)
        image_count = db.query(StandardDatasetImage).filter(StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id)).count()
        return build_statistics(root, image_count=image_count)

    def get_view(self, db: Session, standard_dataset_id: int, *, page: int = 1, page_size: int = 50, class_id: int | None = None) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        images = (
            db.query(StandardDatasetImage)
            .filter(StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id))
            .order_by(StandardDatasetImage.path.asc())
            .all()
        )
        payload = build_view_payload(self._root_path(row), row.storage_path, images, page=page, page_size=page_size)
        if class_id is not None:
            payload["items"] = [item for item in payload["items"] if int(class_id) in item.get("classes", [])]
            total_items = len(payload["items"])
            payload["meta"]["total_items"] = total_items
            payload["meta"]["total_pages"] = 1
        return payload

    def get_image_annotations(self, db: Session, standard_dataset_id: int, *, image_path: str) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        return build_annotations_payload(self._root_path(row), row.storage_path, image_path)

    def list_files(self, db: Session, standard_dataset_id: int, *, page: int = 1, page_size: int = 100) -> tuple[list[dict[str, Any]], int]:
        row = self.get_dataset(db, standard_dataset_id)
        return build_file_listing(self._root_path(row), row.storage_path, page=page, page_size=page_size)
