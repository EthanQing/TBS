from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.datasets.illegal import versions
from train_platform.domains.datasets.illegal.events import add_event
from train_platform.models.v3.illegal_dataset import (
    IllegalDataset,
    IllegalDatasetEvent,
)
from train_platform.platform.filesystem import remove_tree
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class IllegalDatasetService:
    """CRUD and aggregate-level views for Illegal Dataset."""

    def _next_dataset_id(self, db: Session) -> int:
        current_max = db.query(func.max(IllegalDataset.illegal_dataset_id)).scalar()
        start = int(settings.illegal_dataset_id_start)
        return start if current_max is None else max(start, int(current_max) + 1)

    def _ensure_name_available(self, db: Session, name: str, *, exclude_id: int | None = None) -> None:
        row = db.query(IllegalDataset).filter(IllegalDataset.name == str(name).strip()).first()
        if row and (exclude_id is None or int(row.illegal_dataset_id) != int(exclude_id)):
            raise ConflictError(f"Illegal dataset '{name}' already exists")

    def _dataset_with_statistics(
        self,
        db: Session,
        dataset: IllegalDataset,
        *,
        include_statistics: bool = True,
    ) -> dict[str, Any]:
        if include_statistics:
            try:
                statistics = versions.build_statistics(db, dataset)
            except (NotFoundError, ValidationError):
                statistics = versions.empty_statistics()
        else:
            statistics = None
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
            "preview_image_url": None,
        }

    def list_datasets_page(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        format: str | None = None,
        include_statistics: bool = True,
    ) -> tuple[list[dict[str, Any]], int]:
        query = db.query(IllegalDataset)
        if format:
            query = query.filter(IllegalDataset.format == str(format))
        total = int(query.count())
        rows = query.order_by(IllegalDataset.updated_at.desc()).offset(skip).limit(limit).all()
        items = [
            self._dataset_with_statistics(db, row, include_statistics=bool(include_statistics))
            for row in rows
        ]
        return items, total

    def create_dataset(self, db: Session, *, obj: dict[str, Any]) -> IllegalDataset:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        dataset_format = str(obj.get("format") or "yolo").strip().lower() or "yolo"
        if dataset_format != "yolo":
            raise ValidationError("Only YOLO dataset format is supported")
        self._ensure_name_available(db, name)
        dataset = IllegalDataset(
            illegal_dataset_id=self._next_dataset_id(db),
            name=name,
            dataset_type=obj["dataset_type"],
            format=dataset_format,
            storage_path="pending/illegal",
            description=obj.get("description"),
        )
        db.add(dataset)
        db.flush()
        dataset.storage_path = f"illegal/{int(dataset.illegal_dataset_id)}"
        versions.dataset_root(dataset).mkdir(parents=True, exist_ok=True)
        add_event(db, int(dataset.illegal_dataset_id), "created", message="Illegal dataset created")
        db.commit()
        db.refresh(dataset)
        return dataset

    def get_dataset(self, db: Session, illegal_dataset_id: int) -> IllegalDataset:
        dataset = db.query(IllegalDataset).filter(IllegalDataset.illegal_dataset_id == int(illegal_dataset_id)).first()
        if not dataset:
            raise NotFoundError("Illegal dataset not found")
        return dataset

    def update_dataset(self, db: Session, illegal_dataset_id: int, *, patch: dict[str, Any]) -> IllegalDataset:
        dataset = self.get_dataset(db, illegal_dataset_id)
        if "name" in patch and patch["name"] is not None:
            name = str(patch["name"]).strip()
            if not name:
                raise ValidationError("name cannot be empty")
            self._ensure_name_available(db, name, exclude_id=int(dataset.illegal_dataset_id))
            dataset.name = name
        if "description" in patch:
            dataset.description = patch["description"]
        db.commit()
        db.refresh(dataset)
        return dataset

    def delete_dataset(
        self,
        db: Session,
        illegal_dataset_id: int,
        *,
        delete_files: bool = False,
        force: bool = False,
    ) -> None:
        dataset = self.get_dataset(db, illegal_dataset_id)
        root = versions.dataset_root(dataset)
        version_root = settings.datasets_dir / "illegal" / ".versions" / str(int(dataset.illegal_dataset_id))
        db.delete(dataset)
        db.commit()
        if delete_files:
            remove_tree(root, ignore_errors=True)
            remove_tree(version_root, ignore_errors=True)

    def list_versions(self, db: Session, illegal_dataset_id: int, *, skip: int = 0, limit: int = 100):
        self.get_dataset(db, illegal_dataset_id)
        return versions.list_versions(db, int(illegal_dataset_id), skip=skip, limit=limit)

    @staticmethod
    def _events_query(db: Session, illegal_dataset_id: int):
        return (
            db.query(IllegalDatasetEvent)
            .filter(IllegalDatasetEvent.illegal_dataset_id == int(illegal_dataset_id))
            .order_by(IllegalDatasetEvent.created_at.desc(), IllegalDatasetEvent.event_id.desc())
        )

    def list_events(self, db: Session, illegal_dataset_id: int, *, skip: int = 0, limit: int = 100):
        self.get_dataset(db, illegal_dataset_id)
        return (
            self._events_query(db, illegal_dataset_id)
            .offset(max(0, int(skip)))
            .limit(max(0, int(limit)))
            .all()
        )

    def list_events_page(
        self,
        db: Session,
        illegal_dataset_id: int,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[IllegalDatasetEvent], int]:
        self.get_dataset(db, illegal_dataset_id)
        query = self._events_query(db, illegal_dataset_id)
        total = int(query.count())
        items = (
            query
            .offset(max(0, int(skip)))
            .limit(max(0, int(limit)))
            .all()
        )
        return items, total

    def get_detail(
        self,
        db: Session,
        illegal_dataset_id: int,
        *,
        versions_limit: int = 20,
        events_limit: int = 20,
    ) -> dict[str, Any]:
        dataset = self.get_dataset(db, illegal_dataset_id)
        active = versions.active_version(db, dataset)
        return {
            "dataset": dataset,
            "statistics": versions.build_statistics(db, dataset, version=active),
            "active_version": active,
            "versions": self.list_versions(db, int(dataset.illegal_dataset_id), limit=versions_limit),
            "events": self.list_events(db, int(dataset.illegal_dataset_id), limit=events_limit),
        }

    def get_statistics(self, db: Session, illegal_dataset_id: int, *, version_id: int | None = None):
        dataset = self.get_dataset(db, illegal_dataset_id)
        selected = versions.selected_version(db, dataset, version_id=version_id)
        return versions.build_statistics(db, dataset, version=selected)

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
        self.get_dataset(db, illegal_dataset_id)
        return {
            "categories": [],
            "items": [],
            "meta": {
                "page": int(page),
                "page_size": int(page_size),
                "total_items": 0,
                "total_pages": 1,
            },
        }
