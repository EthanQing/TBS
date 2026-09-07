from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from train_platform.domains.datasets.storage.paths import resolve_storage_token
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

from .events import add_event, list_events


class StandardDatasetService:
    """Owns the StandardDataset aggregate lifecycle and metadata."""

    def dataset_root(self, dataset: StandardDataset) -> Path:
        return resolve_storage_token(dataset.storage_path)

    def _ensure_name_available(self, db: Session, name: str, *, exclude_id: int | None = None) -> None:
        row = db.query(StandardDataset).filter(StandardDataset.name == str(name).strip()).first()
        if row and (exclude_id is None or int(row.standard_dataset_id) != int(exclude_id)):
            raise ConflictError(f"Standard dataset '{name}' already exists")

    def create_dataset(self, db: Session, *, obj: dict[str, Any], commit: bool = True) -> StandardDataset:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        dataset_format = str(obj.get("format") or "yolo").strip().lower() or "yolo"
        if dataset_format != "yolo":
            raise ValidationError("Only YOLO dataset format is supported")
        self._ensure_name_available(db, name)
        row = StandardDataset(
            name=name,
            dataset_type=obj["dataset_type"],
            format=dataset_format,
            storage_path="pending/standard",
            description=obj.get("description"),
            source_type=obj.get("source_type"),
            publish_config=obj.get("publish_config"),
        )
        db.add(row)
        db.flush()
        row.storage_path = f"standard/{int(row.standard_dataset_id)}"
        self.dataset_root(row).mkdir(parents=True, exist_ok=True)
        add_event(db, int(row.standard_dataset_id), "created", message="Standard dataset created")
        if commit:
            db.commit()
            db.refresh(row)
        return row

    def get_dataset(self, db: Session, standard_dataset_id: int) -> StandardDataset:
        row = db.query(StandardDataset).filter(StandardDataset.standard_dataset_id == int(standard_dataset_id)).first()
        if not row:
            raise NotFoundError("Standard dataset not found")
        return row

    def list_datasets_page(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        format: str | None = None,
        include_statistics: bool = True,
    ) -> tuple[list[dict[str, Any]], int]:
        from .queries import dataset_with_statistics

        query = db.query(StandardDataset)
        if format:
            query = query.filter(StandardDataset.format == str(format))
        total = int(query.count())
        rows = query.order_by(StandardDataset.updated_at.desc()).offset(skip).limit(limit).all()
        items = [
            dataset_with_statistics(db, row, include_statistics=bool(include_statistics))
            for row in rows
        ]
        return items, total

    def update_dataset(self, db: Session, standard_dataset_id: int, *, patch: dict[str, Any]) -> StandardDataset:
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

    def delete_dataset(
        self,
        db: Session,
        standard_dataset_id: int,
        *,
        delete_files: bool = False,
        force: bool = False,
    ) -> None:
        from train_platform.platform.filesystem import remove_tree

        row = self.get_dataset(db, standard_dataset_id)
        projects = db.query(Project).filter(Project.standard_dataset_id == int(row.standard_dataset_id)).all()
        runs = db.query(TrainingRun).filter(TrainingRun.standard_dataset_id == int(row.standard_dataset_id)).all()
        if (projects or runs) and not force:
            raise ConflictError("Standard dataset is still referenced by projects or training runs")
        for run in runs:
            db.delete(run)
        for project in projects:
            db.delete(project)
        root = self.dataset_root(row)
        db.delete(row)
        db.commit()
        if delete_files:
            remove_tree(root, ignore_errors=True)

    def get_detail(self, db: Session, standard_dataset_id: int, *, events_limit: int = 20) -> dict[str, Any]:
        from .queries import dataset_with_statistics

        row = self.get_dataset(db, standard_dataset_id)
        return {
            "dataset": row,
            "statistics": dataset_with_statistics(db, row)["statistics"],
            "events": list_events(db, int(row.standard_dataset_id), skip=0, limit=events_limit),
        }


__all__ = ["StandardDatasetService"]
