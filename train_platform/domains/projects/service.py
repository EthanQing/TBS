from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

from .baselines import preserve_baselines


class ProjectService:
    """Owns Project aggregate lifecycle operations."""

    def list_projects(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        standard_dataset_id: int | None = None,
    ) -> list[Project]:
        query = self._project_query(db, standard_dataset_id=standard_dataset_id)
        return query.order_by(Project.updated_at.desc()).offset(skip).limit(limit).all()

    def count_projects(self, db: Session, *, standard_dataset_id: int | None = None) -> int:
        return int(self._project_query(db, standard_dataset_id=standard_dataset_id).count())

    def _project_query(self, db: Session, *, standard_dataset_id: int | None = None):
        query = db.query(Project)
        if standard_dataset_id is not None:
            query = query.filter(Project.standard_dataset_id == int(standard_dataset_id))
        return query

    def get_project(self, db: Session, project_id: int) -> Project:
        row = db.query(Project).filter(Project.project_id == int(project_id)).first()
        if not row:
            raise NotFoundError("Project not found")
        return row

    @staticmethod
    def validate_delete(*, runs: list[Any], model_versions: list[Any]) -> None:
        if not runs and not model_versions:
            return
        parts = []
        if runs:
            parts.append(f"{len(runs)} training run(s)")
        if model_versions:
            parts.append(f"{len(model_versions)} model version(s)")
        raise ConflictError(f"Cannot delete project; {' and '.join(parts)} still reference it")

    def create_project(self, db: Session, *, obj: dict[str, Any]) -> Project:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        if db.query(Project).filter(Project.name == name).first():
            raise ConflictError(f"Project '{name}' already exists")

        standard_dataset_id = int(obj["standard_dataset_id"])
        dataset = db.query(StandardDataset).filter(StandardDataset.standard_dataset_id == standard_dataset_id).first()
        if not dataset:
            raise NotFoundError("Standard dataset not found")
        dataset_type = getattr(getattr(dataset, "dataset_type", None), "value", getattr(dataset, "dataset_type", None))
        task_type = getattr(obj["task_type"], "value", obj["task_type"])
        if str(dataset_type or "") != str(task_type or ""):
            raise ValidationError("Project task_type must match standard dataset dataset_type")

        row = Project(
            name=name,
            description=obj.get("description"),
            standard_dataset_id=standard_dataset_id,
            task_type=obj["task_type"],
            created_by=obj.get("created_by"),
            tags=preserve_baselines(None, obj.get("tags")),
            is_active=True,
        )
        db.add(row)
        db.flush()
        db.refresh(row)
        db.commit()
        db.refresh(row)
        return row

    def update_project(self, db: Session, project_id: int, *, patch: dict[str, Any]) -> Project:
        row = self.get_project(db, project_id)
        if "name" in patch and patch["name"] is not None:
            new_name = str(patch["name"]).strip()
            if not new_name:
                raise ValidationError("name cannot be empty")
            exists = db.query(Project).filter(Project.name == new_name).first()
            if exists and int(exists.project_id) != int(row.project_id):
                raise ConflictError(f"Project '{new_name}' already exists")
            row.name = new_name
        if "description" in patch:
            row.description = patch["description"]
        if "tags" in patch:
            row.tags = preserve_baselines(row.tags, patch["tags"])
        if "is_active" in patch and patch["is_active"] is not None:
            row.is_active = bool(patch["is_active"])
        db.commit()
        db.refresh(row)
        return row


__all__ = ["ProjectService"]
