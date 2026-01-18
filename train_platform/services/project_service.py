from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun
from train_platform.repositories.dataset_repo import DatasetRepository
from train_platform.repositories.project_repo import ProjectRepository
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class ProjectService:
    def __init__(self) -> None:
        self.projects = ProjectRepository()
        self.datasets = DatasetRepository()

    def list_projects(self, db: Session, *, skip: int = 0, limit: int = 100, dataset_id: int | None = None) -> list[Project]:
        q = db.query(Project)
        if dataset_id is not None:
            q = q.filter(Project.dataset_id == int(dataset_id))
        return q.order_by(Project.updated_at.desc()).offset(skip).limit(limit).all()

    def get_project(self, db: Session, project_id: int) -> Project:
        p = self.projects.get(db, int(project_id))
        if not p:
            raise NotFoundError("Project not found")
        return p

    def create_project(self, db: Session, *, obj: dict) -> Project:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")

        exists = self.projects.get_by_name(db, name)
        if exists:
            raise ConflictError(f"Project '{name}' already exists")

        dataset_id = int(obj["dataset_id"])
        ds = self.datasets.get(db, dataset_id)
        if not ds:
            raise NotFoundError("Dataset not found")

        p = self.projects.create(
            db,
            obj_in={
                "name": name,
                "description": obj.get("description"),
                "dataset_id": dataset_id,
                "task_type": obj["task_type"],
                "created_by": obj.get("created_by"),
                "tags": obj.get("tags"),
                "is_active": True,
            },
        )
        db.commit()
        db.refresh(p)
        return p

    def update_project(self, db: Session, project_id: int, *, patch: dict) -> Project:
        p = self.get_project(db, project_id)

        if "name" in patch and patch["name"] is not None:
            new_name = str(patch["name"]).strip()
            exists = self.projects.get_by_name(db, new_name)
            if exists and int(exists.project_id) != int(p.project_id):
                raise ConflictError(f"Project '{new_name}' already exists")
            p.name = new_name

        if "description" in patch:
            p.description = patch["description"]

        if "tags" in patch:
            p.tags = patch["tags"]

        if "is_active" in patch and patch["is_active"] is not None:
            p.is_active = bool(patch["is_active"])

        db.commit()
        db.refresh(p)
        return p

    def delete_project(self, db: Session, project_id: int) -> None:
        p = self.get_project(db, project_id)

        runs_count = db.query(TrainingRun).filter(TrainingRun.project_id == p.project_id).count()
        if runs_count > 0:
            raise ConflictError(f"Cannot delete project; {runs_count} training run(s) still reference it")

        db.delete(p)
        db.commit()
