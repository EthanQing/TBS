from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.deployment import Deployment
from train_platform.models.inference import InferenceRun
from train_platform.models.model_registry import ModelVersion
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun
from train_platform.repositories.dataset_repo import DatasetRepository
from train_platform.repositories.project_repo import ProjectRepository
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


def _safe_remove_dir(path: Path) -> None:
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


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

    @staticmethod
    def _normalize_framework_key(framework_key: str) -> str:
        raw = str(framework_key or "").strip().lower()
        if not raw:
            raise ValidationError("framework_key is required")
        if raw in ("pytorch", "paddle"):
            return raw
        if raw.startswith("engine:") and len(raw) > len("engine:"):
            return raw
        raise ValidationError("framework_key must be pytorch, paddle, or engine:<name>")

    @staticmethod
    def _resolve_framework_from_engine(engine: str | None) -> str:
        raw = str(engine or "").strip().lower()
        if not raw:
            return "engine:unknown"
        if raw == "ultralytics-yolo":
            return "pytorch"
        if raw == "paddle-det":
            return "paddle"
        return f"engine:{raw}"

    @staticmethod
    def _get_compare_baseline_map(tags: dict | None) -> dict[str, str]:
        data = tags if isinstance(tags, dict) else {}
        bucket = data.get("compare_baseline")
        if not isinstance(bucket, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in bucket.items():
            key = str(k or "").strip().lower()
            rid = str(v or "").strip()
            if not key or not rid:
                continue
            out[key] = rid
        return out

    def get_compare_baseline(self, db: Session, project_id: int, framework_key: str) -> dict:
        project = self.get_project(db, int(project_id))
        key = self._normalize_framework_key(framework_key)
        baseline_map = self._get_compare_baseline_map(project.tags)
        run_id = baseline_map.get(key)

        baseline_run = None
        if run_id:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
            if run and int(run.project_id) == int(project.project_id):
                engine = None
                try:
                    engine = str(getattr(run.architecture, "engine", "") or "").strip().lower() or None
                except Exception:
                    engine = None
                baseline_run = {
                    "run_id": str(run.run_id),
                    "name": run.name,
                    "status": str(getattr(run.status, "value", run.status) or ""),
                    "architecture_id": int(run.architecture_id),
                    "engine": engine,
                }

        return {
            "project_id": int(project.project_id),
            "framework_key": key,
            "baseline_run_id": run_id,
            "baseline_run": baseline_run,
        }

    def set_compare_baseline(self, db: Session, project_id: int, framework_key: str, baseline_run_id: str) -> dict:
        project = self.get_project(db, int(project_id))
        key = self._normalize_framework_key(framework_key)

        run = db.query(TrainingRun).filter(TrainingRun.run_id == str(baseline_run_id).strip()).first()
        if not run:
            raise NotFoundError("Training run not found")
        if int(run.project_id) != int(project.project_id):
            raise ConflictError("Baseline run does not belong to this project")

        arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(run.architecture_id)).first()
        if not arch:
            raise NotFoundError("Architecture not found")
        run_framework = self._resolve_framework_from_engine(getattr(arch, "engine", None))
        if run_framework != key:
            raise ConflictError("Baseline run framework does not match framework_key")

        tags = dict(project.tags) if isinstance(project.tags, dict) else {}
        baseline_map = self._get_compare_baseline_map(tags)
        baseline_map[key] = str(run.run_id)
        tags["compare_baseline"] = baseline_map
        project.tags = tags
        db.commit()
        db.refresh(project)
        return self.get_compare_baseline(db, int(project.project_id), key)

    def clear_compare_baseline(self, db: Session, project_id: int, framework_key: str) -> dict:
        project = self.get_project(db, int(project_id))
        key = self._normalize_framework_key(framework_key)

        tags = dict(project.tags) if isinstance(project.tags, dict) else {}
        baseline_map = self._get_compare_baseline_map(tags)
        if key in baseline_map:
            baseline_map.pop(key, None)
            if baseline_map:
                tags["compare_baseline"] = baseline_map
            else:
                tags.pop("compare_baseline", None)
            project.tags = tags
            db.commit()
            db.refresh(project)

        return {
            "project_id": int(project.project_id),
            "framework_key": key,
            "baseline_run_id": None,
            "baseline_run": None,
        }

    def delete_project(self, db: Session, project_id: int, *, force: bool = False) -> None:
        p = self.get_project(db, project_id)

        runs = db.query(TrainingRun).filter(TrainingRun.project_id == p.project_id).all()
        model_versions = db.query(ModelVersion).filter(ModelVersion.project_id == p.project_id).all()

        if not force and (runs or model_versions):
            parts = []
            if runs:
                parts.append(f"{len(runs)} training run(s)")
            if model_versions:
                parts.append(f"{len(model_versions)} model version(s)")
            detail = " and ".join(parts) if parts else "references exist"
            raise ConflictError(f"Cannot delete project; {detail} still reference it")

        if model_versions:
            mv_ids = [int(m.model_version_id) for m in model_versions]
            dep_ids: list[int] = []
            if mv_ids:
                deployments = db.query(Deployment).filter(Deployment.model_version_id.in_(mv_ids)).all()
                dep_ids = [int(d.deployment_id) for d in deployments]
                inf_filters = [InferenceRun.model_version_id.in_(mv_ids)]
                if dep_ids:
                    inf_filters.append(InferenceRun.deployment_id.in_(dep_ids))
                for inf in db.query(InferenceRun).filter(or_(*inf_filters)).all():
                    db.delete(inf)
                for dep in deployments:
                    db.delete(dep)
            for mv in model_versions:
                db.delete(mv)

        for run in runs:
            _safe_remove_dir(settings.training_dir / str(run.run_id))
            db.delete(run)

        db.delete(p)
        db.commit()
