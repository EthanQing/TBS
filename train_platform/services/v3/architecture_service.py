from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TaskType
from train_platform.repositories.v3.architecture_repo import ArchitectureRepository
from train_platform.training.registry import get_plugin
from train_platform.utils.exceptions import ConflictError, ValidationError


def _normalize_and_validate_engine(value: str | None) -> str:
    engine = str(value or "").strip().lower() or "ultralytics-yolo"
    try:
        plugin = get_plugin(engine)
    except Exception as e:
        raise ValidationError(f"Unknown architecture engine: {engine}") from e
    return str(getattr(plugin, "plugin_id", engine) or engine).strip().lower()


class ArchitectureService:
    def __init__(self) -> None:
        self.repo = ArchitectureRepository()

    def list_architectures(
        self,
        db: Session,
        *,
        family: str | None = None,
        task_type: TaskType | None = None,
        q: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelArchitecture]:
        return self.repo.list(db, family=family, task_type=task_type, q=q, skip=skip, limit=limit)

    def create_architecture(self, db: Session, *, obj: dict) -> ModelArchitecture:
        family = str(obj.get("family") or "").strip()
        variant = str(obj.get("variant") or "").strip()
        if not family or not variant:
            raise ValidationError("family and variant are required")

        task_type = obj["task_type"]
        exists = self.repo.get_by_family_variant(db, family=family, variant=variant, task_type=task_type)
        if exists:
            raise ConflictError("Architecture already exists")
        engine = _normalize_and_validate_engine(obj.get("engine"))

        row = self.repo.create(
            db,
            obj_in={
                "family": family,
                "variant": variant,
                "task_type": task_type,
                "engine": engine,
                "pretrained_path": obj.get("pretrained_path"),
                "description": obj.get("description"),
                "default_params": obj.get("default_params"),
            },
        )
        db.commit()
        db.refresh(row)
        return row
