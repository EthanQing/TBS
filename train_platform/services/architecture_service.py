from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.models.architecture import ModelArchitecture
from train_platform.models.enums import TaskType
from train_platform.repositories.architecture_repo import ArchitectureRepository
from train_platform.utils.exceptions import ConflictError, ValidationError


HIDDEN_ARCH_ENGINES = {"paddle-det"}


def _normalize_engine(value: str | None) -> str:
    return str(value or "").strip().lower()


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
        rows = self.repo.list(db, family=family, task_type=task_type, q=q, skip=skip, limit=limit)
        return [row for row in rows if _normalize_engine(getattr(row, "engine", None)) not in HIDDEN_ARCH_ENGINES]

    def create_architecture(self, db: Session, *, obj: dict) -> ModelArchitecture:
        family = str(obj.get("family") or "").strip()
        variant = str(obj.get("variant") or "").strip()
        if not family or not variant:
            raise ValidationError("family and variant are required")

        task_type = obj["task_type"]
        exists = self.repo.get_by_family_variant(db, family=family, variant=variant, task_type=task_type)
        if exists:
            raise ConflictError("Architecture already exists")

        row = self.repo.create(
            db,
            obj_in={
                "family": family,
                "variant": variant,
                "task_type": task_type,
                "engine": obj.get("engine") or "ultralytics-yolo",
                "pretrained_path": obj.get("pretrained_path"),
                "description": obj.get("description"),
                "default_params": obj.get("default_params"),
            },
        )
        db.commit()
        db.refresh(row)
        return row
