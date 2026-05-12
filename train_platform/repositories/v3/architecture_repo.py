from __future__ import annotations

from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TaskType
from train_platform.repositories.v3.base import BaseRepository


class ArchitectureRepository(BaseRepository[ModelArchitecture]):
    def __init__(self) -> None:
        super().__init__(ModelArchitecture)

    def list(
        self,
        db: Session,
        *,
        family: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        engine: Optional[str] = None,
        q: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelArchitecture]:
        query = db.query(ModelArchitecture)
        if family:
            query = query.filter(func.lower(ModelArchitecture.family) == str(family).strip().lower())
        if task_type:
            query = query.filter(ModelArchitecture.task_type == task_type)
        if engine:
            query = query.filter(func.lower(ModelArchitecture.engine) == str(engine).strip().lower())
        if q:
            like = f"%{str(q).strip()}%"
            query = query.filter(ModelArchitecture.variant.ilike(like))
        return query.order_by(ModelArchitecture.family.asc(), ModelArchitecture.variant.asc()).offset(skip).limit(limit).all()

    def get_by_family_variant(self, db: Session, *, family: str, variant: str, task_type: TaskType) -> Optional[ModelArchitecture]:
        return (
            db.query(ModelArchitecture)
            .filter(
                func.lower(ModelArchitecture.family) == str(family).strip().lower(),
                func.lower(ModelArchitecture.variant) == str(variant).strip().lower(),
                ModelArchitecture.task_type == task_type,
            )
            .first()
        )

