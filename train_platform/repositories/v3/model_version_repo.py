from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.enums import ModelStage
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.repositories.v3.base import BaseRepository


class ModelVersionRepository(BaseRepository[ModelVersion]):
    def __init__(self) -> None:
        super().__init__(ModelVersion)

    def get_by_project_and_version(self, db: Session, project_id: int, version: str) -> Optional[ModelVersion]:
        return (
            db.query(ModelVersion)
            .filter(ModelVersion.project_id == int(project_id), ModelVersion.version == str(version))
            .first()
        )

    def list(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        run_id: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelVersion]:
        q = db.query(ModelVersion)
        if project_id is not None:
            q = q.filter(ModelVersion.project_id == int(project_id))
        if run_id:
            q = q.filter(ModelVersion.run_id == str(run_id))
        if stage:
            q = q.filter(ModelVersion.stage == stage)
        return q.order_by(ModelVersion.updated_at.desc()).offset(skip).limit(limit).all()

