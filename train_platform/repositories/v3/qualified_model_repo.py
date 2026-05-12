from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.qualified_model import QualifiedModel
from train_platform.repositories.v3.base import BaseRepository


class QualifiedModelRepository(BaseRepository[QualifiedModel]):
    def __init__(self) -> None:
        super().__init__(QualifiedModel)

    def get_by_model_version_id(self, db: Session, model_version_id: int) -> Optional[QualifiedModel]:
        return (
            db.query(QualifiedModel)
            .filter(QualifiedModel.model_version_id == int(model_version_id))
            .first()
        )

    def list(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        standard_dataset_id: Optional[int] = None,
        run_id: Optional[str] = None,
        model_version_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[QualifiedModel]:
        q = db.query(QualifiedModel)
        if project_id is not None:
            q = q.filter(QualifiedModel.project_id == int(project_id))
        if standard_dataset_id is not None:
            q = q.filter(QualifiedModel.standard_dataset_id == int(standard_dataset_id))
        if run_id:
            q = q.filter(QualifiedModel.run_id == str(run_id))
        if model_version_id is not None:
            q = q.filter(QualifiedModel.model_version_id == int(model_version_id))
        return q.order_by(QualifiedModel.created_at.desc(), QualifiedModel.qualified_model_id.desc()).offset(skip).limit(limit).all()
