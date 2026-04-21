from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.project import Project
from train_platform.repositories.v3.base import BaseRepository


class ProjectRepository(BaseRepository[Project]):
    def __init__(self) -> None:
        super().__init__(Project)

    def get_by_name(self, db: Session, name: str) -> Optional[Project]:
        return db.query(Project).filter(Project.name == str(name)).first()
