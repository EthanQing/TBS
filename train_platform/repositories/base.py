from __future__ import annotations

from typing import Any, Generic, Optional, Type, TypeVar

from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from train_platform.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def get(self, db: Session, id_value: Any) -> Optional[ModelType]:
        pk = inspect(self.model).primary_key[0].name
        return db.query(self.model).filter(getattr(self.model, pk) == id_value).first()

    def get_multi(self, db: Session, *, skip: int = 0, limit: int = 100) -> list[ModelType]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: dict) -> ModelType:
        """
        Create without committing.

        Transaction boundaries are owned by the service layer (API request scope).
        """
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id_value: Any) -> Optional[ModelType]:
        obj = self.get(db, id_value)
        if not obj:
            return None
        db.delete(obj)
        db.flush()
        return obj
