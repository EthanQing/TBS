from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base
from train_platform.models.enums import TaskType


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.dataset_id"), nullable=False, index=True)
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    dataset = relationship("Dataset", back_populates="projects")
    training_runs = relationship("TrainingRun", back_populates="project")
    model_versions = relationship("ModelVersion", back_populates="project")

