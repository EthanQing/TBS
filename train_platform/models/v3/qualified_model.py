from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base as Base


class QualifiedModel(Base):
    __tablename__ = "qualified_models"

    qualified_model_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model_versions.model_version_id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    standard_dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("standard_datasets.standard_dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    qualified_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    weights_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    model_version = relationship("ModelVersion", back_populates="qualified_model")
    project = relationship("Project", back_populates="qualified_models")
    training_run = relationship("TrainingRun", back_populates="qualified_models")
    standard_dataset = relationship("StandardDataset", back_populates="qualified_models")

    __table_args__ = (UniqueConstraint("model_version_id", name="uq_qualified_models_model_version"),)
