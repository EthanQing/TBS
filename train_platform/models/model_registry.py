from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base
from train_platform.models.enums import ModelStage


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)

    version: Mapped[str] = mapped_column(String(50), nullable=False)
    stage: Mapped[ModelStage] = mapped_column(
        Enum(ModelStage, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ModelStage.DEVELOPMENT,
        index=True,
    )

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relative to BASE_TRAINING_DIR (portable).
    weights_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    project = relationship("Project", back_populates="model_versions")
    training_run = relationship("TrainingRun", back_populates="model_versions")
    deployments = relationship("Deployment", back_populates="model_version")

    __table_args__ = (UniqueConstraint("project_id", "version", name="uq_model_versions_project_version"),)

