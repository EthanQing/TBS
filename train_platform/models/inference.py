from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    inference_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.model_version_id"), nullable=False, index=True)
    deployment_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("deployments.deployment_id"), nullable=True, index=True)

    input_path: Mapped[str] = mapped_column(String(500), nullable=False)
    input_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    model_version = relationship("ModelVersion")
    deployment = relationship("Deployment")

