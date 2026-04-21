from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base as Base
from train_platform.models.v3.enums import DeploymentRunPhase, DeploymentRunStatus, DeploymentTriggerType


class DeploymentRun(Base):
    __tablename__ = "deployment_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    deployment_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("deployments.deployment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model_versions.model_version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    trigger_type: Mapped[DeploymentTriggerType] = mapped_column(
        Enum(DeploymentTriggerType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DeploymentTriggerType.MANUAL,
    )
    status: Mapped[DeploymentRunStatus] = mapped_column(
        Enum(DeploymentRunStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DeploymentRunStatus.QUEUED,
        index=True,
    )
    phase: Mapped[DeploymentRunPhase] = mapped_column(
        Enum(DeploymentRunPhase, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DeploymentRunPhase.PREPARING,
    )
    current_step: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
    )

    deployment = relationship("Deployment", back_populates="runs")
    model_version = relationship("ModelVersion")

