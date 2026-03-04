from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base
from train_platform.models.enums import DeploymentPlatform, DeploymentStatus, LogLevel


class Deployment(Base):
    __tablename__ = "deployments"

    deployment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.model_version_id"), nullable=False, index=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    platform: Mapped[DeploymentPlatform] = mapped_column(
        Enum(DeploymentPlatform, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DeploymentStatus.PENDING,
        index=True,
    )

    endpoint_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    health_check_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    api_key_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    api_key_hint: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, index=True)

    model_version = relationship("ModelVersion", back_populates="deployments")
    logs = relationship("DeploymentLog", back_populates="deployment", cascade="all, delete-orphan")
    runs = relationship("DeploymentRun", back_populates="deployment", cascade="all, delete-orphan")


class DeploymentLog(Base):
    __tablename__ = "deployment_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deployment_id: Mapped[int] = mapped_column(Integer, ForeignKey("deployments.deployment_id", ondelete="CASCADE"), nullable=False, index=True)

    level: Mapped[LogLevel] = mapped_column(
        Enum(LogLevel, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=LogLevel.INFO,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    deployment = relationship("Deployment", back_populates="logs")
