from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base as Base
from train_platform.models.v3.enums import TaskType


class ModelArchitecture(Base):
    __tablename__ = "model_architectures"

    architecture_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    family: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # e.g. YOLOv8/PaddleDetection/DETR
    variant: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # e.g. yolov8n / ppyoloe_s
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True,
    )

    # Trainer implementation key (used by plugin registry).
    engine: Mapped[str] = mapped_column(String(64), nullable=False, default="ultralytics-yolo")

    pretrained_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    default_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    training_runs = relationship("TrainingRun", back_populates="architecture")

    __table_args__ = (UniqueConstraint("family", "variant", "task_type", name="uq_model_architectures_family_variant_task"),)

