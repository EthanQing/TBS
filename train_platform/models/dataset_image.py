from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.db.base import Base
from train_platform.models.enums import DatasetSplit


class DatasetImage(Base):
    __tablename__ = "dataset_images"

    image_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dataset_versions.version_id", ondelete="CASCADE"), nullable=False, index=True
    )

    path: Mapped[str] = mapped_column(String(500), nullable=False)
    split: Mapped[Optional[DatasetSplit]] = mapped_column(
        Enum(DatasetSplit, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    dataset = relationship("Dataset", back_populates="images")
    version = relationship("DatasetVersion", back_populates="images")

    __table_args__ = (UniqueConstraint("dataset_version_id", "path", name="uq_dataset_images_version_path"),)
