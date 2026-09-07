from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base as Base


class CustomModelPackage(Base):
    __tablename__ = "custom_model_packages"

    package_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)

    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    sdk_version: Mapped[str] = mapped_column(String(50), nullable=False, default="1")

    entrypoint_module: Mapped[str] = mapped_column(String(255), nullable=False)
    entrypoint_class: Mapped[str] = mapped_column(String(255), nullable=False)

    runtime_profile: Mapped[str] = mapped_column(String(100), nullable=False, default="pytorch-default")

    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    archive_path: Mapped[str] = mapped_column(String(500), nullable=False)

    manifest_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    retired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    architectures = relationship("ModelArchitecture", back_populates="custom_model_package")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_custom_model_packages_name_version"),
    )


__all__ = ["CustomModelPackage"]
