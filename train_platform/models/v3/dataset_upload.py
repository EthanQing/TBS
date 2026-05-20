from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BIGINT, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from train_platform.models.v3.base import V3Base


class DatasetUploadSession(V3Base):
    __tablename__ = "dataset_upload_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, server_default="upload")
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    total_size: Mapped[int] = mapped_column(BIGINT, nullable=False)
    chunk_size: Mapped[int] = mapped_column(BIGINT, nullable=False)
    total_parts: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_parts: Mapped[Optional[list[int]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, server_default="uploading")
    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class DatasetUploadTask(V3Base):
    __tablename__ = "dataset_upload_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, server_default="upload")
    source_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="zip")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, server_default="queued")
    stage: Mapped[str] = mapped_column(String(32), nullable=False, server_default="queued")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
