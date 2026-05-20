"""Add dataset chunk upload sessions and tasks.

Revision ID: 0016_dataset_chunk_uploads
Revises: 0015_qualified_models
Create Date: 2026-05-20
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0016_dataset_chunk_uploads"
down_revision = "0015_qualified_models"
branch_labels = None
depends_on = None


UPLOAD_SESSIONS = "dataset_upload_sessions"
UPLOAD_TASKS = "dataset_upload_tasks"


def _table_names() -> set[str]:
    return {str(name) for name in inspect(op.get_bind()).get_table_names()}


def upgrade() -> None:
    tables = _table_names()
    if UPLOAD_SESSIONS not in tables:
        op.create_table(
            UPLOAD_SESSIONS,
            sa.Column("session_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("dataset_kind", sa.String(length=16), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            sa.Column("mode", sa.String(length=16), nullable=False, server_default="upload"),
            sa.Column("filename", sa.String(length=500), nullable=False),
            sa.Column("total_size", sa.BIGINT(), nullable=False),
            sa.Column("chunk_size", sa.BIGINT(), nullable=False),
            sa.Column("total_parts", sa.Integer(), nullable=False),
            sa.Column("uploaded_parts", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="uploading"),
            sa.Column("created_by", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
        )
        op.create_index("ix_dataset_upload_sessions_dataset_kind", UPLOAD_SESSIONS, ["dataset_kind"], unique=False)
        op.create_index("ix_dataset_upload_sessions_dataset_id", UPLOAD_SESSIONS, ["dataset_id"], unique=False)
        op.create_index("ix_dataset_upload_sessions_status", UPLOAD_SESSIONS, ["status"], unique=False)
        op.create_index("ix_dataset_upload_sessions_expires_at", UPLOAD_SESSIONS, ["expires_at"], unique=False)

    if UPLOAD_TASKS not in tables:
        op.create_table(
            UPLOAD_TASKS,
            sa.Column("task_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("dataset_kind", sa.String(length=16), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            sa.Column("session_id", sa.String(length=36), nullable=True),
            sa.Column("mode", sa.String(length=16), nullable=False, server_default="upload"),
            sa.Column("source_path", sa.String(length=1000), nullable=False),
            sa.Column("source_type", sa.String(length=16), nullable=False, server_default="zip"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
            sa.Column("stage", sa.String(length=32), nullable=False, server_default="queued"),
            sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_by", sa.String(length=128), nullable=True),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_dataset_upload_tasks_dataset_kind", UPLOAD_TASKS, ["dataset_kind"], unique=False)
        op.create_index("ix_dataset_upload_tasks_dataset_id", UPLOAD_TASKS, ["dataset_id"], unique=False)
        op.create_index("ix_dataset_upload_tasks_session_id", UPLOAD_TASKS, ["session_id"], unique=False)
        op.create_index("ix_dataset_upload_tasks_status", UPLOAD_TASKS, ["status"], unique=False)


def downgrade() -> None:
    tables = _table_names()
    if UPLOAD_TASKS in tables:
        op.drop_table(UPLOAD_TASKS)
    if UPLOAD_SESSIONS in tables:
        op.drop_table(UPLOAD_SESSIONS)
