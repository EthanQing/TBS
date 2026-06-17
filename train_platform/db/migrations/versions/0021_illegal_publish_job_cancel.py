"""Add cancellation flag to illegal dataset publish jobs.

Revision ID: 0021_illegal_publish_job_cancel
Revises: 0020_dataset_upload_task_progress_details
Create Date: 2026-06-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0021_illegal_publish_job_cancel"
down_revision = "0020_dataset_upload_task_progress_details"
branch_labels = None
depends_on = None


TABLE_NAME = "illegal_dataset_publish_jobs"
COLUMN_NAME = "cancel_requested"


def _table_names() -> set[str]:
    return {str(name) for name in inspect(op.get_bind()).get_table_names()}


def _column_names(table_name: str) -> set[str]:
    if table_name not in _table_names():
        return set()
    return {str(column["name"]) for column in inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    if TABLE_NAME not in _table_names():
        return
    if COLUMN_NAME not in _column_names(TABLE_NAME):
        op.add_column(
            TABLE_NAME,
            sa.Column(COLUMN_NAME, sa.Boolean(), server_default=sa.text("0"), nullable=False),
        )


def downgrade() -> None:
    if TABLE_NAME in _table_names() and COLUMN_NAME in _column_names(TABLE_NAME):
        op.drop_column(TABLE_NAME, COLUMN_NAME)
