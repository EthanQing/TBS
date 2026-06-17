"""Add detailed progress fields to dataset upload tasks.

Revision ID: 0020_dataset_upload_task_progress_details
Revises: 0019_training_run_lr_scheduler
Create Date: 2026-06-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0020_dataset_upload_task_progress_details"
down_revision = "0019_training_run_lr_scheduler"
branch_labels = None
depends_on = None


TABLE_NAME = "dataset_upload_tasks"
PROCESSED_COUNT_COLUMN = "processed_count"
TOTAL_COUNT_COLUMN = "total_count"
CURRENT_ITEM_COLUMN = "current_item"
DETAIL_MESSAGE_COLUMN = "detail_message"


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if not columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        if PROCESSED_COUNT_COLUMN not in columns:
            batch_op.add_column(sa.Column(PROCESSED_COUNT_COLUMN, sa.Integer(), nullable=False, server_default="0"))
        if TOTAL_COUNT_COLUMN not in columns:
            batch_op.add_column(sa.Column(TOTAL_COUNT_COLUMN, sa.Integer(), nullable=False, server_default="0"))
        if CURRENT_ITEM_COLUMN not in columns:
            batch_op.add_column(sa.Column(CURRENT_ITEM_COLUMN, sa.String(length=1000), nullable=True))
        if DETAIL_MESSAGE_COLUMN not in columns:
            batch_op.add_column(sa.Column(DETAIL_MESSAGE_COLUMN, sa.Text(), nullable=True))


def downgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if not columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        if DETAIL_MESSAGE_COLUMN in columns:
            batch_op.drop_column(DETAIL_MESSAGE_COLUMN)
        if CURRENT_ITEM_COLUMN in columns:
            batch_op.drop_column(CURRENT_ITEM_COLUMN)
        if TOTAL_COUNT_COLUMN in columns:
            batch_op.drop_column(TOTAL_COUNT_COLUMN)
        if PROCESSED_COUNT_COLUMN in columns:
            batch_op.drop_column(PROCESSED_COUNT_COLUMN)
