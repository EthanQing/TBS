"""Add keep/delete status to illegal dataset label mappings.

Revision ID: 0013_label_mapping_status
Revises: 0012_unified_task_system
Create Date: 2026-05-09
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0013_label_mapping_status"
down_revision = "0012_unified_task_system"
branch_labels = None
depends_on = None


TABLE_NAME = "illegal_dataset_label_mappings"
STATUS_COLUMN = "status"


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

    if STATUS_COLUMN not in columns:
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            batch_op.add_column(
                sa.Column(STATUS_COLUMN, sa.String(length=16), nullable=False, server_default="keep")
            )

    op.execute(
        sa.text(
            "UPDATE illegal_dataset_label_mappings "
            "SET status = 'delete' "
            "WHERE mapped_label = '__DISCARD__'"
        )
    )


def downgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if STATUS_COLUMN not in columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column(STATUS_COLUMN)
