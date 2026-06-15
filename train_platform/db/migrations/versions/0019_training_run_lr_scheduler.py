"""Add learning rate scheduler to training run parameters.

Revision ID: 0019_training_run_lr_scheduler
Revises: 0018_illegal_publish_jobs_idempotency
Create Date: 2026-06-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0019_training_run_lr_scheduler"
down_revision = "0018_illegal_publish_jobs_idempotency"
branch_labels = None
depends_on = None


TABLE_NAME = "training_run_parameters"
LR_SCHEDULER_COLUMN = "lr_scheduler"


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if not columns or LR_SCHEDULER_COLUMN in columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.add_column(
            sa.Column(
                LR_SCHEDULER_COLUMN,
                sa.String(length=32),
                nullable=False,
                server_default="linear",
            )
        )


def downgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if LR_SCHEDULER_COLUMN not in columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column(LR_SCHEDULER_COLUMN)
