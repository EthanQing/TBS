"""Add loss weights to training run parameters.

Revision ID: 0014_training_run_loss_weights
Revises: 0013_label_mapping_status
Create Date: 2026-05-11
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0014_training_run_loss_weights"
down_revision = "0013_label_mapping_status"
branch_labels = None
depends_on = None


TABLE_NAME = "training_run_parameters"
LOSS_WEIGHTS_COLUMN = "loss_weights"


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if not columns or LOSS_WEIGHTS_COLUMN in columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.add_column(sa.Column(LOSS_WEIGHTS_COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    columns = _column_names(TABLE_NAME)
    if LOSS_WEIGHTS_COLUMN not in columns:
        return

    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column(LOSS_WEIGHTS_COLUMN)
