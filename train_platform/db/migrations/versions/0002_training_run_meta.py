"""Add training run meta table (v2).

Revision ID: 0002_training_run_meta
Revises: 0001_initial_v2
Create Date: 2026-01-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_training_run_meta"
down_revision = "0001_initial_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_run_meta",
        sa.Column("run_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("creator", sa.String(length=128), nullable=True),
        sa.Column("group_name", sa.String(length=128), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_training_run_meta_group_name", "training_run_meta", ["group_name"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_training_run_meta_group_name", table_name="training_run_meta")
    op.drop_table("training_run_meta")

