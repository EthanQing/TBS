"""Add dataset events table for upload history (v2).

Revision ID: 0003_dataset_events
Revises: 0002_training_run_meta
Create Date: 2026-01-16
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_dataset_events"
down_revision = "0002_training_run_meta"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_events",
        sa.Column("event_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["version_id"], ["dataset_versions.version_id"]),
    )
    op.create_index("ix_dataset_events_dataset_id", "dataset_events", ["dataset_id"], unique=False)
    op.create_index("ix_dataset_events_version_id", "dataset_events", ["version_id"], unique=False)
    op.create_index("ix_dataset_events_event_type", "dataset_events", ["event_type"], unique=False)
    op.create_index("ix_dataset_events_created_at", "dataset_events", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_dataset_events_created_at", table_name="dataset_events")
    op.drop_index("ix_dataset_events_event_type", table_name="dataset_events")
    op.drop_index("ix_dataset_events_version_id", table_name="dataset_events")
    op.drop_index("ix_dataset_events_dataset_id", table_name="dataset_events")
    op.drop_table("dataset_events")

