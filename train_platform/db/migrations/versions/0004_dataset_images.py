"""Add dataset images table for split + image IDs (v2).

Revision ID: 0004_dataset_images
Revises: 0003_dataset_events
Create Date: 2026-01-16
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_dataset_images"
down_revision = "0003_dataset_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_images",
        sa.Column("image_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("dataset_version_id", sa.Integer(), nullable=False),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("split", sa.Enum("train", "val", "test"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.version_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("dataset_version_id", "path", name="uq_dataset_images_version_path"),
    )
    op.create_index("ix_dataset_images_dataset_id", "dataset_images", ["dataset_id"], unique=False)
    op.create_index("ix_dataset_images_dataset_version_id", "dataset_images", ["dataset_version_id"], unique=False)
    op.create_index("ix_dataset_images_split", "dataset_images", ["split"], unique=False)
    op.create_index("ix_dataset_images_created_at", "dataset_images", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_dataset_images_created_at", table_name="dataset_images")
    op.drop_index("ix_dataset_images_split", table_name="dataset_images")
    op.drop_index("ix_dataset_images_dataset_version_id", table_name="dataset_images")
    op.drop_index("ix_dataset_images_dataset_id", table_name="dataset_images")
    op.drop_table("dataset_images")
