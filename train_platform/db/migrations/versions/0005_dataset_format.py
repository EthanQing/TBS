"""Add datasets.format column (yolo/coco, etc.).

Revision ID: 0005_dataset_format
Revises: 0004_dataset_images
Create Date: 2026-01-28
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_dataset_format"
down_revision = "0004_dataset_images"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add a non-null format column with default for existing rows.
    op.add_column(
        "datasets",
        sa.Column("format", sa.String(length=50), server_default=sa.text("'yolo'"), nullable=False),
    )
    op.create_index("ix_datasets_format", "datasets", ["format"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_datasets_format", table_name="datasets")
    op.drop_column("datasets", "format")

