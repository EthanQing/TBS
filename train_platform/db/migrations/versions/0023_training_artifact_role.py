"""Add semantic role to training run artifacts.

Revision ID: 0023_training_artifact_role
Revises: 0022_custom_model_training_foundation
Create Date: 2026-09-07
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0023_training_artifact_role"
down_revision = "0022_custom_model_training_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    columns = {str(column["name"]) for column in inspector.get_columns("training_run_artifacts")}
    if "role" not in columns:
        op.add_column(
            "training_run_artifacts",
            sa.Column("role", sa.String(length=64), nullable=True),
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    columns = {str(column["name"]) for column in inspector.get_columns("training_run_artifacts")}
    if "role" in columns:
        op.drop_column("training_run_artifacts", "role")
