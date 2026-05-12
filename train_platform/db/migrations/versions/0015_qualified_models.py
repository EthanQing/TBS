"""Create qualified models table.

Revision ID: 0015_qualified_models
Revises: 0014_training_run_loss_weights
Create Date: 2026-05-12
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0015_qualified_models"
down_revision = "0014_training_run_loss_weights"
branch_labels = None
depends_on = None


TABLE_NAME = "qualified_models"


def _inspector():
    return inspect(op.get_bind())


def _table_names() -> set[str]:
    return {str(name) for name in _inspector().get_table_names()}


def _index_names(table_name: str) -> set[str]:
    if table_name not in _table_names():
        return set()
    return {str(index["name"]) for index in _inspector().get_indexes(table_name)}


def upgrade() -> None:
    if TABLE_NAME in _table_names():
        return

    op.create_table(
        TABLE_NAME,
        sa.Column("qualified_model_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("standard_dataset_id", sa.Integer(), nullable=False),
        sa.Column("qualified_by", sa.String(length=128), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("weights_path", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.model_version_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["standard_dataset_id"], ["standard_datasets.standard_dataset_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("model_version_id", name="uq_qualified_models_model_version"),
    )
    op.create_index("ix_qualified_models_project_id", TABLE_NAME, ["project_id"], unique=False)
    op.create_index("ix_qualified_models_run_id", TABLE_NAME, ["run_id"], unique=False)
    op.create_index("ix_qualified_models_standard_dataset_id", TABLE_NAME, ["standard_dataset_id"], unique=False)


def downgrade() -> None:
    if TABLE_NAME not in _table_names():
        return

    indexes = _index_names(TABLE_NAME)
    for index_name in (
        "ix_qualified_models_standard_dataset_id",
        "ix_qualified_models_run_id",
        "ix_qualified_models_project_id",
    ):
        if index_name in indexes:
            op.drop_index(index_name, table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)
