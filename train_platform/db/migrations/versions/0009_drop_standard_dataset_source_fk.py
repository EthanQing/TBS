"""Drop direct illegal-dataset foreign keys from standard datasets.

Revision ID: 0009_drop_std_dataset_fk
Revises: 0008_v3_backend_rebuild
Create Date: 2026-04-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0009_drop_std_dataset_fk"
down_revision = "0008_v3_backend_rebuild"
branch_labels = None
depends_on = None


def _drop_standard_dataset_fk(batch_op, inspector) -> None:
    for fk in inspector.get_foreign_keys("standard_datasets"):
        constrained = set(fk.get("constrained_columns") or [])
        if constrained.intersection({"source_illegal_dataset_id", "source_illegal_version_id"}):
            name = fk.get("name")
            if name:
                batch_op.drop_constraint(name, type_="foreignkey")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("standard_datasets")}
    if not {"source_illegal_dataset_id", "source_illegal_version_id"}.intersection(columns):
        return

    with op.batch_alter_table("standard_datasets") as batch_op:
        _drop_standard_dataset_fk(batch_op, inspector)
        if "source_illegal_dataset_id" in columns:
            batch_op.drop_column("source_illegal_dataset_id")
        if "source_illegal_version_id" in columns:
            batch_op.drop_column("source_illegal_version_id")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("standard_datasets")}

    with op.batch_alter_table("standard_datasets") as batch_op:
        if "source_illegal_dataset_id" not in columns:
            batch_op.add_column(sa.Column("source_illegal_dataset_id", sa.Integer(), nullable=True))
        if "source_illegal_version_id" not in columns:
            batch_op.add_column(sa.Column("source_illegal_version_id", sa.Integer(), nullable=True))
        batch_op.create_index(
            "ix_standard_datasets_source_illegal_dataset_id",
            ["source_illegal_dataset_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_standard_datasets_source_illegal_version_id",
            ["source_illegal_version_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            None,
            "illegal_datasets",
            ["source_illegal_dataset_id"],
            ["illegal_dataset_id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            None,
            "illegal_dataset_versions",
            ["source_illegal_version_id"],
            ["version_id"],
            ondelete="SET NULL",
        )
