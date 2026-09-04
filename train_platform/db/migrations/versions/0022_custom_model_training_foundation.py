"""Create custom_model_packages table and add foreign keys to architectures and runs.

Revision ID: 0022_custom_model_training_foundation
Revises: 0021_illegal_publish_job_cancel
Create Date: 2026-09-04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0022_custom_model_training_foundation"
down_revision = "0021_illegal_publish_job_cancel"
branch_labels = None
depends_on = None

CUSTOM_PACKAGES_TABLE = "custom_model_packages"
ARCHITECTURES_TABLE = "model_architectures"
TRAINING_RUNS_TABLE = "training_runs"


def _inspector():
    return inspect(op.get_bind())


def _table_names() -> set[str]:
    return {str(name) for name in _inspector().get_table_names()}


def _column_names(table_name: str) -> set[str]:
    if table_name not in _table_names():
        return set()
    return {str(col["name"]) for col in _inspector().get_columns(table_name)}


def upgrade() -> None:
    # 1. Create custom_model_packages table
    if CUSTOM_PACKAGES_TABLE not in _table_names():
        op.create_table(
            CUSTOM_PACKAGES_TABLE,
            sa.Column("package_id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("version", sa.String(length=50), nullable=False),
            sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("sdk_version", sa.String(length=50), nullable=False, server_default="1"),
            sa.Column("entrypoint_module", sa.String(length=255), nullable=False),
            sa.Column("entrypoint_class", sa.String(length=255), nullable=False),
            sa.Column("runtime_profile", sa.String(length=100), nullable=False, server_default="pytorch-default"),
            sa.Column("source_sha256", sa.String(length=64), nullable=False),
            sa.Column("archive_path", sa.String(length=500), nullable=False),
            sa.Column("manifest_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("name", "version", name="uq_custom_model_packages_name_version"),
        )
        op.create_index("ix_custom_model_packages_name", CUSTOM_PACKAGES_TABLE, ["name"], unique=False)
        op.create_index("ix_custom_model_packages_source_sha256", CUSTOM_PACKAGES_TABLE, ["source_sha256"], unique=False)
        op.create_index("ix_custom_model_packages_created_at", CUSTOM_PACKAGES_TABLE, ["created_at"], unique=False)
        op.create_index("ix_custom_model_packages_retired_at", CUSTOM_PACKAGES_TABLE, ["retired_at"], unique=False)

    # 2. Add custom_model_package_id to model_architectures
    if ARCHITECTURES_TABLE in _table_names():
        arch_cols = _column_names(ARCHITECTURES_TABLE)
        if "custom_model_package_id" not in arch_cols:
            op.add_column(
                ARCHITECTURES_TABLE,
                sa.Column("custom_model_package_id", sa.Integer(), nullable=True),
            )
            op.create_foreign_key(
                "fk_model_architectures_custom_model_package_id",
                ARCHITECTURES_TABLE,
                CUSTOM_PACKAGES_TABLE,
                ["custom_model_package_id"],
                ["package_id"],
                ondelete="RESTRICT",
            )
            op.create_index(
                "ix_model_architectures_custom_model_package_id",
                ARCHITECTURES_TABLE,
                ["custom_model_package_id"],
                unique=False,
            )

    # 3. Add custom_model_package_id and custom_model_source_sha256 to training_runs
    if TRAINING_RUNS_TABLE in _table_names():
        runs_cols = _column_names(TRAINING_RUNS_TABLE)
        if "custom_model_package_id" not in runs_cols:
            op.add_column(
                TRAINING_RUNS_TABLE,
                sa.Column("custom_model_package_id", sa.Integer(), nullable=True),
            )
            op.create_foreign_key(
                "fk_training_runs_custom_model_package_id",
                TRAINING_RUNS_TABLE,
                CUSTOM_PACKAGES_TABLE,
                ["custom_model_package_id"],
                ["package_id"],
                ondelete="RESTRICT",
            )
            op.create_index(
                "ix_training_runs_custom_model_package_id",
                TRAINING_RUNS_TABLE,
                ["custom_model_package_id"],
                unique=False,
            )

        if "custom_model_source_sha256" not in runs_cols:
            op.add_column(
                TRAINING_RUNS_TABLE,
                sa.Column("custom_model_source_sha256", sa.String(length=64), nullable=True),
            )
            op.create_index(
                "ix_training_runs_custom_model_source_sha256",
                TRAINING_RUNS_TABLE,
                ["custom_model_source_sha256"],
                unique=False,
            )


def downgrade() -> None:
    # 1. Drop from training_runs
    if TRAINING_RUNS_TABLE in _table_names():
        runs_cols = _column_names(TRAINING_RUNS_TABLE)
        if "custom_model_source_sha256" in runs_cols:
            op.drop_index("ix_training_runs_custom_model_source_sha256", table_name=TRAINING_RUNS_TABLE)
            op.drop_column(TRAINING_RUNS_TABLE, "custom_model_source_sha256")
        if "custom_model_package_id" in runs_cols:
            op.drop_constraint("fk_training_runs_custom_model_package_id", table_name=TRAINING_RUNS_TABLE, type_="foreignkey")
            op.drop_index("ix_training_runs_custom_model_package_id", table_name=TRAINING_RUNS_TABLE)
            op.drop_column(TRAINING_RUNS_TABLE, "custom_model_package_id")

    # 2. Drop from model_architectures
    if ARCHITECTURES_TABLE in _table_names():
        arch_cols = _column_names(ARCHITECTURES_TABLE)
        if "custom_model_package_id" in arch_cols:
            op.drop_constraint("fk_model_architectures_custom_model_package_id", table_name=ARCHITECTURES_TABLE, type_="foreignkey")
            op.drop_index("ix_model_architectures_custom_model_package_id", table_name=ARCHITECTURES_TABLE)
            op.drop_column(ARCHITECTURES_TABLE, "custom_model_package_id")

    # 3. Drop custom_model_packages table
    if CUSTOM_PACKAGES_TABLE in _table_names():
        for idx in (
            "ix_custom_model_packages_retired_at",
            "ix_custom_model_packages_created_at",
            "ix_custom_model_packages_source_sha256",
            "ix_custom_model_packages_name",
        ):
            op.drop_index(idx, table_name=CUSTOM_PACKAGES_TABLE)
        op.drop_table(CUSTOM_PACKAGES_TABLE)
