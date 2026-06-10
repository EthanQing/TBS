"""Add idempotent illegal dataset publish jobs.

Revision ID: 0018_illegal_publish_jobs_idempotency
Revises: 0017_dataset_list_sort_indexes
Create Date: 2026-06-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0018_illegal_publish_jobs_idempotency"
down_revision = "0017_dataset_list_sort_indexes"
branch_labels = None
depends_on = None


TABLE_NAME = "illegal_dataset_publish_jobs"
STANDARD_DATASET_ID_START = 2000000


def _table_names() -> set[str]:
    return {str(name) for name in inspect(op.get_bind()).get_table_names()}


def _index_names(table_name: str) -> set[str]:
    if table_name not in _table_names():
        return set()
    inspector = inspect(op.get_bind())
    names = {str(index["name"]) for index in inspector.get_indexes(table_name)}
    names.update(str(item["name"]) for item in inspector.get_unique_constraints(table_name) if item.get("name"))
    return names


def _max_standard_dataset_id() -> int:
    bind = op.get_bind()
    value = bind.execute(sa.text("SELECT COALESCE(MAX(standard_dataset_id), 0) FROM standard_datasets")).scalar()
    return int(value or 0)


def _bump_standard_dataset_auto_increment() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "mysql" or "standard_datasets" not in _table_names():
        return
    next_id = max(STANDARD_DATASET_ID_START, _max_standard_dataset_id() + 1)
    op.execute(sa.text(f"ALTER TABLE standard_datasets AUTO_INCREMENT = {int(next_id)}"))


def upgrade() -> None:
    tables = _table_names()
    if TABLE_NAME not in tables:
        op.create_table(
            TABLE_NAME,
            sa.Column("job_id", sa.String(length=36), nullable=False),
            sa.Column("illegal_dataset_id", sa.Integer(), nullable=False),
            sa.Column("source_illegal_version_id", sa.Integer(), nullable=False),
            sa.Column("idempotency_key", sa.String(length=64), nullable=False),
            sa.Column("standard_dataset_id", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(length=32), server_default="queued", nullable=False),
            sa.Column("phase", sa.String(length=32), server_default="queued", nullable=False),
            sa.Column("progress", sa.Integer(), server_default="0", nullable=False),
            sa.Column("processed", sa.Integer(), server_default="0", nullable=False),
            sa.Column("total", sa.Integer(), server_default="0", nullable=False),
            sa.Column("seq", sa.Integer(), server_default="1", nullable=False),
            sa.Column("request_payload", sa.JSON(), nullable=True),
            sa.Column("request_summary", sa.JSON(), nullable=True),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("logs", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["illegal_dataset_id"], ["illegal_datasets.illegal_dataset_id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["source_illegal_version_id"], ["illegal_dataset_versions.version_id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["standard_dataset_id"], ["standard_datasets.standard_dataset_id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("job_id"),
        )

    indexes = _index_names(TABLE_NAME)
    index_specs = (
        ("ix_illegal_dataset_publish_jobs_illegal_dataset_id", ["illegal_dataset_id"], False),
        ("ix_illegal_dataset_publish_jobs_source_illegal_version_id", ["source_illegal_version_id"], False),
        ("ix_illegal_dataset_publish_jobs_standard_dataset_id", ["standard_dataset_id"], False),
        ("ix_illegal_dataset_publish_jobs_status", ["status"], False),
        ("ix_illegal_publish_jobs_dataset_status", ["illegal_dataset_id", "status"], False),
        ("uq_illegal_dataset_publish_jobs_idempotency_key", ["idempotency_key"], True),
    )
    for index_name, columns, unique in index_specs:
        if index_name not in indexes:
            op.create_index(index_name, TABLE_NAME, columns, unique=unique)

    _bump_standard_dataset_auto_increment()


def downgrade() -> None:
    if TABLE_NAME in _table_names():
        op.drop_table(TABLE_NAME)
