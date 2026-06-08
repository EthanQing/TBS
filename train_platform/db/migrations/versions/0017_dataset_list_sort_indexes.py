"""Add dataset list sort indexes.

Revision ID: 0017_dataset_list_sort_indexes
Revises: 0016_dataset_chunk_uploads
Create Date: 2026-06-08
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect


revision = "0017_dataset_list_sort_indexes"
down_revision = "0016_dataset_chunk_uploads"
branch_labels = None
depends_on = None


INDEXES = (
    ("standard_datasets", "ix_standard_datasets_updated_at", ["updated_at"]),
    ("illegal_datasets", "ix_illegal_datasets_updated_at", ["updated_at"]),
)


def _table_names() -> set[str]:
    return {str(name) for name in inspect(op.get_bind()).get_table_names()}


def _index_names(table_name: str) -> set[str]:
    return {str(index["name"]) for index in inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    tables = _table_names()
    for table_name, index_name, columns in INDEXES:
        if table_name in tables and index_name not in _index_names(table_name):
            op.create_index(index_name, table_name, columns, unique=False)


def downgrade() -> None:
    tables = _table_names()
    for table_name, index_name, _columns in reversed(INDEXES):
        if table_name in tables and index_name in _index_names(table_name):
            op.drop_index(index_name, table_name=table_name)
