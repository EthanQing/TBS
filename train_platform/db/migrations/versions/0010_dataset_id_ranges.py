"""Reserve distinct numeric ID ranges for illegal vs standard datasets.

Revision ID: 0010_dataset_id_ranges
Revises: 0009_drop_std_dataset_fk
Create Date: 2026-04-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0010_dataset_id_ranges"
down_revision = "0009_drop_std_dataset_fk"
branch_labels = None
depends_on = None


ILLEGAL_DATASET_ID_START = 1000000
STANDARD_DATASET_ID_START = 2000000


def _max_id(bind, table_name: str, pk_name: str) -> int:
    value = bind.execute(sa.text(f"SELECT COALESCE(MAX({pk_name}), 0) FROM {table_name}")).scalar()
    return int(value or 0)


def _bump_mysql_auto_increment(bind, table_name: str, next_id: int) -> None:
    op.execute(sa.text(f"ALTER TABLE {table_name} AUTO_INCREMENT = {int(next_id)}"))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if bind.dialect.name != "mysql":
        return

    tables = set(inspector.get_table_names())
    if "illegal_datasets" in tables:
        next_id = max(ILLEGAL_DATASET_ID_START, _max_id(bind, "illegal_datasets", "illegal_dataset_id") + 1)
        _bump_mysql_auto_increment(bind, "illegal_datasets", next_id)
    if "standard_datasets" in tables:
        next_id = max(STANDARD_DATASET_ID_START, _max_id(bind, "standard_datasets", "standard_dataset_id") + 1)
        _bump_mysql_auto_increment(bind, "standard_datasets", next_id)


def downgrade() -> None:
    # No-op: lowering AUTO_INCREMENT is unnecessary and may be unsafe when rows already exist.
    return
