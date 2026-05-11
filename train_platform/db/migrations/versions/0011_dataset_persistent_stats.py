"""Placeholder for previously applied dataset persistent stats migration.

Revision ID: 0011_dataset_persistent_stats
Revises: 0010_dataset_id_ranges
Create Date: 2026-05-08

This project database may already be stamped with this historical revision.
The original source migration is not present in this checkout, so this file
keeps the Alembic graph resolvable. Current metadata/migrations are idempotent
and newer schema changes are applied by later revisions.
"""

from __future__ import annotations


revision = "0011_dataset_persistent_stats"
down_revision = "0010_dataset_id_ranges"
branch_labels = None
depends_on = None


def upgrade() -> None:
    return


def downgrade() -> None:
    return
