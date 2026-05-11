"""Placeholder for previously applied unified task system migration.

Revision ID: 0012_unified_task_system
Revises: 0011_dataset_persistent_stats
Create Date: 2026-05-08

This project database may already be stamped with this historical revision.
The original source migration is not present in this checkout, so this file
keeps the Alembic graph resolvable. Current metadata/migrations are idempotent
and newer schema changes are applied by later revisions.
"""

from __future__ import annotations


revision = "0012_unified_task_system"
down_revision = "0011_dataset_persistent_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    return


def downgrade() -> None:
    return
