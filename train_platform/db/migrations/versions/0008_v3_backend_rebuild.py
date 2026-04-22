"""Rebuild backend schema for V3 dataset split.

Revision ID: 0008_v3_backend_rebuild
Revises: 0007_alarm_rules_and_alerts
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

from train_platform.models.v3 import V3Base  # noqa: F401
from train_platform.models import v3 as _models_v3  # noqa: F401


revision = "0008_v3_backend_rebuild"
down_revision = "0007_alarm_rules_and_alerts"
branch_labels = None
depends_on = None


_DROP_ORDER = [
    "inference_runs",
    "deployment_runs",
    "deployment_logs",
    "deployments",
    "model_versions",
    "training_run_meta",
    "training_run_artifacts",
    "training_run_epoch_metrics",
    "training_run_events",
    "training_run_results",
    "training_run_parameters",
    "training_runs",
    "projects",
    "dataset_images",
    "dataset_events",
    "dataset_versions",
    "datasets",
]


def _drop_if_exists(table_name: str) -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name in inspector.get_table_names():
        op.drop_table(table_name)


def _set_fk_checks(enabled: bool) -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "mysql":
        op.execute(f"SET FOREIGN_KEY_CHECKS = {1 if enabled else 0}")
    elif dialect == "sqlite":
        op.execute(f"PRAGMA foreign_keys = {'ON' if enabled else 'OFF'}")


def upgrade() -> None:
    bind = op.get_bind()
    _set_fk_checks(False)
    try:
        for table_name in _DROP_ORDER:
            _drop_if_exists(table_name)
        V3Base.metadata.create_all(bind=bind, checkfirst=True)
    finally:
        _set_fk_checks(True)


def downgrade() -> None:
    bind = op.get_bind()
    _set_fk_checks(False)
    try:
        for table_name in reversed(_DROP_ORDER):
            _drop_if_exists(table_name)
        for table_name in [
            "standard_dataset_images",
            "standard_dataset_events",
            "standard_datasets",
            "illegal_dataset_label_mappings",
            "illegal_dataset_images",
            "illegal_dataset_events",
            "illegal_dataset_versions",
            "illegal_datasets",
        ]:
            _drop_if_exists(table_name)
    finally:
        _set_fk_checks(True)
