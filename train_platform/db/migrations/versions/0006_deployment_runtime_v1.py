"""Deployment runtime v1 (deployment_runs + api key columns).

Revision ID: 0006_deployment_runtime_v1
Revises: 0005_dataset_format
Create Date: 2026-03-04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_deployment_runtime_v1"
down_revision = "0005_dataset_format"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("deployments", sa.Column("api_key_hash", sa.String(length=128), nullable=True))
    op.add_column("deployments", sa.Column("api_key_hint", sa.String(length=32), nullable=True))

    op.create_table(
        "deployment_runs",
        sa.Column("run_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("deployment_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("trigger_type", sa.Enum("manual"), nullable=False),
        sa.Column("status", sa.Enum("queued", "running", "completed", "failed", "cancelled"), nullable=False),
        sa.Column(
            "phase",
            sa.Enum("preparing", "validate_artifacts", "materialize_runtime", "smoke_test", "activate", "done", "cancelled"),
            nullable=False,
        ),
        sa.Column("current_step", sa.String(length=64), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["deployment_id"], ["deployments.deployment_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.model_version_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_deployment_runs_deployment_id", "deployment_runs", ["deployment_id"], unique=False)
    op.create_index("ix_deployment_runs_project_id", "deployment_runs", ["project_id"], unique=False)
    op.create_index("ix_deployment_runs_model_version_id", "deployment_runs", ["model_version_id"], unique=False)
    op.create_index("ix_deployment_runs_status", "deployment_runs", ["status"], unique=False)
    op.create_index("ix_deployment_runs_updated_at", "deployment_runs", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_table("deployment_runs")
    op.drop_column("deployments", "api_key_hint")
    op.drop_column("deployments", "api_key_hash")

