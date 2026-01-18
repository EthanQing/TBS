"""Initial schema (v2, redesigned).

Revision ID: 0001_initial_v2
Revises:
Create Date: 2026-01-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_v2"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --------------------
    # datasets + versions
    # --------------------
    op.create_table(
        "datasets",
        sa.Column("dataset_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("dataset_type", sa.Enum("detection", "segmentation", "classification"), nullable=False),
        sa.Column("storage_path", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("active_version_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("name", name="uq_datasets_name"),
    )
    op.create_index("ix_datasets_name", "datasets", ["name"], unique=True)

    op.create_table(
        "dataset_versions",
        sa.Column("version_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("parent_version_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.Enum("created", "finalized", "failed"), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("manifest_path", sa.String(length=500), nullable=True),
        sa.Column("snapshot_path", sa.String(length=500), nullable=True),
        sa.Column("file_count", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.dataset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parent_version_id"], ["dataset_versions.version_id"]),
        sa.UniqueConstraint("dataset_id", "version", name="uq_dataset_versions_dataset_version"),
    )
    op.create_index("ix_dataset_versions_dataset_id", "dataset_versions", ["dataset_id"], unique=False)
    op.create_index("ix_dataset_versions_created_at", "dataset_versions", ["created_at"], unique=False)
    op.create_index("ix_dataset_versions_status", "dataset_versions", ["status"], unique=False)

    # Add FK to datasets.active_version_id after dataset_versions exists.
    op.create_foreign_key(
        "fk_datasets_active_version_id",
        "datasets",
        "dataset_versions",
        ["active_version_id"],
        ["version_id"],
    )
    op.create_index("ix_datasets_active_version_id", "datasets", ["active_version_id"], unique=False)

    # --------------------
    # projects
    # --------------------
    op.create_table(
        "projects",
        sa.Column("project_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("task_type", sa.Enum("detection", "segmentation", "classification"), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.dataset_id"]),
        sa.UniqueConstraint("name", name="uq_projects_name"),
    )
    op.create_index("ix_projects_name", "projects", ["name"], unique=True)
    op.create_index("ix_projects_dataset_id", "projects", ["dataset_id"], unique=False)
    op.create_index("ix_projects_is_active", "projects", ["is_active"], unique=False)

    # --------------------
    # model architectures
    # --------------------
    op.create_table(
        "model_architectures",
        sa.Column("architecture_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("family", sa.String(length=50), nullable=False),
        sa.Column("variant", sa.String(length=100), nullable=False),
        sa.Column("task_type", sa.Enum("detection", "segmentation", "classification"), nullable=False),
        sa.Column("engine", sa.String(length=64), nullable=False),
        sa.Column("pretrained_path", sa.String(length=500), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("default_params", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("family", "variant", "task_type", name="uq_model_architectures_family_variant_task"),
    )
    op.create_index("ix_model_architectures_family", "model_architectures", ["family"], unique=False)
    op.create_index("ix_model_architectures_variant", "model_architectures", ["variant"], unique=False)
    op.create_index("ix_model_architectures_task_type", "model_architectures", ["task_type"], unique=False)

    # --------------------
    # training runs (+ params/results/metrics/events/artifacts)
    # --------------------
    op.create_table(
        "training_runs",
        sa.Column("run_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("dataset_version_id", sa.Integer(), nullable=False),
        sa.Column("architecture_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "status",
            sa.Enum("created", "queued", "running", "completed", "failed", "cancelled", "deleted"),
            nullable=False,
        ),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("current_epoch", sa.Integer(), nullable=False),
        sa.Column("total_epochs", sa.Integer(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_reason", sa.Text(), nullable=True),
        sa.Column("delete_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hidden", sa.Boolean(), nullable=False),
        sa.Column("run_dir", sa.String(length=500), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["architecture_id"], ["model_architectures.architecture_id"]),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.version_id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"]),
    )
    op.create_index("ix_training_runs_project_id", "training_runs", ["project_id"], unique=False)
    op.create_index("ix_training_runs_dataset_version_id", "training_runs", ["dataset_version_id"], unique=False)
    op.create_index("ix_training_runs_architecture_id", "training_runs", ["architecture_id"], unique=False)
    op.create_index("ix_training_runs_status", "training_runs", ["status"], unique=False)
    op.create_index("ix_training_runs_created_at", "training_runs", ["created_at"], unique=False)
    op.create_index("ix_training_runs_queued_at", "training_runs", ["queued_at"], unique=False)
    op.create_index("ix_training_runs_worker_id", "training_runs", ["worker_id"], unique=False)
    op.create_index("ix_training_runs_heartbeat_at", "training_runs", ["heartbeat_at"], unique=False)
    op.create_index("ix_training_runs_hidden", "training_runs", ["hidden"], unique=False)
    op.create_index("ix_training_runs_cancel_requested_at", "training_runs", ["cancel_requested_at"], unique=False)
    op.create_index("ix_training_runs_delete_requested_at", "training_runs", ["delete_requested_at"], unique=False)

    op.create_table(
        "training_run_parameters",
        sa.Column("param_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("epochs", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("image_size", sa.Integer(), nullable=False),
        sa.Column("learning_rate", sa.DECIMAL(precision=10, scale=8), nullable=True),
        sa.Column("patience", sa.Integer(), nullable=False),
        sa.Column("device", sa.String(length=32), nullable=False),
        sa.Column("workers", sa.Integer(), nullable=False),
        sa.Column("use_pretrained", sa.Boolean(), nullable=False),
        sa.Column("optimizer", sa.String(length=64), nullable=False),
        sa.Column("augmentation", sa.JSON(), nullable=True),
        sa.Column("additional_params", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", name="uq_training_run_parameters_run_id"),
    )

    op.create_table(
        "training_run_results",
        sa.Column("result_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("best_weights_path", sa.String(length=500), nullable=True),
        sa.Column("last_weights_path", sa.String(length=500), nullable=True),
        sa.Column("results_dir", sa.String(length=500), nullable=True),
        sa.Column("final_metrics", sa.JSON(), nullable=True),
        sa.Column("best_metrics", sa.JSON(), nullable=True),
        sa.Column("model_size_mb", sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column("inference_time_ms", sa.DECIMAL(precision=10, scale=4), nullable=True),
        sa.Column("flops", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", name="uq_training_run_results_run_id"),
    )

    op.create_table(
        "training_run_events",
        sa.Column("event_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("level", sa.Enum("DEBUG", "INFO", "WARNING", "ERROR"), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_training_run_events_run_id", "training_run_events", ["run_id"], unique=False)
    op.create_index("ix_training_run_events_event_type", "training_run_events", ["event_type"], unique=False)
    op.create_index("ix_training_run_events_created_at", "training_run_events", ["created_at"], unique=False)

    op.create_table(
        "training_run_epoch_metrics",
        sa.Column("metric_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "epoch", name="uq_training_run_epoch_metrics_run_epoch"),
    )
    op.create_index("ix_training_run_epoch_metrics_run_id", "training_run_epoch_metrics", ["run_id"], unique=False)
    op.create_index("ix_training_run_epoch_metrics_epoch", "training_run_epoch_metrics", ["epoch"], unique=False)

    op.create_table(
        "training_run_artifacts",
        sa.Column("artifact_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_training_run_artifacts_run_id", "training_run_artifacts", ["run_id"], unique=False)
    op.create_index("ix_training_run_artifacts_kind", "training_run_artifacts", ["kind"], unique=False)
    op.create_index("ix_training_run_artifacts_created_at", "training_run_artifacts", ["created_at"], unique=False)

    # --------------------
    # model registry
    # --------------------
    op.create_table(
        "model_versions",
        sa.Column("model_version_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("stage", sa.Enum("development", "testing", "production", "deprecated"), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("weights_path", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"]),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.run_id"]),
        sa.UniqueConstraint("project_id", "version", name="uq_model_versions_project_version"),
    )
    op.create_index("ix_model_versions_project_id", "model_versions", ["project_id"], unique=False)
    op.create_index("ix_model_versions_run_id", "model_versions", ["run_id"], unique=False)
    op.create_index("ix_model_versions_stage", "model_versions", ["stage"], unique=False)

    # --------------------
    # deployments + logs
    # --------------------
    op.create_table(
        "deployments",
        sa.Column("deployment_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("platform", sa.Enum("local", "docker", "kubernetes", "aws", "azure", "gcp"), nullable=False),
        sa.Column("status", sa.Enum("pending", "deploying", "active", "inactive", "failed", "deleting"), nullable=False),
        sa.Column("endpoint_url", sa.String(length=500), nullable=True),
        sa.Column("health_check_url", sa.String(length=500), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.model_version_id"]),
    )
    op.create_index("ix_deployments_model_version_id", "deployments", ["model_version_id"], unique=False)
    op.create_index("ix_deployments_name", "deployments", ["name"], unique=False)
    op.create_index("ix_deployments_status", "deployments", ["status"], unique=False)
    op.create_index("ix_deployments_is_active", "deployments", ["is_active"], unique=False)
    op.create_index("ix_deployments_updated_at", "deployments", ["updated_at"], unique=False)

    op.create_table(
        "deployment_logs",
        sa.Column("log_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("deployment_id", sa.Integer(), nullable=False),
        sa.Column("level", sa.Enum("DEBUG", "INFO", "WARNING", "ERROR"), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["deployment_id"], ["deployments.deployment_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_deployment_logs_deployment_id", "deployment_logs", ["deployment_id"], unique=False)
    op.create_index("ix_deployment_logs_created_at", "deployment_logs", ["created_at"], unique=False)

    # --------------------
    # inference runs
    # --------------------
    op.create_table(
        "inference_runs",
        sa.Column("inference_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("deployment_id", sa.Integer(), nullable=True),
        sa.Column("input_path", sa.String(length=500), nullable=False),
        sa.Column("input_meta", sa.JSON(), nullable=True),
        sa.Column("output", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["deployment_id"], ["deployments.deployment_id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.model_version_id"]),
    )
    op.create_index("ix_inference_runs_model_version_id", "inference_runs", ["model_version_id"], unique=False)
    op.create_index("ix_inference_runs_deployment_id", "inference_runs", ["deployment_id"], unique=False)
    op.create_index("ix_inference_runs_created_at", "inference_runs", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_table("inference_runs")
    op.drop_table("deployment_logs")
    op.drop_table("deployments")
    op.drop_table("model_versions")
    op.drop_table("training_run_artifacts")
    op.drop_table("training_run_epoch_metrics")
    op.drop_table("training_run_events")
    op.drop_table("training_run_results")
    op.drop_table("training_run_parameters")
    op.drop_table("training_runs")
    op.drop_table("model_architectures")
    op.drop_table("projects")
    op.drop_constraint("fk_datasets_active_version_id", "datasets", type_="foreignkey")
    op.drop_table("dataset_versions")
    op.drop_table("datasets")

