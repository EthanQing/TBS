"""Add GPU inventory and training resource requests.

Revision ID: 0024_gpu_inventory_resources
Revises: 0023_training_artifact_role
"""

from alembic import op
import sqlalchemy as sa


revision = "0024_gpu_inventory_resources"
down_revision = "0023_training_artifact_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gpu_devices",
        sa.Column("gpu_uuid", sa.String(80), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("pci_bus_id", sa.String(64)),
        sa.Column("node_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_gpu_devices_node_id", "gpu_devices", ["node_id"])

    op.create_table(
        "gpu_worker_instances",
        sa.Column("instance_id", sa.String(36), primary_key=True),
        sa.Column("worker_id", sa.String(128), nullable=False),
        sa.Column("node_id", sa.String(128)),
        sa.Column("hostname", sa.String(255), nullable=False),
        sa.Column("process_scope", sa.JSON(), nullable=True),
        sa.Column("allowed_engines", sa.JSON(), nullable=False),
        sa.Column("nvidia_visible_devices", sa.Text()),
        sa.Column("cuda_visible_devices", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True)),
        sa.Column("inventory_status", sa.String(32), nullable=False),
        sa.Column("inventory_error", sa.Text()),
        sa.Column("last_successful_inventory_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_gpu_worker_instances_worker_id", "gpu_worker_instances", ["worker_id"])
    op.create_index("ix_gpu_worker_instances_node_id", "gpu_worker_instances", ["node_id"])
    op.create_index("ix_gpu_worker_instances_heartbeat_at", "gpu_worker_instances", ["heartbeat_at"])

    op.create_table(
        "gpu_worker_observations",
        sa.Column("observation_id", sa.Integer(), primary_key=True),
        sa.Column("instance_id", sa.String(36), sa.ForeignKey("gpu_worker_instances.instance_id", ondelete="CASCADE"), nullable=False),
        sa.Column("gpu_uuid", sa.String(80), sa.ForeignKey("gpu_devices.gpu_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("observed_index", sa.Integer()),
        sa.Column("memory_total_mib", sa.Integer()),
        sa.Column("memory_used_mib", sa.Integer()),
        sa.Column("memory_free_mib", sa.Integer()),
        sa.Column("utilization_percent", sa.Integer()),
        sa.Column("compute_mode", sa.String(64)),
        sa.Column("mig_mode", sa.String(64)),
        sa.Column("probe_source", sa.String(32), nullable=False),
        sa.Column("probe_status", sa.String(32), nullable=False),
        sa.Column("probe_error", sa.Text()),
        sa.Column("present", sa.Boolean(), nullable=False),
        sa.Column("sampled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("instance_id", "gpu_uuid", name="uq_gpu_worker_observation_instance_gpu"),
    )
    op.create_index("ix_gpu_worker_observations_instance_id", "gpu_worker_observations", ["instance_id"])
    op.create_index("ix_gpu_worker_observations_gpu_uuid", "gpu_worker_observations", ["gpu_uuid"])
    op.create_index("ix_gpu_worker_observations_sampled_at", "gpu_worker_observations", ["sampled_at"])

    op.create_table(
        "training_run_resource_requests",
        sa.Column("run_id", sa.String(36), sa.ForeignKey("training_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("selection", sa.String(16), nullable=False),
        sa.Column("gpu_count", sa.Integer(), nullable=False),
        sa.Column("gpu_uuids", sa.JSON(), nullable=False),
        sa.Column("memory_mib_per_gpu", sa.Integer()),
        sa.Column("sharing", sa.String(16), nullable=False),
        sa.Column("node_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("training_run_resource_requests")
    op.drop_table("gpu_worker_observations")
    op.drop_table("gpu_worker_instances")
    op.drop_table("gpu_devices")
