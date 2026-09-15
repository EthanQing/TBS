"""Add managed GPU scheduling ledger.

Revision ID: 0025_gpu_managed_scheduling
Revises: 0024_gpu_inventory_resources
"""
from alembic import op
import sqlalchemy as sa

revision = "0025_gpu_managed_scheduling"
down_revision = "0024_gpu_inventory_resources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gpu_worker_instances", sa.Column("cuda_inventory_status", sa.String(32), nullable=False, server_default="pending"))
    op.add_column("gpu_worker_instances", sa.Column("cuda_inventory_error", sa.Text()))
    op.add_column("gpu_worker_instances", sa.Column("cuda_environment_fingerprint", sa.String(64)))
    op.add_column("gpu_worker_instances", sa.Column("last_successful_cuda_inventory_at", sa.DateTime(timezone=True)))
    op.add_column("gpu_worker_instances", sa.Column("max_training_slots", sa.Integer(), nullable=False, server_default="2"))
    op.add_column("gpu_worker_instances", sa.Column("running_task_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("gpu_worker_instances", sa.Column("accepting_tasks", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("gpu_worker_instances", sa.Column("launcher_identity", sa.JSON()))
    op.add_column("gpu_worker_observations", sa.Column("process_snapshot", sa.JSON()))
    op.add_column("training_runs", sa.Column("current_allocation_id", sa.String(36)))
    op.add_column("training_runs", sa.Column("resource_wait_reason", sa.String(64)))
    op.add_column("training_runs", sa.Column("resource_wait_details", sa.JSON()))
    op.create_index("ix_training_runs_current_allocation_id", "training_runs", ["current_allocation_id"])
    op.create_index("ix_training_runs_resource_wait_reason", "training_runs", ["resource_wait_reason"])

    op.create_table(
        "gpu_node_scheduling_states",
        sa.Column("node_id", sa.String(128), primary_key=True),
        sa.Column("managed", sa.Boolean(), nullable=False),
        sa.Column("accepting_allocations", sa.Boolean(), nullable=False),
        sa.Column("shared_execution_enabled", sa.Boolean(), nullable=False),
        sa.Column("max_shared_tasks_per_device", sa.Integer(), nullable=False),
        sa.Column("memory_safety_mib", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "gpu_allocations",
        sa.Column("allocation_id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("training_runs.run_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("active_run_id", sa.String(36), nullable=True),
        sa.Column("worker_instance_id", sa.String(36), sa.ForeignKey("gpu_worker_instances.instance_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("worker_id", sa.String(128), nullable=False),
        sa.Column("node_id", sa.String(128), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("authorization_state", sa.String(16), nullable=False),
        sa.Column("request_snapshot", sa.JSON(), nullable=False),
        sa.Column("execution_owner", sa.JSON()),
        sa.Column("launcher_identity", sa.JSON()),
        sa.Column("execution_result", sa.JSON()),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("launch_deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("released_at", sa.DateTime(timezone=True)),
        sa.Column("exit_code", sa.Integer()),
        sa.Column("error_message", sa.Text()),
        sa.UniqueConstraint("active_run_id", name="uq_gpu_allocations_active_run_id"),
    )
    for column in ("run_id", "worker_instance_id", "worker_id", "node_id", "state"):
        op.create_index(f"ix_gpu_allocations_{column}", "gpu_allocations", [column])
    op.create_table(
        "gpu_allocation_devices",
        sa.Column("allocation_device_id", sa.Integer(), primary_key=True),
        sa.Column("allocation_id", sa.String(36), sa.ForeignKey("gpu_allocations.allocation_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("gpu_uuid", sa.String(80), sa.ForeignKey("gpu_devices.gpu_uuid", ondelete="RESTRICT"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("sharing", sa.String(16), nullable=False),
        sa.Column("requested_memory_mib", sa.Integer()),
        sa.Column("reserved_memory_mib", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("allocation_id", "gpu_uuid", name="uq_gpu_allocation_device_gpu"),
        sa.UniqueConstraint("allocation_id", "ordinal", name="uq_gpu_allocation_device_ordinal"),
    )
    op.create_index("ix_gpu_allocation_devices_allocation_id", "gpu_allocation_devices", ["allocation_id"])
    op.create_index("ix_gpu_allocation_devices_gpu_uuid", "gpu_allocation_devices", ["gpu_uuid"])
    op.create_table(
        "gpu_cuda_bindings",
        sa.Column("binding_id", sa.Integer(), primary_key=True),
        sa.Column("instance_id", sa.String(36), sa.ForeignKey("gpu_worker_instances.instance_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("gpu_uuid", sa.String(80), sa.ForeignKey("gpu_devices.gpu_uuid", ondelete="RESTRICT"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("pci_bus_id", sa.String(64)),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("environment_fingerprint", sa.String(64), nullable=False),
        sa.Column("present", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error", sa.Text()),
        sa.UniqueConstraint("instance_id", "gpu_uuid", name="uq_gpu_cuda_binding_instance_gpu"),
    )
    op.create_index("ix_gpu_cuda_bindings_instance_id", "gpu_cuda_bindings", ["instance_id"])
    op.create_index("ix_gpu_cuda_bindings_gpu_uuid", "gpu_cuda_bindings", ["gpu_uuid"])
    op.create_index("ix_gpu_cuda_bindings_verified_at", "gpu_cuda_bindings", ["verified_at"])


def downgrade() -> None:
    op.drop_table("gpu_cuda_bindings")
    op.drop_table("gpu_allocation_devices")
    op.drop_table("gpu_allocations")
    op.drop_table("gpu_node_scheduling_states")
    op.drop_index("ix_training_runs_resource_wait_reason", table_name="training_runs")
    op.drop_index("ix_training_runs_current_allocation_id", table_name="training_runs")
    for column in ("resource_wait_details", "resource_wait_reason", "current_allocation_id"):
        op.drop_column("training_runs", column)
    op.drop_column("gpu_worker_observations", "process_snapshot")
    for column in ("launcher_identity", "accepting_tasks", "running_task_count", "max_training_slots", "last_successful_cuda_inventory_at", "cuda_environment_fingerprint", "cuda_inventory_error", "cuda_inventory_status"):
        op.drop_column("gpu_worker_instances", column)
