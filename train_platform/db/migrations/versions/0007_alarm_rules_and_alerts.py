"""Add alarm rules and alerts tables.

Revision ID: 0007_alarm_rules_and_alerts
Revises: 0006_deployment_runtime_v1
Create Date: 2026-03-13
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_alarm_rules_and_alerts"
down_revision = "0006_deployment_runtime_v1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alarm_rules",
        sa.Column("rule_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("rule_type", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("cooldown_seconds", sa.Integer(), nullable=False, server_default=sa.text("300")),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("rule_type", name="uq_alarm_rules_rule_type"),
    )
    op.create_index("ix_alarm_rules_rule_type", "alarm_rules", ["rule_type"], unique=True)
    op.create_index("ix_alarm_rules_enabled", "alarm_rules", ["enabled"], unique=False)
    op.create_index("ix_alarm_rules_severity", "alarm_rules", ["severity"], unique=False)

    op.create_table(
        "alarm_alerts",
        sa.Column("alert_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("rule_id", sa.Integer(), nullable=True),
        sa.Column("rule_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("trigger_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("first_triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acked_by", sa.String(length=128), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["rule_id"], ["alarm_rules.rule_id"], ondelete="SET NULL"),
    )
    op.create_index("ix_alarm_alerts_rule_id", "alarm_alerts", ["rule_id"], unique=False)
    op.create_index("ix_alarm_alerts_rule_type", "alarm_alerts", ["rule_type"], unique=False)
    op.create_index("ix_alarm_alerts_severity", "alarm_alerts", ["severity"], unique=False)
    op.create_index("ix_alarm_alerts_status", "alarm_alerts", ["status"], unique=False)
    op.create_index("ix_alarm_alerts_source_type", "alarm_alerts", ["source_type"], unique=False)
    op.create_index("ix_alarm_alerts_source_id", "alarm_alerts", ["source_id"], unique=False)
    op.create_index("ix_alarm_alerts_last_triggered_at", "alarm_alerts", ["last_triggered_at"], unique=False)
    op.create_index("ix_alarm_alerts_source", "alarm_alerts", ["source_type", "source_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_alarm_alerts_source", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_last_triggered_at", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_source_id", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_source_type", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_status", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_severity", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_rule_type", table_name="alarm_alerts")
    op.drop_index("ix_alarm_alerts_rule_id", table_name="alarm_alerts")
    op.drop_table("alarm_alerts")

    op.drop_index("ix_alarm_rules_severity", table_name="alarm_rules")
    op.drop_index("ix_alarm_rules_enabled", table_name="alarm_rules")
    op.drop_index("ix_alarm_rules_rule_type", table_name="alarm_rules")
    op.drop_table("alarm_rules")

