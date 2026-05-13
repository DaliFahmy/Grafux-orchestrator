"""Initial schema for Grafux-orchestrator

Revision ID: 001
Revises:
Create Date: 2026-05-12

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── workflows ──────────────────────────────────────────────────────────────
    op.create_table(
        "workflows",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("graph_definition", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_workflows_org_project", "workflows", ["org_id", "project_id"])

    # ── workflow_templates ─────────────────────────────────────────────────────
    op.create_table(
        "workflow_templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("graph_definition", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("is_public", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_workflow_templates_category", "workflow_templates", ["category"])

    # ── executions ─────────────────────────────────────────────────────────────
    op.create_table(
        "executions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("workflow_id", sa.String(36), sa.ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True),
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("triggered_by", sa.String(36)),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("input", sa.JSON, server_default="{}"),
        sa.Column("output", sa.JSON),
        sa.Column("error_message", sa.Text),
        sa.Column("celery_task_id", sa.String(255)),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_executions_org_id", "executions", ["org_id"])
    op.create_index("ix_executions_project_id", "executions", ["project_id"])
    op.create_index("ix_executions_status", "executions", ["status"])
    op.create_index("ix_executions_org_status", "executions", ["org_id", "status"])

    # ── execution_steps ────────────────────────────────────────────────────────
    op.create_table(
        "execution_steps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), sa.ForeignKey("executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("node_name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("input", sa.JSON),
        sa.Column("output", sa.JSON),
        sa.Column("error_message", sa.Text),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
    )
    op.create_index("ix_execution_steps_execution_id", "execution_steps", ["execution_id"])

    # ── execution_logs ─────────────────────────────────────────────────────────
    op.create_table(
        "execution_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), sa.ForeignKey("executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_id", sa.String(36)),
        sa.Column("level", sa.String(20), nullable=False, server_default="info"),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_execution_logs_execution_id", "execution_logs", ["execution_id"])
    op.create_index("ix_execution_logs_timestamp", "execution_logs", ["timestamp"])

    # ── execution_events ───────────────────────────────────────────────────────
    op.create_table(
        "execution_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), sa.ForeignKey("executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", sa.JSON, server_default="{}"),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_execution_events_execution_id", "execution_events", ["execution_id"])

    # ── agent_sessions ─────────────────────────────────────────────────────────
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), sa.ForeignKey("executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_id", sa.String(255), nullable=False),
        sa.Column("memory", sa.JSON, server_default="{}"),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ── workflow_checkpoints ───────────────────────────────────────────────────
    op.create_table(
        "workflow_checkpoints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), sa.ForeignKey("executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("thread_id", sa.String(255), nullable=False),
        sa.Column("checkpoint_data", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_workflow_checkpoints_execution_id", "workflow_checkpoints", ["execution_id"])

    # ── mcp_invocations ────────────────────────────────────────────────────────
    op.create_table(
        "mcp_invocations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), nullable=False),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("input", sa.JSON, server_default="{}"),
        sa.Column("output", sa.JSON),
        sa.Column("error_message", sa.Text),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_mcp_invocations_execution_id", "mcp_invocations", ["execution_id"])

    # ── research_results ───────────────────────────────────────────────────────
    op.create_table(
        "research_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("sources", sa.JSON, server_default="[]"),
        sa.Column("summary", sa.Text),
        sa.Column("citations", sa.JSON, server_default="[]"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_research_results_execution_id", "research_results", ["execution_id"])

    # ── sandbox_sessions ───────────────────────────────────────────────────────
    op.create_table(
        "sandbox_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), nullable=False),
        sa.Column("e2b_session_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("output", sa.Text),
        sa.Column("error_output", sa.Text),
        sa.Column("exit_code", sa.Integer),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime),
    )

    # ── scheduled_tasks ────────────────────────────────────────────────────────
    op.create_table(
        "scheduled_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("workflow_id", sa.String(36)),
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("cron_expr", sa.String(100)),
        sa.Column("input", sa.JSON, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("next_run_at", sa.DateTime),
        sa.Column("last_run_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_scheduled_tasks_org_id", "scheduled_tasks", ["org_id"])
    op.create_index("ix_scheduled_tasks_next_run_at", "scheduled_tasks", ["next_run_at"])

    # ── execution_retries ──────────────────────────────────────────────────────
    op.create_table(
        "execution_retries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("execution_id", sa.String(36), nullable=False),
        sa.Column("attempt", sa.Integer, nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("celery_task_id", sa.String(255)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_execution_retries_execution_id", "execution_retries", ["execution_id"])


def downgrade() -> None:
    op.drop_table("execution_retries")
    op.drop_table("scheduled_tasks")
    op.drop_table("sandbox_sessions")
    op.drop_table("research_results")
    op.drop_table("mcp_invocations")
    op.drop_table("workflow_checkpoints")
    op.drop_table("agent_sessions")
    op.drop_table("execution_events")
    op.drop_table("execution_logs")
    op.drop_table("execution_steps")
    op.drop_table("executions")
    op.drop_table("workflow_templates")
    op.drop_table("workflows")
