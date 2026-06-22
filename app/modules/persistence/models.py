from __future__ import annotations

from datetime import datetime  # noqa: F401 — referenced by Mapped[datetime] annotations

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.shared.factories import generate_uuid as _uuid
from app.shared.factories import utcnow as _now

# ── Workflow ───────────────────────────────────────────────────────────────────

class Workflow(Base):
    """Persistent workflow definition (graph + metadata)."""

    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    graph_definition: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    executions: Mapped[list[Execution]] = relationship(
        back_populates="workflow", lazy="noload"
    )

    __table_args__ = (Index("ix_workflows_org_project", "org_id", "project_id"),)


class WorkflowTemplate(Base):
    """Reusable workflow templates available to all organisations."""

    __tablename__ = "workflow_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    graph_definition: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


# ── Execution ─────────────────────────────────────────────────────────────────

class Execution(Base):
    """A single run of a workflow graph."""

    __tablename__ = "executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workflow_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True
    )
    org_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    triggered_by: Mapped[str | None] = mapped_column(String(36))  # user_id or service name
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )
    # pending | running | complete | failed | cancelled | retrying
    input: Mapped[dict] = mapped_column(JSON, default=dict)
    output: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    workflow: Mapped[Workflow | None] = relationship(
        back_populates="executions", lazy="noload"
    )
    steps: Mapped[list[ExecutionStep]] = relationship(
        back_populates="execution", lazy="noload", cascade="all, delete-orphan"
    )
    logs: Mapped[list[ExecutionLog]] = relationship(
        back_populates="execution", lazy="noload", cascade="all, delete-orphan"
    )
    events: Mapped[list[ExecutionEvent]] = relationship(
        back_populates="execution", lazy="noload", cascade="all, delete-orphan"
    )
    checkpoints: Mapped[list[WorkflowCheckpoint]] = relationship(
        back_populates="execution", lazy="noload", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_executions_org_status", "org_id", "status"),)


class ExecutionStep(Base):
    """One node execution within a workflow run."""

    __tablename__ = "execution_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("executions.id", ondelete="CASCADE"), index=True
    )
    node_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    input: Mapped[dict | None] = mapped_column(JSON)
    output: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    execution: Mapped[Execution] = relationship(
        back_populates="steps", lazy="noload"
    )


class ExecutionLog(Base):
    """Structured log entries for an execution."""

    __tablename__ = "execution_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("executions.id", ondelete="CASCADE"), index=True
    )
    step_id: Mapped[str | None] = mapped_column(String(36))
    level: Mapped[str] = mapped_column(String(20), nullable=False, default="info")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra_data: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)

    execution: Mapped[Execution] = relationship(
        back_populates="logs", lazy="noload"
    )


class ExecutionEvent(Base):
    """Structured events published during an execution (also stored for replay)."""

    __tablename__ = "execution_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("executions.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)

    execution: Mapped[Execution] = relationship(
        back_populates="events", lazy="noload"
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentSession(Base):
    """Runtime state for an AI agent within an execution."""

    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("executions.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False)
    memory: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


# ── Checkpoints ───────────────────────────────────────────────────────────────

class WorkflowCheckpoint(Base):
    """LangGraph execution state snapshot for resumability."""

    __tablename__ = "workflow_checkpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("executions.id", ondelete="CASCADE"), index=True
    )
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    checkpoint_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    execution: Mapped[Execution] = relationship(
        back_populates="checkpoints", lazy="noload"
    )


# ── External Integrations ─────────────────────────────────────────────────────

class MCPInvocation(Base):
    """Audit log for every MCP tool call made during an execution."""

    __tablename__ = "mcp_invocations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(String(36), index=True)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    input: Mapped[dict] = mapped_column(JSON, default=dict)
    output: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ResearchResult(Base):
    """Stored results from research pipelines."""

    __tablename__ = "research_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(String(36), index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class SandboxSession(Base):
    """E2B sandbox session tracking."""

    __tablename__ = "sandbox_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(String(36), index=True)
    e2b_session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
    output: Mapped[str | None] = mapped_column(Text)
    error_output: Mapped[str | None] = mapped_column(Text)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


# ── Scheduling ────────────────────────────────────────────────────────────────

class ScheduledTask(Base):
    """Periodic or future-scheduled workflow runs."""

    __tablename__ = "scheduled_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    workflow_id: Mapped[str | None] = mapped_column(String(36))
    org_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    cron_expr: Mapped[str | None] = mapped_column(String(100))
    input: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


# ── Execution Retries ─────────────────────────────────────────────────────────

class ExecutionRetry(Base):
    """Records each retry attempt for a failed execution."""

    __tablename__ = "execution_retries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    execution_id: Mapped[str] = mapped_column(String(36), index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
