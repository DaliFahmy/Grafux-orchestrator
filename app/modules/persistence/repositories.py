from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.persistence.models import (
    Execution,
    ExecutionEvent,
    ExecutionLog,
    ExecutionRetry,
    ExecutionStep,
    MCPInvocation,
    ResearchResult,
    SandboxSession,
    ScheduledTask,
    Workflow,
    WorkflowCheckpoint,
    WorkflowTemplate,
)


# ── Execution Repository ───────────────────────────────────────────────────────

class ExecutionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, **kwargs: Any) -> Execution:
        obj = Execution(**kwargs)
        self._db.add(obj)
        await self._db.flush()
        return obj

    async def get(self, execution_id: str) -> Execution | None:
        result = await self._db.execute(
            select(Execution).where(Execution.id == execution_id)
        )
        return result.scalar_one_or_none()

    async def get_or_raise(self, execution_id: str) -> Execution:
        from app.core.exceptions import ExecutionNotFoundError
        obj = await self.get(execution_id)
        if obj is None:
            raise ExecutionNotFoundError(execution_id)
        return obj

    async def list_by_org(
        self,
        org_id: str,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Execution], int]:
        from sqlalchemy import func
        q = select(Execution).where(Execution.org_id == org_id)
        if project_id:
            q = q.where(Execution.project_id == project_id)
        if status:
            q = q.where(Execution.status == status)

        count_q = select(func.count()).select_from(q.subquery())
        total = (await self._db.execute(count_q)).scalar_one()
        q = q.order_by(Execution.created_at.desc()).limit(limit).offset(offset)
        rows = (await self._db.execute(q)).scalars().all()
        return list(rows), total

    async def update_status(
        self,
        execution_id: str,
        status: str,
        *,
        output: dict | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if output is not None:
            values["output"] = output
        if error_message is not None:
            values["error_message"] = error_message
        if started_at is not None:
            values["started_at"] = started_at
        if finished_at is not None:
            values["finished_at"] = finished_at

        await self._db.execute(
            update(Execution).where(Execution.id == execution_id).values(**values)
        )

    async def increment_retry(self, execution_id: str) -> None:
        obj = await self.get_or_raise(execution_id)
        obj.retry_count += 1
        await self._db.flush()


# ── Step Repository ────────────────────────────────────────────────────────────

class StepRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, **kwargs: Any) -> ExecutionStep:
        obj = ExecutionStep(**kwargs)
        self._db.add(obj)
        await self._db.flush()
        return obj

    async def update(self, step_id: str, **kwargs: Any) -> None:
        await self._db.execute(
            update(ExecutionStep).where(ExecutionStep.id == step_id).values(**kwargs)
        )

    async def list_by_execution(self, execution_id: str) -> list[ExecutionStep]:
        result = await self._db.execute(
            select(ExecutionStep)
            .where(ExecutionStep.execution_id == execution_id)
            .order_by(ExecutionStep.started_at)
        )
        return list(result.scalars().all())


# ── Log Repository ─────────────────────────────────────────────────────────────

class LogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def append(
        self,
        execution_id: str,
        message: str,
        level: str = "info",
        step_id: str | None = None,
        extra_data: dict | None = None,
    ) -> ExecutionLog:
        obj = ExecutionLog(
            execution_id=execution_id,
            message=message,
            level=level,
            step_id=step_id,
            extra_data=extra_data or {},
        )
        self._db.add(obj)
        await self._db.flush()
        return obj

    async def list_by_execution(
        self, execution_id: str, limit: int = 500
    ) -> list[ExecutionLog]:
        result = await self._db.execute(
            select(ExecutionLog)
            .where(ExecutionLog.execution_id == execution_id)
            .order_by(ExecutionLog.timestamp)
            .limit(limit)
        )
        return list(result.scalars().all())


# ── Event Repository ───────────────────────────────────────────────────────────

class EventRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def append(
        self, execution_id: str, event_type: str, payload: dict
    ) -> ExecutionEvent:
        obj = ExecutionEvent(
            execution_id=execution_id,
            event_type=event_type,
            payload=payload,
        )
        self._db.add(obj)
        await self._db.flush()
        return obj

    async def list_by_execution(self, execution_id: str) -> list[ExecutionEvent]:
        result = await self._db.execute(
            select(ExecutionEvent)
            .where(ExecutionEvent.execution_id == execution_id)
            .order_by(ExecutionEvent.timestamp)
        )
        return list(result.scalars().all())


# ── Workflow Repository ────────────────────────────────────────────────────────

class WorkflowRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, **kwargs: Any) -> Workflow:
        obj = Workflow(**kwargs)
        self._db.add(obj)
        await self._db.flush()
        return obj

    async def get(self, workflow_id: str) -> Workflow | None:
        result = await self._db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        return result.scalar_one_or_none()

    async def get_or_raise(self, workflow_id: str) -> Workflow:
        from app.core.exceptions import WorkflowNotFoundError
        obj = await self.get(workflow_id)
        if obj is None:
            raise WorkflowNotFoundError(workflow_id)
        return obj

    async def list_by_project(
        self, org_id: str, project_id: str
    ) -> list[Workflow]:
        result = await self._db.execute(
            select(Workflow)
            .where(Workflow.org_id == org_id, Workflow.project_id == project_id)
            .order_by(Workflow.created_at.desc())
        )
        return list(result.scalars().all())
