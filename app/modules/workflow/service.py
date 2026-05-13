from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.modules.persistence.repositories import (
    EventRepository,
    ExecutionRepository,
    LogRepository,
    StepRepository,
    WorkflowRepository,
)
from app.modules.workflow.schemas import WorkflowCreate, WorkflowResponse

log = get_logger("workflow.service")


class WorkflowService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._workflows = WorkflowRepository(db)
        self._executions = ExecutionRepository(db)
        self._steps = StepRepository(db)
        self._logs = LogRepository(db)
        self._events = EventRepository(db)

    async def create_workflow(self, payload: WorkflowCreate) -> WorkflowResponse:
        workflow = await self._workflows.create(
            org_id=payload.org_id,
            project_id=payload.project_id,
            name=payload.name,
            description=payload.description,
            graph_definition=payload.graph_definition.model_dump(),
        )
        await self._db.commit()
        return WorkflowResponse.model_validate(workflow)

    async def get_execution_with_logs(
        self, execution_id: str
    ) -> dict[str, Any]:
        execution = await self._executions.get_or_raise(execution_id)
        logs = await self._logs.list_by_execution(execution_id)
        steps = await self._steps.list_by_execution(execution_id)
        return {
            "execution": execution,
            "logs": logs,
            "steps": steps,
        }

    async def mark_execution_running(self, execution_id: str) -> None:
        await self._executions.update_status(
            execution_id, "running", started_at=datetime.utcnow()
        )
        await self._db.commit()

    async def mark_execution_complete(
        self, execution_id: str, output: dict[str, Any]
    ) -> None:
        await self._executions.update_status(
            execution_id,
            "complete",
            output=output,
            finished_at=datetime.utcnow(),
        )
        await self._db.commit()

    async def mark_execution_failed(
        self, execution_id: str, error: str
    ) -> None:
        await self._executions.update_status(
            execution_id,
            "failed",
            error_message=error,
            finished_at=datetime.utcnow(),
        )
        await self._db.commit()
