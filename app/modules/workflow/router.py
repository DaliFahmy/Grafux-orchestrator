from __future__ import annotations

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.core.security import verify_jwt
from app.dependencies import CurrentUser, DBSession
from app.modules.persistence.repositories import ExecutionRepository, LogRepository, WorkflowRepository
from app.modules.streaming.websocket import connection_manager
from app.modules.workflow.schemas import (
    ExecutionLogResponse,
    ExecutionResponse,
    WorkflowCreate,
    WorkflowResponse,
    WorkflowTemplateResponse,
)
from app.modules.workflow.service import WorkflowService
from app.modules.workflow.templates import TemplateService

log = get_logger("workflow.router")

router = APIRouter(prefix="/workflows", tags=["workflows"])
executions_router = APIRouter(prefix="/executions", tags=["executions"])


# ── Workflow CRUD ─────────────────────────────────────────────────────────────

@router.post("", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    db: DBSession,
    user: CurrentUser,
) -> WorkflowResponse:
    service = WorkflowService(db)
    return await service.create_workflow(body)


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: DBSession,
    user: CurrentUser,
) -> WorkflowResponse:
    repo = WorkflowRepository(db)
    workflow = await repo.get_or_raise(workflow_id)
    return WorkflowResponse.model_validate(workflow)


@router.get("/templates/", response_model=list[WorkflowTemplateResponse])
async def list_templates(
    db: DBSession,
    category: str | None = Query(default=None),
) -> list[WorkflowTemplateResponse]:
    service = TemplateService(db)
    return await service.list_templates(category=category)


# ── Execution queries ─────────────────────────────────────────────────────────

@executions_router.get("/{execution_id}", response_model=ExecutionResponse)
async def get_execution(
    execution_id: str,
    db: DBSession,
    user: CurrentUser,
) -> ExecutionResponse:
    repo = ExecutionRepository(db)
    execution = await repo.get_or_raise(execution_id)
    return ExecutionResponse(
        id=execution.id,
        workflow_id=execution.workflow_id,
        org_id=execution.org_id,
        project_id=execution.project_id,
        status=execution.status,
        input=execution.input,
        output=execution.output,
        error_message=execution.error_message,
        retry_count=execution.retry_count,
        started_at=execution.started_at.isoformat() if execution.started_at else None,
        finished_at=execution.finished_at.isoformat() if execution.finished_at else None,
        created_at=execution.created_at.isoformat(),
    )


@executions_router.get("/{execution_id}/logs", response_model=list[ExecutionLogResponse])
async def get_execution_logs(
    execution_id: str,
    db: DBSession,
    user: CurrentUser,
) -> list[ExecutionLogResponse]:
    repo = LogRepository(db)
    logs = await repo.list_by_execution(execution_id)
    return [
        ExecutionLogResponse(
            id=log_entry.id,
            level=log_entry.level,
            message=log_entry.message,
            step_id=log_entry.step_id,
            metadata=log_entry.extra_data,
            timestamp=log_entry.timestamp.isoformat(),
        )
        for log_entry in logs
    ]


# ── WebSocket execution streaming ─────────────────────────────────────────────

@executions_router.websocket("/{execution_id}/stream")
async def execution_stream(
    execution_id: str,
    websocket: WebSocket,
    token: str = Query(default=""),
) -> None:
    """WebSocket endpoint for real-time execution event streaming."""
    payload = await verify_jwt(token)
    if not payload:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await connection_manager.connect(execution_id, websocket)
    try:
        # Keep the connection alive; events are pushed by the broadcaster
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(execution_id, websocket)
