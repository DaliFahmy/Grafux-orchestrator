from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import update

from app.core.logging import get_logger
from app.dependencies import DBSession, InternalService
from app.modules.internal_api.health import get_health
from app.modules.internal_api.schemas import (
    CancelExecutionResponse,
    ExecutionStatusResponse,
    HealthResponse,
    TriggerExecutionRequest,
    TriggerExecutionResponse,
)
from app.modules.persistence.models import Execution
from app.modules.persistence.repositories import ExecutionRepository

log = get_logger("internal_api")

router = APIRouter(prefix="/internal", tags=["internal"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness + readiness probe. No auth required for load-balancer checks."""
    return await get_health()


@router.post(
    "/executions/trigger",
    response_model=TriggerExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[InternalService],  # type: ignore[list-item]
)
async def trigger_execution(
    body: TriggerExecutionRequest,
    db: DBSession,
) -> TriggerExecutionResponse:
    """Accept an execution request from Grafux-backend and enqueue it."""
    repo = ExecutionRepository(db)
    execution = await repo.create(
        id=str(uuid.uuid4()),
        workflow_id=body.workflow_id,
        org_id=body.org_id,
        project_id=body.project_id,
        triggered_by=body.triggered_by,
        input=body.input,
        status="pending",
    )
    await db.commit()

    # Enqueue Celery task (import here to avoid circular import at module load)
    from app.modules.queue.tasks import execute_workflow

    task = execute_workflow.apply_async(
        kwargs={
            "execution_id": execution.id,
            "workflow_id": body.workflow_id,
            "org_id": body.org_id,
            "project_id": body.project_id,
            "input_data": body.input,
            "graph_definition": body.graph_definition,
        },
        queue="ai_heavy",
    )

    # Store the Celery task ID
    await repo.update_status(execution.id, "pending")
    await db.execute(
        update(Execution)
        .where(Execution.id == execution.id)
        .values(celery_task_id=task.id)
    )
    await db.commit()

    log.info(
        "execution_triggered",
        execution_id=execution.id,
        celery_task_id=task.id,
        org_id=body.org_id,
    )

    return TriggerExecutionResponse(execution_id=execution.id, status="pending")


@router.get(
    "/executions/{execution_id}",
    response_model=ExecutionStatusResponse,
    dependencies=[InternalService],  # type: ignore[list-item]
)
async def get_execution_status(
    execution_id: str,
    db: DBSession,
) -> ExecutionStatusResponse:
    repo = ExecutionRepository(db)
    execution = await repo.get_or_raise(execution_id)
    return ExecutionStatusResponse(
        execution_id=execution.id,
        status=execution.status,
        workflow_id=execution.workflow_id,
        org_id=execution.org_id,
        project_id=execution.project_id,
        retry_count=execution.retry_count,
        started_at=execution.started_at.isoformat() if execution.started_at else None,
        finished_at=execution.finished_at.isoformat() if execution.finished_at else None,
        error_message=execution.error_message,
    )


@router.post(
    "/executions/{execution_id}/cancel",
    response_model=CancelExecutionResponse,
    dependencies=[InternalService],  # type: ignore[list-item]
)
async def cancel_execution(
    execution_id: str,
    db: DBSession,
) -> CancelExecutionResponse:
    repo = ExecutionRepository(db)
    execution = await repo.get_or_raise(execution_id)

    if execution.status in ("complete", "failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Execution is already in terminal state: {execution.status}",
        )

    # Revoke the Celery task if it's running
    if execution.celery_task_id:
        from app.core.celery_app import get_celery_app
        celery = get_celery_app()
        celery.control.revoke(execution.celery_task_id, terminate=True, signal="SIGTERM")

    # Signal cancellation via Redis so the running graph can detect it
    from app.core.redis import get_redis_client
    redis = get_redis_client()
    await redis.set(f"cancel:{execution_id}", "1", ex=3600)

    await repo.update_status(
        execution_id,
        "cancelled",
        finished_at=datetime.utcnow(),
    )
    await db.commit()

    log.info("execution_cancelled", execution_id=execution_id)

    return CancelExecutionResponse(execution_id=execution_id, status="cancelled")


@router.post(
    "/workflows/validate",
    dependencies=[InternalService],  # type: ignore[list-item]
)
async def validate_workflow(body: dict) -> dict:
    """Pre-flight validation of a workflow graph definition."""
    from app.modules.workflow.engine import WorkflowEngine
    try:
        engine = WorkflowEngine()
        await engine.validate_graph(body.get("graph_definition", {}))
        return {"valid": True}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
