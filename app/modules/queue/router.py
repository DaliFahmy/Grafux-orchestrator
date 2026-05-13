from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.celery_app import get_celery_app
from app.dependencies import InternalService

router = APIRouter(prefix="/internal/queue", tags=["queue"])


@router.get(
    "/status",
    dependencies=[InternalService],  # type: ignore[list-item]
)
async def queue_status() -> dict[str, Any]:
    """Return current Celery worker and queue status."""
    celery = get_celery_app()
    import asyncio
    inspector = celery.control.inspect(timeout=2.0)
    active = await asyncio.get_event_loop().run_in_executor(None, inspector.active)
    reserved = await asyncio.get_event_loop().run_in_executor(None, inspector.reserved)
    stats = await asyncio.get_event_loop().run_in_executor(None, inspector.stats)
    return {
        "active_tasks": active or {},
        "reserved_tasks": reserved or {},
        "worker_stats": stats or {},
    }


@router.post(
    "/retry/{execution_id}",
    dependencies=[InternalService],  # type: ignore[list-item]
)
async def retry_execution(execution_id: str) -> dict[str, Any]:
    """Manually trigger a retry for a failed execution."""
    from app.modules.queue.tasks import retry_failed_execution
    task = retry_failed_execution.apply_async(
        kwargs={"execution_id": execution_id},
        queue="ai_heavy",
    )
    return {"task_id": task.id, "execution_id": execution_id}
