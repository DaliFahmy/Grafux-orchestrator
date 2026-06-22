from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.celery_app import get_celery_app
from app.dependencies import require_internal_service

_internal = Depends(require_internal_service)

router = APIRouter(prefix="/internal/queue", tags=["queue"])


@router.get(
    "/status",
    dependencies=[_internal],
)
async def queue_status() -> dict[str, Any]:
    """Return current Celery worker and queue status."""
    import asyncio

    celery = get_celery_app()
    inspector = celery.control.inspect(timeout=2.0)
    loop = asyncio.get_running_loop()
    # The three inspect RPCs are independent — run them concurrently, not back-to-back.
    active, reserved, stats = await asyncio.gather(
        loop.run_in_executor(None, inspector.active),
        loop.run_in_executor(None, inspector.reserved),
        loop.run_in_executor(None, inspector.stats),
    )
    return {
        "active_tasks": active or {},
        "reserved_tasks": reserved or {},
        "worker_stats": stats or {},
    }


@router.post(
    "/retry/{execution_id}",
    dependencies=[_internal],
)
async def retry_execution(execution_id: str) -> dict[str, Any]:
    """Manually trigger a retry for a failed execution."""
    from app.modules.queue.tasks import retry_failed_execution
    task = retry_failed_execution.apply_async(
        kwargs={"execution_id": execution_id},
        queue="ai_heavy",
    )
    return {"task_id": task.id, "execution_id": execution_id}
