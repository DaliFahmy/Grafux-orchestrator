from __future__ import annotations

import asyncio
import logging
from typing import Any

from celery import Task

from app.core.celery_app import get_celery_app
from app.modules.queue._helpers import run_async

celery_app = get_celery_app()
log = logging.getLogger("queue.tasks")


# ── Base task class with shared helpers ───────────────────────────────────────

class OrchestratorTask(Task):
    abstract = True

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        log.error("task_failed", extra={"task_id": task_id, "error": str(exc)})
        execution_id = kwargs.get("execution_id")
        if execution_id:
            asyncio.run(_mark_failed(execution_id, str(exc)))

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        log.warning("task_retrying", extra={"task_id": task_id, "error": str(exc)})


# ── Core execution task ────────────────────────────────────────────────────────

@celery_app.task(
    base=OrchestratorTask,
    bind=True,
    name="app.modules.queue.tasks.execute_workflow",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    queue="ai_heavy",
)
def execute_workflow(
    self: Task,
    *,
    execution_id: str,
    workflow_id: str | None = None,
    org_id: str,
    project_id: str,
    input_data: dict[str, Any] | None = None,
    graph_definition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Main Celery task: runs a LangGraph workflow to completion."""
    log.info("execute_workflow_started", extra={"execution_id": execution_id})
    return asyncio.run(
        _execute_workflow_async(
            task=self,
            execution_id=execution_id,
            workflow_id=workflow_id,
            org_id=org_id,
            project_id=project_id,
            input_data=input_data or {},
            graph_definition=graph_definition,
        )
    )


async def _execute_workflow_async(
    task: Task,
    execution_id: str,
    workflow_id: str | None,
    org_id: str,
    project_id: str,
    input_data: dict[str, Any],
    graph_definition: dict[str, Any] | None,
) -> dict[str, Any]:
    from datetime import datetime

    from app.core.database import get_db_session
    from app.modules.workflow.engine import WorkflowEngine
    from app.modules.workflow.service import WorkflowService

    async with get_db_session() as db:
        service = WorkflowService(db)

        # If workflow_id given, load graph definition from DB
        if workflow_id and not graph_definition:
            from app.modules.persistence.repositories import WorkflowRepository
            repo = WorkflowRepository(db)
            workflow = await repo.get_or_raise(workflow_id)
            graph_definition = workflow.graph_definition

        await service.mark_execution_running(execution_id)

    try:
        engine = WorkflowEngine()
        result = await engine.run(
            execution_id=execution_id,
            input_data=input_data,
            org_id=org_id,
            project_id=project_id,
            workflow_id=workflow_id,
            graph_definition=graph_definition,
        )

        async with get_db_session() as db:
            service = WorkflowService(db)
            if result.get("error"):
                await service.mark_execution_failed(execution_id, result["error"])
            elif result.get("cancelled"):
                from app.modules.persistence.repositories import ExecutionRepository
                repo = ExecutionRepository(db)
                await repo.update_status(
                    execution_id, "cancelled", finished_at=datetime.utcnow()
                )
                await db.commit()
            else:
                await service.mark_execution_complete(execution_id, result)

        log.info("execute_workflow_complete", extra={"execution_id": execution_id})
        return result

    except Exception as exc:
        log.exception("execute_workflow_error", extra={"execution_id": execution_id})
        try:
            task.retry(exc=exc, countdown=2 ** task.request.retries * 10)
        except task.MaxRetriesExceededError:
            await _mark_failed(execution_id, str(exc))
        raise


# ── Research task ──────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.modules.queue.tasks.run_research_pipeline",
    max_retries=2,
    default_retry_delay=20,
    queue="research",
)
@run_async
async def run_research_pipeline(
    *,
    execution_id: str,
    query: str,
    org_id: str,
) -> dict[str, Any]:
    from app.modules.research.pipeline import ResearchPipeline
    pipeline = ResearchPipeline()
    return await pipeline.run(query=query, execution_id=execution_id)


# ── Sandbox task ──────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.modules.queue.tasks.run_sandbox_execution",
    max_retries=1,
    queue="sandbox",
)
@run_async
async def run_sandbox_execution(
    *,
    execution_id: str,
    code: str,
    language: str = "python",
) -> dict[str, Any]:
    from app.modules.sandbox.e2b_client import E2BClient
    client = E2BClient()
    return await client.execute_code(code=code, execution_id=execution_id)


# ── Scheduled workflow task ────────────────────────────────────────────────────

@celery_app.task(
    name="app.modules.queue.tasks.run_scheduled_workflow",
    queue="scheduled",
)
@run_async
async def run_scheduled_workflow(*, scheduled_task_id: str) -> None:
    from datetime import datetime

    from sqlalchemy import select

    from app.core.database import get_db_session
    from app.modules.persistence.models import ScheduledTask

    workflow_id: str | None = None
    org_id: str = ""
    project_id: str = ""
    task_input: dict = {}

    async with get_db_session() as db:
        result = await db.execute(
            select(ScheduledTask).where(ScheduledTask.id == scheduled_task_id)
        )
        task_row = result.scalar_one_or_none()
        if not task_row or not task_row.is_active:
            return

        workflow_id = task_row.workflow_id
        org_id = task_row.org_id
        project_id = task_row.project_id
        task_input = task_row.input or {}

        task_row.last_run_at = datetime.utcnow()
        await db.commit()

    execute_workflow.apply_async(
        kwargs={
            "execution_id": f"sched-{scheduled_task_id}-{int(datetime.utcnow().timestamp())}",
            "workflow_id": workflow_id,
            "org_id": org_id,
            "project_id": project_id,
            "input_data": task_input,
        },
        queue="ai_heavy",
    )


# ── Maintenance task ───────────────────────────────────────────────────────────

@celery_app.task(
    name="app.modules.queue.tasks.cleanup_expired_executions",
    queue="default",
)
@run_async
async def cleanup_expired_executions() -> int:
    from datetime import datetime, timedelta

    from sqlalchemy import update

    from app.core.database import get_db_session
    from app.modules.persistence.models import Execution

    cutoff = datetime.utcnow() - timedelta(days=30)
    async with get_db_session() as db:
        result = await db.execute(
            update(Execution)
            .where(Execution.status == "pending", Execution.created_at < cutoff)
            .values(status="failed", error_message="Expired: never started")
        )
        await db.commit()
        return result.rowcount


# ── Retry failed execution ─────────────────────────────────────────────────────

@celery_app.task(
    name="app.modules.queue.tasks.retry_failed_execution",
    queue="ai_heavy",
)
@run_async
async def retry_failed_execution(*, execution_id: str) -> dict[str, Any]:
    from app.core.database import get_db_session
    from app.modules.persistence.repositories import ExecutionRepository

    async with get_db_session() as db:
        repo = ExecutionRepository(db)
        execution = await repo.get_or_raise(execution_id)
        await repo.increment_retry(execution_id)
        await repo.update_status(execution_id, "pending")
        await db.commit()

    execute_workflow.apply_async(
        kwargs={
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "org_id": execution.org_id,
            "project_id": execution.project_id,
            "input_data": execution.input,
        },
        queue="ai_heavy",
    )
    return {"execution_id": execution_id, "status": "retrying"}


# ── Helper ─────────────────────────────────────────────────────────────────────

async def _mark_failed(execution_id: str, error: str) -> None:
    try:
        from app.core.database import get_db_session
        from app.modules.workflow.service import WorkflowService
        async with get_db_session() as db:
            await WorkflowService(db).mark_execution_failed(execution_id, error)
    except Exception as exc:
        log.error("mark_failed_error", extra={"execution_id": execution_id, "error": str(exc)})
