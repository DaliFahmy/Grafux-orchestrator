from __future__ import annotations

from celery import Celery

from app.config import get_settings

_celery_app: Celery | None = None


def create_celery_app() -> Celery:
    settings = get_settings()

    app = Celery(
        "grafux_orchestrator",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=[
            "app.modules.queue.tasks",
        ],
    )

    app.conf.update(
        task_serializer=settings.celery_task_serializer,
        result_serializer="json",
        accept_content=["json"],
        result_expires=settings.celery_result_expires,
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        worker_send_task_events=True,
        task_send_sent_event=True,
        # Retry policy defaults
        task_max_retries=3,
        task_default_retry_delay=60,
        # Routing
        task_routes={
            "app.modules.queue.tasks.execute_workflow": {"queue": "ai_heavy"},
            "app.modules.queue.tasks.run_research_pipeline": {"queue": "research"},
            "app.modules.queue.tasks.run_sandbox_execution": {"queue": "sandbox"},
            "app.modules.queue.tasks.run_scheduled_workflow": {"queue": "scheduled"},
            "app.modules.queue.tasks.cleanup_expired_executions": {"queue": "default"},
            "app.modules.queue.tasks.retry_failed_execution": {"queue": "ai_heavy"},
        },
        # Queue definitions
        task_queues={
            "default": {"exchange": "default", "routing_key": "default"},
            "ai_heavy": {"exchange": "ai_heavy", "routing_key": "ai_heavy"},
            "research": {"exchange": "research", "routing_key": "research"},
            "sandbox": {"exchange": "sandbox", "routing_key": "sandbox"},
            "scheduled": {"exchange": "scheduled", "routing_key": "scheduled"},
        },
        task_default_queue="default",
        # Beat schedule populated in scheduler.py via app.conf.beat_schedule
    )

    return app


def get_celery_app() -> Celery:
    global _celery_app
    if _celery_app is None:
        _celery_app = create_celery_app()
    return _celery_app
