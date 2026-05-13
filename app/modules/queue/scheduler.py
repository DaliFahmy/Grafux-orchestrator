from __future__ import annotations

from celery.schedules import crontab

from app.core.celery_app import get_celery_app


def configure_beat_schedule() -> None:
    """Register Celery Beat periodic tasks."""
    celery_app = get_celery_app()
    celery_app.conf.beat_schedule = {
        "cleanup-expired-executions-daily": {
            "task": "app.modules.queue.tasks.cleanup_expired_executions",
            "schedule": crontab(hour=2, minute=0),  # 02:00 UTC daily
            "options": {"queue": "default"},
        },
        "check-scheduled-workflows-minutely": {
            "task": "app.modules.queue.tasks.run_scheduled_workflow",
            "schedule": crontab(minute="*/1"),
            "options": {"queue": "scheduled"},
            "kwargs": {"scheduled_task_id": "__poll__"},  # handled specially in task
        },
    }
