"""Celery worker entry point.

Start a worker with:
    celery -A worker.celery_worker worker -Q ai_heavy -c 2 --loglevel=info

Start the beat scheduler with:
    celery -A worker.celery_worker beat --loglevel=info
"""
from __future__ import annotations

from app.core.celery_app import get_celery_app
from app.core.logging import configure_logging
from app.modules.queue.scheduler import configure_beat_schedule

configure_logging()
configure_beat_schedule()

# Expose the Celery app for the `-A` flag
app = get_celery_app()

# Ensure all tasks are registered
import app.modules.queue.tasks  # noqa: F401, E402
