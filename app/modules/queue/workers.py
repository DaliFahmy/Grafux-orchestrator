from __future__ import annotations

"""Worker pool configuration helpers.

Each queue maps to a dedicated pool for resource isolation:
- default: 4 workers  — lightweight tasks
- ai_heavy: 2 workers — LangGraph + LLM calls (CPU/IO heavy)
- research: 2 workers — Tavily / Firecrawl HTTP calls
- sandbox: 2 workers — E2B SDK calls
- scheduled: 1 worker — periodic/beat tasks
"""

QUEUE_CONCURRENCY: dict[str, int] = {
    "default": 4,
    "ai_heavy": 2,
    "research": 2,
    "sandbox": 2,
    "scheduled": 1,
}

# Worker startup command templates (used in Docker / Render)
WORKER_COMMANDS: dict[str, str] = {
    "default": "celery -A worker.celery_worker worker -Q default -c 4 --loglevel=info",
    "ai_heavy": "celery -A worker.celery_worker worker -Q ai_heavy -c 2 --loglevel=info",
    "research": "celery -A worker.celery_worker worker -Q research -c 2 --loglevel=info",
    "sandbox": "celery -A worker.celery_worker worker -Q sandbox -c 2 --loglevel=info",
    "beat": "celery -A worker.celery_worker beat --loglevel=info",
}
