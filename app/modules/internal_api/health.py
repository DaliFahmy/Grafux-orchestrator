from __future__ import annotations

import asyncio

from app.config import get_settings
from app.core.database import get_engine
from app.core.redis import get_redis_client
from app.modules.internal_api.schemas import HealthResponse


async def get_health() -> HealthResponse:
    settings = get_settings()

    db_status = "ok"
    redis_status = "ok"
    celery_status = "ok"

    # Check database
    try:
        async with get_engine().connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_status = "unavailable"

    # Check Redis
    try:
        redis = get_redis_client()
        await redis.ping()
    except Exception:
        redis_status = "unavailable"

    # Check Celery (inspect ping with short timeout)
    try:
        from app.core.celery_app import get_celery_app
        celery = get_celery_app()
        inspector = celery.control.inspect(timeout=1.0)
        ping_result = await asyncio.get_event_loop().run_in_executor(
            None, inspector.ping
        )
        if not ping_result:
            celery_status = "no_workers"
    except Exception:
        celery_status = "unavailable"

    overall = (
        "ok"
        if all(s == "ok" for s in [db_status, redis_status])
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        service=settings.service_name,
        version="2.0.0",
        database=db_status,
        redis=redis_status,
        celery=celery_status,
        environment=settings.environment,
    )
