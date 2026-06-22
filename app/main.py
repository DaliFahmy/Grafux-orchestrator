from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.database import create_all_tables, dispose_engine
from app.core.exceptions import (
    OrchestratorError,
    orchestrator_error_handler,
    validation_exception_handler,
)
from app.core.http_client import aclose_http_client
from app.core.logging import configure_logging, get_logger
from app.core.redis import close_redis, get_redis_client
from app.modules.streaming.broadcaster import start_broadcaster, stop_broadcaster
from app.modules.workflow.templates import TemplateService

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging()
    log.info("orchestrator_starting", environment=settings.environment)

    # Initialise database (dev only — production uses Alembic migrations)
    if settings.environment == "development":
        await create_all_tables()

    # Seed default workflow templates
    from app.core.database import get_db_session
    async with get_db_session() as db:
        await TemplateService(db).seed_defaults()

    # Start Redis → WebSocket broadcaster
    await start_broadcaster()

    log.info("orchestrator_ready", port=settings.port)
    yield

    # Graceful shutdown
    await stop_broadcaster()
    await close_redis()
    await aclose_http_client()
    await dispose_engine()
    log.info("orchestrator_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Grafux Orchestrator",
        description=(
            "Production-grade AI execution runtime: workflow engine, "
            "agent orchestration, LangGraph, Celery, MCP, E2B, research pipelines."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.middleware("http")(_audit_log_middleware)
    app.middleware("http")(_rate_limit_middleware)

    # ── Exception handlers ─────────────────────────────────────────────────────
    app.add_exception_handler(OrchestratorError, orchestrator_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]

    # ── Routers ────────────────────────────────────────────────────────────────
    from app.modules.agent.router import router as agent_router
    from app.modules.blocks.router import router as blocks_router
    from app.modules.devices.router import router as devices_router
    from app.modules.internal_api.router import router as internal_router
    from app.modules.mcp.router import router as mcp_router
    from app.modules.queue.router import router as queue_router
    from app.modules.research.router import router as research_router
    from app.modules.sandbox.router import router as sandbox_router
    from app.modules.session.router import router as session_router
    from app.modules.workflow.router import executions_router
    from app.modules.workflow.router import router as workflow_router

    app.include_router(internal_router)
    app.include_router(workflow_router)
    app.include_router(executions_router)
    app.include_router(mcp_router)
    app.include_router(research_router)
    app.include_router(sandbox_router)
    app.include_router(devices_router)
    app.include_router(agent_router)
    app.include_router(queue_router)
    app.include_router(session_router)
    app.include_router(blocks_router)

    # ── Root health check (unauthenticated — for load balancers) ───────────────
    @app.get("/health", include_in_schema=False)
    async def root_health() -> dict:
        return {"status": "ok", "service": "grafux-orchestrator", "version": "2.0.0"}

    return app


# ── Middleware implementations ─────────────────────────────────────────────────

async def _audit_log_middleware(request: Request, call_next):
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        method=request.method,
        path=request.url.path,
    )
    response = await call_next(request)
    if not request.url.path.startswith("/health"):
        log.info("http_request", status=response.status_code)

    return response


async def _rate_limit_middleware(request: Request, call_next):
    # Skip rate-limiting for health checks and internal routes
    if request.url.path.startswith(("/health", "/internal")):
        return await call_next(request)

    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"

    try:
        redis = get_redis_client()
        key = f"rate_limit:{client_ip}"
        current = await redis.incr(key)
        if current == 1:
            await redis.expire(key, settings.rate_limit_window)
        if current > settings.rate_limit_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "rate_limit_exceeded", "message": "Too many requests"},
            )
    except Exception as exc:
        # Redis unavailable — fail open (allow request) but record why.
        log.debug("rate_limit_unavailable", error=str(exc))

    return await call_next(request)


async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import os

    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", settings.port)),
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
