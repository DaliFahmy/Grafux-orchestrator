from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("workflow.checkpoints")


@asynccontextmanager
async def get_postgres_checkpointer():
    """Async context manager that yields a LangGraph AsyncPostgresSaver.

    Uses the psycopg v3 URL (raw ``postgresql://`` scheme, not ``+asyncpg``).
    The checkpointer creates its own connection pool internally.
    """
    settings = get_settings()
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(settings.psycopg_url) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    except ImportError:
        log.warning(
            "langgraph_checkpoint_postgres_unavailable",
            hint="Install langgraph-checkpoint-postgres",
        )
        from langgraph.checkpoint.memory import MemorySaver
        yield MemorySaver()


def get_memory_checkpointer():
    """Synchronous in-memory checkpointer for testing / offline use."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
