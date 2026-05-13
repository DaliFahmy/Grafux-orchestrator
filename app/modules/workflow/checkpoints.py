from __future__ import annotations

from contextlib import asynccontextmanager

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("workflow.checkpoints")


@asynccontextmanager
async def get_postgres_checkpointer():
    """Async context manager that yields a LangGraph AsyncPostgresSaver.

    langgraph-checkpoint-postgres 3.x uses psycopg v3 under the hood.
    The connection string must be the raw ``postgresql://`` scheme
    (not the SQLAlchemy ``+asyncpg`` variant).
    """
    settings = get_settings()
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.psycopg_url) as checkpointer:
            # 3.x: setup() is handled internally by the context manager
            try:
                await checkpointer.setup()
            except Exception:
                # setup() may not exist in all 3.x point releases; safe to ignore
                pass
            yield checkpointer

    except ImportError:
        log.warning(
            "langgraph_checkpoint_postgres_unavailable",
            hint="Install langgraph-checkpoint-postgres>=3.0.0",
        )
        from langgraph.checkpoint.memory import MemorySaver
        yield MemorySaver()

    except Exception as exc:
        log.error("postgres_checkpointer_failed", error=str(exc))
        from langgraph.checkpoint.memory import MemorySaver
        log.warning("falling_back_to_memory_checkpointer")
        yield MemorySaver()


def get_memory_checkpointer():
    """In-memory checkpointer for testing / offline use."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
