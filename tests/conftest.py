from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio

# Set test environment before importing app modules
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DATABASE_SYNC_URL", "sqlite:///./test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("INTERNAL_SERVICE_SECRET", "test_secret")


@pytest.fixture(autouse=True)
def _reset_llm_module_state():
    """Clear app.core.llm's module globals around every test.

    The Anthropic auth latch and the per-loop SDK client pools are process-wide
    by design (see the notes in that module). Left alone they make test order
    load-bearing: one test tripping the latch would silently route later tests
    to OpenAI.
    """
    from app.core import llm

    def _clear() -> None:
        llm.reset_anthropic_fallback()
        llm._anthropic_clients.clear()
        llm._openai_clients.clear()
        llm._gemini_clients.clear()

    _clear()
    yield
    _clear()


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session():
    """In-memory SQLite session for unit tests."""
    import app.modules.persistence.models  # noqa
    from app.core.database import Base
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()
