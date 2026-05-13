from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("core.redis")

_pool: aioredis.ConnectionPool | None = None
_client: Redis | None = None


def get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
    return _pool


def get_redis_client() -> Redis:
    """Return a shared async Redis client (re-uses connection pool)."""
    global _client
    if _client is None:
        _client = aioredis.Redis(connection_pool=get_pool())
    return _client


@asynccontextmanager
async def get_pubsub() -> AsyncGenerator[aioredis.client.PubSub, None]:
    """Context manager yielding a fresh PubSub object."""
    client = get_redis_client()
    pubsub = client.pubsub()
    try:
        yield pubsub
    finally:
        await pubsub.close()


async def publish(channel: str, message: str) -> None:
    """Publish a message to a Redis channel."""
    client = get_redis_client()
    await client.publish(channel, message)


async def close_redis() -> None:
    global _client, _pool
    if _client is not None:
        await _client.aclose()
        _client = None
    if _pool is not None:
        await _pool.aclose()
        _pool = None
    log.info("redis_connection_closed")


# ── Dependency ────────────────────────────────────────────────────────────────

async def get_redis() -> AsyncGenerator[Redis, None]:
    """FastAPI dependency that yields the shared Redis client."""
    yield get_redis_client()
