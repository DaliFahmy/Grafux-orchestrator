from __future__ import annotations

import asyncio

import redis.asyncio as aioredis

from app.config import get_settings
from app.core.logging import get_logger
from app.modules.streaming.websocket import connection_manager

log = get_logger("streaming.broadcaster")

_broadcaster_task: asyncio.Task | None = None


async def _redis_listener() -> None:
    """Long-running coroutine that subscribes to execution:* channels and
    fans out incoming messages to connected WebSocket clients."""
    settings = get_settings()
    while True:
        try:
            async with aioredis.from_url(
                settings.redis_url, decode_responses=True
            ) as redis:
                pubsub = redis.pubsub()
                await pubsub.psubscribe("execution:*")
                log.info("broadcaster_subscribed", pattern="execution:*")
                async for message in pubsub.listen():
                    if message["type"] not in ("pmessage", "message"):
                        continue
                    raw: str = message.get("data", "")
                    if not raw:
                        continue
                    # Extract execution_id from channel name: "execution:{id}"
                    channel: str = message.get("channel") or message.get("pattern", "")
                    parts = channel.split(":", 1)
                    if len(parts) == 2:
                        execution_id = parts[1]
                        await connection_manager.send_to_execution(execution_id, raw)
        except asyncio.CancelledError:
            log.info("broadcaster_stopped")
            break
        except Exception as exc:
            log.error("broadcaster_error", error=str(exc))
            await asyncio.sleep(2)  # brief backoff before reconnect


async def start_broadcaster() -> None:
    """Start the Redis → WebSocket broadcaster as a background task."""
    global _broadcaster_task
    if _broadcaster_task is None or _broadcaster_task.done():
        _broadcaster_task = asyncio.create_task(_redis_listener())
        log.info("broadcaster_started")


async def stop_broadcaster() -> None:
    global _broadcaster_task
    if _broadcaster_task and not _broadcaster_task.done():
        _broadcaster_task.cancel()
        try:
            await _broadcaster_task
        except asyncio.CancelledError:
            pass
        _broadcaster_task = None
        log.info("broadcaster_stopped")
