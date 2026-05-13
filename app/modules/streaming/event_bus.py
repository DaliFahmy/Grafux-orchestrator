from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.core.redis import get_redis_client, publish
from app.modules.streaming.schemas import ExecutionEvent
from app.shared.events import execution_channel

log = get_logger("streaming.event_bus")


async def emit(
    execution_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Publish an execution event to the Redis pub/sub channel."""
    event = ExecutionEvent(
        execution_id=execution_id,
        event_type=event_type,
        payload=payload or {},
    )
    channel = execution_channel(execution_id)
    await publish(channel, event.to_json())
    log.debug("event_emitted", execution_id=execution_id, event_type=event_type)


async def emit_log(
    execution_id: str,
    message: str,
    level: str = "info",
    step_id: str | None = None,
    **extra: Any,
) -> None:
    """Publish a log message as an execution.log event."""
    from app.shared.events import EXECUTION_LOG
    payload = {"message": message, "level": level, **extra}
    if step_id:
        payload["step_id"] = step_id
    await emit(execution_id, EXECUTION_LOG, payload)


async def emit_agent_thought(
    execution_id: str,
    agent_id: str,
    thought: str,
) -> None:
    from app.shared.events import AGENT_THOUGHT
    await emit(execution_id, AGENT_THOUGHT, {
        "agent_id": agent_id,
        "thought": thought,
    })


async def emit_step(
    execution_id: str,
    step_id: str,
    node_name: str,
    step_status: str,
    **kwargs: Any,
) -> None:
    from app.shared.events import EXECUTION_STEP
    await emit(execution_id, EXECUTION_STEP, {
        "step_id": step_id,
        "node_name": node_name,
        "status": step_status,
        **kwargs,
    })
