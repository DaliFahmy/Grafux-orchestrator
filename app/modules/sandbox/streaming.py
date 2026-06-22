"""Sandbox output streaming helpers.

When E2B stdout/stderr is produced, events are pushed to the execution's
Redis channel so connected WebSocket clients receive them in real-time.
This module provides helpers used by the E2B client and Celery tasks.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.modules.streaming.event_bus import emit
from app.shared.events import SANDBOX_STDERR, SANDBOX_STDOUT

log = get_logger("sandbox.streaming")


async def stream_stdout(execution_id: str, text: str) -> None:
    if execution_id and text:
        await emit(execution_id, SANDBOX_STDOUT, {"text": text})


async def stream_stderr(execution_id: str, text: str) -> None:
    if execution_id and text:
        await emit(execution_id, SANDBOX_STDERR, {"text": text})
