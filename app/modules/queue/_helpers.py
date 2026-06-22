"""Shared helpers for Celery task definitions.

Celery executes tasks synchronously, but all of this service's real work is async
(DB, HTTP, LLM). Each task therefore bridges the two worlds with
``asyncio.run(...)``. ``run_async`` captures that bridge once so individual task
bodies can be written as plain coroutines.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(fn: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Wrap an async function so a synchronous Celery task can call it.

    Apply *below* the ``@celery_app.task(...)`` decorator so Celery registers the
    synchronous wrapper::

        @celery_app.task(name="...")
        @run_async
        async def my_task(*, execution_id: str) -> dict:
            ...
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper
