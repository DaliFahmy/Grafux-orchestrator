"""Reusable resilience policies for outbound calls.

Transient network failures (connection resets, read/connect timeouts) are
expected when talking to internal services and third-party APIs. ``retry_transient``
centralizes the "retry a few times with exponential backoff, then give up and
re-raise" policy so every client doesn't hand-roll its own tenacity config.

Only ``httpx.TransportError`` (which includes the timeout exceptions) is retried —
HTTP error *responses* (4xx/5xx via ``raise_for_status``) are NOT retried here,
since a 4xx won't fix itself and retrying a non-idempotent write could double it.
Apply this only to idempotent operations (reads, lookups, tool invocations that
are safe to repeat).
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Default policy: up to 3 attempts, exponential backoff 1s→10s, re-raise the
# final error so callers' own error handling still fires.
DEFAULT_MAX_ATTEMPTS = 3


def retry_transient(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    *,
    wait_min: float = 1.0,
    wait_max: float = 10.0,
    multiplier: float = 1.0,
):
    """Decorator: retry an async call on transient HTTP transport errors.

    Usage::

        @retry_transient()
        async def fetch(...): ...

    Backoff is exponential between ``wait_min`` and ``wait_max`` seconds. Safe only
    for idempotent operations — see the module docstring.
    """
    return retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=wait_min, max=wait_max),
        reraise=True,
    )
