"""Shared, connection-pooled ``httpx.AsyncClient`` keyed by event loop.

Creating an ``httpx.AsyncClient`` per request throws away the TCP/TLS connection
pool, so every outbound call to an internal service (MCP, devices) or external
API pays a fresh handshake. This module hands out a reused client instead.

An ``httpx.AsyncClient`` is bound to the event loop it was created on. The web
process runs a single long-lived loop, so it gets exactly one client reused
across every request. Celery, by contrast, runs ``asyncio.run(...)`` per task —
a fresh loop each time — so we key the client by the *running loop*: one client
per loop, reused across all calls within that loop's lifetime, and transparently
recreated for the next loop. This keeps pooling wins in both worlds without ever
reusing a client bound to a closed loop.

Usage::

    from app.core.http_client import get_http_client

    resp = await get_http_client().post(url, json=payload, timeout=30.0)

Per-call ``timeout=`` overrides the client default, so callers keep full control
of individual request budgets. The web process closes its client in the FastAPI
lifespan shutdown (see ``app/main.py``); worker clients are released when their
per-task loop closes.
"""

from __future__ import annotations

import asyncio

import httpx

from app.config import get_settings

# Maps an event loop to the client created on it. Keyed by id() to avoid holding
# a strong reference to the loop; entries are pruned when their loop is gone.
_clients: dict[int, tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = {}


def _new_client() -> httpx.AsyncClient:
    settings = get_settings()
    return httpx.AsyncClient(
        timeout=settings.http_default_timeout,
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
        ),
    )


def _prune_closed() -> None:
    """Drop entries whose loop has been closed (e.g. finished Celery tasks)."""
    for key, (loop, _client) in list(_clients.items()):
        if loop.is_closed():
            _clients.pop(key, None)


def get_http_client() -> httpx.AsyncClient:
    """Return a pooled async HTTP client bound to the currently running loop."""
    loop = asyncio.get_running_loop()
    entry = _clients.get(id(loop))
    if entry is not None and entry[0] is loop and not entry[1].is_closed:
        return entry[1]
    _prune_closed()
    client = _new_client()
    _clients[id(loop)] = (loop, client)
    return client


async def aclose_http_client() -> None:
    """Close the current loop's client and release its pooled connections.

    Idempotent. Safe to call in FastAPI lifespan shutdown; worker loops that exit
    via ``asyncio.run`` drop their entry implicitly when the loop is replaced.
    """
    loop = asyncio.get_running_loop()
    entry = _clients.pop(id(loop), None)
    if entry is not None and not entry[1].is_closed:
        await entry[1].aclose()
