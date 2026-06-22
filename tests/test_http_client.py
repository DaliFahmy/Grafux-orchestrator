from __future__ import annotations

import pytest
from app.core import http_client


@pytest.mark.asyncio
async def test_same_client_reused_within_loop():
    """get_http_client returns the same pooled client on repeated calls in one loop."""
    await http_client.aclose_http_client()  # start clean
    c1 = http_client.get_http_client()
    c2 = http_client.get_http_client()
    assert c1 is c2
    assert not c1.is_closed


@pytest.mark.asyncio
async def test_close_releases_and_recreates():
    """After aclose, the next get_http_client builds a fresh client."""
    c1 = http_client.get_http_client()
    await http_client.aclose_http_client()
    assert c1.is_closed
    c2 = http_client.get_http_client()
    assert c2 is not c1
    assert not c2.is_closed
    await http_client.aclose_http_client()


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    """Calling aclose twice (or with nothing open) does not raise."""
    await http_client.aclose_http_client()
    await http_client.aclose_http_client()
