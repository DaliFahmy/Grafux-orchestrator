from __future__ import annotations

import httpx
import pytest
from app.core.resilience import retry_transient

# Zero-backoff variant so the retry tests don't actually sleep.
_no_wait = {"wait_min": 0, "wait_max": 0, "multiplier": 0}


@pytest.mark.asyncio
async def test_passes_through_success_without_retry():
    calls = 0

    @retry_transient(**_no_wait)
    async def ok():
        nonlocal calls
        calls += 1
        return "done"

    assert await ok() == "done"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_transient_then_reraises():
    calls = 0

    @retry_transient(max_attempts=3, **_no_wait)
    async def always_transient():
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("boom")

    with pytest.raises(httpx.ConnectError):
        await always_transient()
    assert calls == 3  # exhausted all attempts, then re-raised


@pytest.mark.asyncio
async def test_recovers_after_a_transient_failure():
    calls = 0

    @retry_transient(max_attempts=5, **_no_wait)
    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadError("transient")
        return "recovered"

    assert await flaky() == "recovered"
    assert calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_transport_errors():
    calls = 0

    @retry_transient(max_attempts=3, **_no_wait)
    async def bad_request():
        nonlocal calls
        calls += 1
        raise ValueError("not a transport error")

    with pytest.raises(ValueError):
        await bad_request()
    assert calls == 1  # ValueError is not transient → no retry


@pytest.mark.asyncio
async def test_timeout_is_treated_as_transient():
    calls = 0

    @retry_transient(max_attempts=2, **_no_wait)
    async def slow():
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(httpx.ReadTimeout):
        await slow()
    assert calls == 2  # TimeoutException subclasses TransportError → retried
