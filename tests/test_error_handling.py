from __future__ import annotations

import json

import pytest

from app.core.exceptions import (
    ExecutionNotFoundError,
    orchestrator_error_handler,
    validation_exception_handler,
)


@pytest.mark.asyncio
async def test_validation_handler_uses_house_schema():
    class FakeValidationError(Exception):
        def errors(self):
            return [{"loc": ["body", "x"], "msg": "field required"}]

    resp = await validation_exception_handler(None, FakeValidationError())
    assert resp.status_code == 422
    body = json.loads(resp.body)
    assert body["error"] == "validation_error"
    assert body["message"]
    assert isinstance(body["details"], list) and body["details"]


@pytest.mark.asyncio
async def test_orchestrator_error_handler_maps_status_and_schema():
    resp = await orchestrator_error_handler(None, ExecutionNotFoundError("abc"))
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["error"] == "execution_not_found"
    assert "abc" in body["message"]


@pytest.mark.asyncio
async def test_send_swallows_client_disconnect():
    from fastapi import WebSocketDisconnect

    from app.modules.session.router import _send

    class DisconnectedWS:
        async def send_text(self, text: str) -> None:
            raise WebSocketDisconnect(code=1000)

    # Must not raise — a turn keeps completing even if the client vanished.
    await _send(DisconnectedWS(), {"type": "text_chunk", "text": "hi"})


@pytest.mark.asyncio
async def test_send_swallows_generic_socket_error():
    from app.modules.session.router import _send

    class BrokenWS:
        async def send_text(self, text: str) -> None:
            raise RuntimeError("socket closed")

    await _send(BrokenWS(), {"type": "pong"})
