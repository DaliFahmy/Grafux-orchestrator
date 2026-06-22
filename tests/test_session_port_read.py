from __future__ import annotations

import asyncio
import json

import pytest
from app.modules.session.router import (
    _OrchestratorSession,
    _render_canvas_context,
    _render_port,
)

# ── Canvas rendering ─────────────────────────────────────────────────────────


def test_render_port_empty_value():
    line = _render_port({"name": "result", "is_output": True, "value": ""})
    assert "[out] result: (empty)" in line


def test_render_port_truncation_note():
    line = _render_port({
        "name": "response",
        "is_output": True,
        "value": "x" * 600,
        "full_length": 4200,
        "truncated": True,
    })
    assert "truncated: showing 600 of 4200 chars" in line
    assert "read_port_value" in line


def test_render_canvas_active_first_and_budget():
    canvas = {
        "blocks": [
            {"name": "A", "type": "tools", "status": "Success",
             "ports": [{"name": "out1", "is_output": True, "value": "av"}]},
            {"name": "B", "type": "topics", "status": "Idle",
             "ports": [{"name": "in1", "is_output": False, "value": "bv"}]},
        ]
    }
    active = [{"name": "B", "type": "topics", "status": "Idle",
               "ports": [{"name": "in1", "is_output": False, "value": "bv"}]}]
    out = _render_canvas_context(canvas, active)
    # Active block B is rendered (tagged) before non-active A.
    assert "[ACTIVE]" in out
    assert out.index('"B"') < out.index('"A"')


def test_render_canvas_overflow_noted_not_dropped_silently():
    blocks = [
        {"name": f"B{i}", "type": "tools", "status": "Idle",
         "ports": [{"name": "p", "is_output": True, "value": "z" * 500}]}
        for i in range(50)
    ]
    out = _render_canvas_context({"blocks": blocks}, [], budget=2000)
    assert "more block(s) not shown" in out


# ── On-demand read round-trip ────────────────────────────────────────────────


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _session() -> tuple[_OrchestratorSession, _FakeWS]:
    ws = _FakeWS()
    return _OrchestratorSession(ws, "sid", "uid", "pid"), ws


@pytest.mark.asyncio
async def test_read_roundtrip_returns_client_content():
    sess, ws = _session()
    task = asyncio.create_task(
        sess._resolve_read_tool(
            {"target_block": "A", "direction": "output", "port_name": "response"}
        )
    )
    # Let the request register its future and emit the read_port frame.
    for _ in range(5):
        await asyncio.sleep(0)
        if ws.sent:
            break
    read_req = ws.sent[-1]
    assert read_req["type"] == "read_port"
    assert read_req["target_block"] == "A"
    assert read_req["port_name"] == "response"

    # Simulate the client replying with the full port-file content.
    await sess._dispatch_text(json.dumps({
        "type": "port_content",
        "request_id": read_req["request_id"],
        "found": True,
        "content": "HELLO WORLD",
    }))
    assert await task == "HELLO WORLD"


@pytest.mark.asyncio
async def test_read_roundtrip_not_found_message():
    sess, ws = _session()
    task = asyncio.create_task(
        sess._resolve_read_tool(
            {"target_block": "Ghost", "direction": "input", "port_name": "x"}
        )
    )
    for _ in range(5):
        await asyncio.sleep(0)
        if ws.sent:
            break
    req_id = ws.sent[-1]["request_id"]
    await sess._dispatch_text(json.dumps({
        "type": "port_content",
        "request_id": req_id,
        "found": False,
        "content": "",
    }))
    result = await task
    assert "Could not read" in result


@pytest.mark.asyncio
async def test_read_timeout_is_graceful(monkeypatch):
    sess, ws = _session()
    # No client reply ever arrives → wait_for times out → found=False.
    result = await sess._request_port_content("A", "output", "p", timeout=0.05)
    assert result == {"found": False, "content": ""}


@pytest.mark.asyncio
async def test_resolve_read_tool_requires_fields():
    sess, _ = _session()
    msg = await sess._resolve_read_tool({"direction": "output"})
    assert "required" in msg
