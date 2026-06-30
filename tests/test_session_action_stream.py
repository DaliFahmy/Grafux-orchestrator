from __future__ import annotations

import asyncio
import json

import pytest
from app.modules.session import enrichment
from app.modules.session.router import _OrchestratorSession


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _session() -> tuple[_OrchestratorSession, _FakeWS]:
    ws = _FakeWS()
    return _OrchestratorSession(ws, "sid", "uid", "pid"), ws


def test_stamp_actions_marks_enrichable_pending_and_ids():
    actions = [
        {"type": "create_block", "block_type": "topics", "block_name": "t"},
        {"type": "create_block", "block_type": "nope", "block_name": "n"},
        {"type": "set_port", "block_type": "tools", "block_name": "s"},
    ]
    _OrchestratorSession._stamp_actions(actions)
    assert actions[0]["pending"] is True and actions[0]["action_id"]
    # Unknown type still gets an id (stable correlation) but is NOT pending.
    assert "pending" not in actions[1] and actions[1]["action_id"]
    # Non-create actions are left untouched.
    assert "action_id" not in actions[2]


def test_extract_patch_only_generated_keys():
    action = {
        "block_name": "x", "action_id": "a1", "pending": True,
        "output_ports": [{"port_name": "p"}], "code": "y = 1", "description": "d",
    }
    patch = _OrchestratorSession._extract_patch(action)
    assert patch == {"output_ports": [{"port_name": "p"}], "code": "y = 1"}


@pytest.mark.asyncio
async def test_enrich_and_stream_emits_action_enriched(monkeypatch):
    sess, ws = _session()

    async def fake(action, session_id):
        action["output_ports"] = [{"port_name": "flask", "port_content": "web framework"}]
        return ("ok", "")

    monkeypatch.setitem(enrichment._ENRICHERS, "topics", fake)
    actions = [{"type": "create_block", "block_type": "topics", "block_name": "frameworks"}]
    _OrchestratorSession._stamp_actions(actions)

    await sess._enrich_and_stream(actions)

    msg = ws.sent[-1]
    assert msg["type"] == "action_enriched"
    assert msg["block_name"] == "frameworks"
    assert msg["action_id"] == actions[0]["action_id"]
    assert msg["enrichment_status"] == "ok"
    assert msg["patch"]["output_ports"][0]["port_name"] == "flask"


@pytest.mark.asyncio
async def test_enrich_and_stream_reports_failure(monkeypatch):
    sess, ws = _session()

    async def fake(action, session_id):
        return ("failed", "AI not configured")

    monkeypatch.setitem(enrichment._ENRICHERS, "code", fake)
    actions = [{"type": "create_block", "block_type": "code", "block_name": "c"}]
    _OrchestratorSession._stamp_actions(actions)

    await sess._enrich_and_stream(actions)

    msg = ws.sent[-1]
    assert msg["type"] == "action_enriched"
    assert msg["enrichment_status"] == "failed"
    assert msg["enrichment_error"] == "AI not configured"


@pytest.mark.asyncio
async def test_turn_does_not_block_on_enrichment(monkeypatch):
    """turn_complete is sent before a slow enricher finishes; the patch arrives after."""
    sess, ws = _session()
    gate = asyncio.Event()

    async def slow(action, session_id):
        await gate.wait()  # never completes until we release it
        action["output_ports"] = [{"port_name": "p"}]
        return ("ok", "")

    monkeypatch.setitem(enrichment._ENRICHERS, "topics", slow)
    actions = [{"type": "create_block", "block_type": "topics", "block_name": "t"}]
    _OrchestratorSession._stamp_actions(actions)

    # Spawn background enrichment (as the turn does) and confirm it doesn't block.
    sess._spawn_enrichment(actions)
    await asyncio.sleep(0)  # let the task start and block on the gate

    # No action_enriched yet — the slow enricher is still gated.
    assert not any(m["type"] == "action_enriched" for m in ws.sent)
    assert len(sess._enrich_tasks) == 1

    # Release the enricher; the patch is delivered.
    gate.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if any(m["type"] == "action_enriched" for m in ws.sent):
            break
    assert any(m["type"] == "action_enriched" for m in ws.sent)


@pytest.mark.asyncio
async def test_cancel_enrichment_stops_inflight_tasks(monkeypatch):
    sess, ws = _session()
    started = asyncio.Event()

    async def hang(action, session_id):
        started.set()
        await asyncio.sleep(100)
        return ("ok", "")

    monkeypatch.setitem(enrichment._ENRICHERS, "topics", hang)
    actions = [{"type": "create_block", "block_type": "topics", "block_name": "t"}]
    _OrchestratorSession._stamp_actions(actions)
    sess._spawn_enrichment(actions)
    await started.wait()

    await sess._cancel_enrichment()
    assert sess._enrich_tasks == set()
