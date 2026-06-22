from __future__ import annotations

import asyncio

import pytest
from app.core.constants import (
    NODE_DEVICE,
    NODE_MCP,
    NODE_RESEARCH,
    NODE_SANDBOX,
    route_tool_to_node,
)
from app.modules.session import enrichment


def test_route_tool_to_node_mapping():
    assert route_tool_to_node("mcp_search") == NODE_MCP
    assert route_tool_to_node("grafux_thing") == NODE_MCP
    assert route_tool_to_node("research_topic") == NODE_RESEARCH
    assert route_tool_to_node("web_search") == NODE_RESEARCH
    assert route_tool_to_node("sandbox_run") == NODE_SANDBOX
    assert route_tool_to_node("execute_code") == NODE_SANDBOX
    assert route_tool_to_node("device_toggle") == NODE_DEVICE
    assert route_tool_to_node("anything_else") == NODE_MCP  # default fall-through
    assert route_tool_to_node("") == NODE_MCP


@pytest.mark.asyncio
async def test_enrich_actions_runs_concurrently(monkeypatch):
    """Independent create_block actions are enriched in parallel, not serially."""
    events: list[tuple[str, str]] = []

    async def fake(action, session_id):
        events.append(("start", action["block_name"]))
        await asyncio.sleep(0.05)
        events.append(("end", action["block_name"]))

    monkeypatch.setitem(enrichment._ENRICHERS, "tools", fake)
    monkeypatch.setitem(enrichment._ENRICHERS, "topics", fake)

    actions = [
        {"type": "create_block", "block_type": "tools", "block_name": "a"},
        {"type": "create_block", "block_type": "topics", "block_name": "b"},
    ]
    await enrichment.enrich_actions(actions, "sid")

    # Both jobs start before either finishes → they overlapped.
    assert events[0][0] == "start"
    assert events[1][0] == "start"


@pytest.mark.asyncio
async def test_enrich_actions_tolerates_single_failure(monkeypatch):
    """One enricher raising does not sink the others (return_exceptions)."""

    async def boom(action, session_id):
        raise RuntimeError("enrichment failed")

    async def ok(action, session_id):
        action["enriched"] = True

    monkeypatch.setitem(enrichment._ENRICHERS, "tools", boom)
    monkeypatch.setitem(enrichment._ENRICHERS, "topics", ok)

    actions = [
        {"type": "create_block", "block_type": "tools", "block_name": "a"},
        {"type": "create_block", "block_type": "topics", "block_name": "b"},
    ]
    await enrichment.enrich_actions(actions, "sid")  # must not raise

    assert actions[1].get("enriched") is True


@pytest.mark.asyncio
async def test_enrich_actions_ignores_non_create_and_unknown_types(monkeypatch):
    called: list[str] = []

    async def track(action, session_id):
        called.append(action["block_name"])

    monkeypatch.setitem(enrichment._ENRICHERS, "tools", track)
    actions = [
        {"type": "set_port", "block_type": "tools", "block_name": "skip-me"},
        {"type": "create_block", "block_type": "unknown_kind", "block_name": "skip-too"},
        {"type": "create_block", "block_type": "tools", "block_name": "do-me"},
    ]
    await enrichment.enrich_actions(actions, "sid")
    assert called == ["do-me"]
