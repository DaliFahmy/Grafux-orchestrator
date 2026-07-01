from __future__ import annotations

import json

import pytest
from app.modules.session import canvas_tools, enrichment


class _FakeDevices:
    """Stand-in for DevicesClient — records the scaffold call, returns canned data."""

    def __init__(self, drafted=None, toolkits=None):
        self._drafted = drafted or {}
        self._toolkits = toolkits or []
        self.last = None

    async def scaffold_claw(self, description, name="", connections=None):
        self.last = {"description": description, "name": name, "connections": connections}
        return self._drafted

    async def list_claw_toolkits(self):
        return self._toolkits


def _use_fake_devices(monkeypatch, fake):
    # The enricher imports DevicesClient inside the function, so patch it at its source.
    monkeypatch.setattr("app.modules.devices.client.DevicesClient", lambda: fake)


def _inputs(action):
    return {p["port_name"]: p for p in action["input_ports"]}


# ── registration ──────────────────────────────────────────────────────────────

def test_claw_registered_to_claw_enricher():
    assert enrichment._ENRICHERS["claw"] is enrichment._enrich_claw_block
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "claw"})


# ── design-port mapping ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enrich_claw_maps_design_ports_and_connections(monkeypatch):
    drafted = {
        "soul": "You are a sheet bot.",
        "task": "append rows",
        "agent": "claude-opus-4-8",
        "connections": json.dumps(["googlesheets"]),
        # Secrets are returned as placeholders but must NOT be written into the block.
        "credentials": "<your-service-credentials>",
        "api_keys": "<your-anthropic-api-key>",
    }
    fake = _FakeDevices(drafted=drafted)
    _use_fake_devices(monkeypatch, fake)

    action = {
        "type": "create_block", "block_type": "claw", "block_name": "sheets_bot",
        "description": "send info to google sheets", "connections": ["google sheets"],
    }
    status = await enrichment._enrich_claw_block(action, "sid")
    assert status == ("ok", "")

    ins = _inputs(action)
    assert ins["connections"]["port_content"] == json.dumps(["googlesheets"])
    assert ins["soul"]["port_content"] == "You are a sheet bot."
    assert ins["task"]["port_content"] == "append rows"
    # Secret ports keep the empty scaffold value.
    assert ins["credentials"]["port_content"] == ""
    assert ins["api_keys"]["port_content"] == ""
    # The explicit apps were forwarded to the devices scaffolder as a hint.
    assert fake.last["connections"] == ["google sheets"]


@pytest.mark.asyncio
async def test_enrich_claw_fallback_normalizes_explicit(monkeypatch):
    # Devices scaffolder unreachable (returns {}) but the model named apps → normalize locally.
    fake = _FakeDevices(drafted={}, toolkits=["googlesheets", "gmail"])
    _use_fake_devices(monkeypatch, fake)
    enrichment._toolkits_cache.update({"at": 0.0, "slugs": []})  # reset the TTL cache

    action = {
        "type": "create_block", "block_type": "claw", "block_name": "b",
        "description": "x", "connections": ["google sheets"],
    }
    await enrichment._enrich_claw_block(action, "sid")
    ins = _inputs(action)
    assert json.loads(ins["connections"]["port_content"]) == ["googlesheets"]


@pytest.mark.asyncio
async def test_enrich_claw_scaffold_only_when_nothing_drafted(monkeypatch):
    fake = _FakeDevices(drafted={})
    _use_fake_devices(monkeypatch, fake)

    action = {
        "type": "create_block", "block_type": "claw", "block_name": "b", "description": "x",
    }
    status = await enrichment._enrich_claw_block(action, "sid")
    assert status == ("ok", "")
    ins = _inputs(action)
    # All canonical claw ports exist; connections stays empty (nothing implied/hinted).
    assert {"soul", "connections", "credentials", "api_keys", "task"} <= set(ins)
    assert ins["connections"]["port_content"] == ""


# ── model tool threading ──────────────────────────────────────────────────────

def test_create_block_threads_connections():
    action = canvas_tools.function_call_to_action(
        "create_block",
        {"block_type": "claw", "block_name": "b", "description": "d",
         "connections": ["googlesheets"]},
    )
    assert action["connections"] == ["googlesheets"]


def test_create_block_omits_connections_when_absent():
    action = canvas_tools.function_call_to_action(
        "create_block", {"block_type": "topics", "block_name": "b", "description": "d"},
    )
    assert "connections" not in action
