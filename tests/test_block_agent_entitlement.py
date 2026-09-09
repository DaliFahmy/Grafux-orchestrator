"""Block agents are a paid capability. These tests pin the gate and its failure policy.

Two things here are worth more than the rest:

1. A refusal must register NO agent. That is what closes the resume path --
   ``_on_agent_user_message`` only reaches ``_resume_block_agent`` when
   ``self._agents.get(block_id)`` exists, so a client-side gate alone would have
   left "type into a finished agent tab" as a way in.

2. A refusal must UNWIND an old client. ``OrchestratorClient::onTextMessage`` is
   a flat if-chain that silently drops unknown frame types, so a lone
   ``upgrade_required`` would leave the block stuck showing "Agent working on...".
   The paired ``agent_state{state:"error"}`` is what every client already knows
   how to handle.
"""
from __future__ import annotations

import json

import pytest

from app.core import entitlements as ent
from app.modules.session.router import _OrchestratorSession


class _FakeWebSocket:
    """Records what the server sent. Strict on purpose: no attribute it does not have."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def close(self, **_kw) -> None:
        return None


def _session() -> tuple[_OrchestratorSession, _FakeWebSocket]:
    ws = _FakeWebSocket()
    return _OrchestratorSession(ws, "sid", "uid", "pid", auth_token="tok"), ws


def _start(block_id="b1", block_name="Spec"):
    return {"type": "start_block_agent", "block_id": block_id, "block_name": block_name}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_refused_agent_is_never_registered(monkeypatch):
    """No entry in _agents is what also closes the resume path."""
    async def deny(_token, _user):
        return False

    monkeypatch.setattr("app.modules.session.router.may_run_block_agents", deny)
    session, _ws = _session()

    await session._start_block_agent(_start())

    assert session._agents == {}, "a refused agent must leave nothing registered"


@pytest.mark.asyncio
async def test_typing_after_a_refusal_cannot_restart_the_loop(monkeypatch):
    """The bypass a client-only gate would have left open."""
    async def deny(_token, _user):
        return False

    monkeypatch.setattr("app.modules.session.router.may_run_block_agents", deny)
    session, ws = _session()

    await session._start_block_agent(_start())
    ws.sent.clear()

    await session._on_agent_user_message({"block_id": "b1", "text": "carry on then"})

    assert session._agents == {}
    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors, "the follow-up message should be refused, not silently dropped"
    assert "no active agent" in errors[0]["message"].lower()


@pytest.mark.asyncio
async def test_a_refusal_unwinds_the_block_on_every_client(monkeypatch):
    """agent_state{error} is the frame old builds already know how to handle."""
    async def deny(_token, _user):
        return False

    monkeypatch.setattr("app.modules.session.router.may_run_block_agents", deny)
    session, ws = _session()

    await session._start_block_agent(_start(block_name="Spec"))

    states = [m for m in ws.sent if m.get("type") == "agent_state"]
    assert states, "without agent_state the block stays stuck showing 'Stop'"
    assert states[0]["state"] == "error"
    assert states[0]["block_id"] == "b1"
    # An empty summary is discarded by the client before it is ever shown.
    assert states[0]["summary"].strip()
    assert "Agentic" in states[0]["summary"]


@pytest.mark.asyncio
async def test_a_refusal_also_sends_the_machine_readable_frame(monkeypatch):
    async def deny(_token, _user):
        return False

    monkeypatch.setattr("app.modules.session.router.may_run_block_agents", deny)
    session, ws = _session()

    await session._start_block_agent(_start())

    upgrades = [m for m in ws.sent if m.get("type") == "upgrade_required"]
    assert len(upgrades) == 1
    assert upgrades[0]["feature"] == "block_agent"
    assert upgrades[0]["plan_required"] == "agentic"
    assert upgrades[0]["block_id"] == "b1"


@pytest.mark.asyncio
async def test_the_refusal_introduces_no_new_agent_prefixed_frame(monkeypatch):
    """test_agent_protocol_parity asserts every `agent_*` handler the client has is
    produced by a normal agent run. A refusal never is, so a refusal-only frame
    named `agent_something` would fail that test the moment the client learned to
    handle it. `agent_state` is exempt because a normal run produces it too."""
    async def deny(_token, _user):
        return False

    monkeypatch.setattr("app.modules.session.router.may_run_block_agents", deny)
    session, ws = _session()

    await session._start_block_agent(_start())

    agent_prefixed = {m["type"] for m in ws.sent if m["type"].startswith("agent_")}
    assert agent_prefixed == {"agent_state"}, (
        f"refusal-only frames must not be agent_*-named: {sorted(agent_prefixed)}"
    )


@pytest.mark.asyncio
async def test_an_entitled_user_starts_normally(monkeypatch):
    """Guard the happy path, so the gate cannot quietly deny everyone."""
    async def allow(_token, _user):
        return True

    async def noop(_token, _user):
        return None

    started: list[str] = []

    class _Loop:
        def __init__(self, **kwargs):
            self.block_id = kwargs["block_id"]
            self.block_name = kwargs["block_name"]
            self.block_type = kwargs.get("block_type", "")
            self.policy = type("P", (), {"max_steps": 1})()

        async def run(self, _instruction):
            started.append(self.block_id)

    monkeypatch.setattr("app.modules.session.router.may_run_block_agents", allow)
    monkeypatch.setattr(
        "app.modules.session.router.audit_block_agent_entitlement", noop
    )
    monkeypatch.setattr("app.modules.session.router.BlockAgentLoop", _Loop)

    session, ws = _session()
    await session._start_block_agent(_start())

    assert "b1" in session._agents
    assert not [m for m in ws.sent if m.get("type") == "upgrade_required"]


# ---------------------------------------------------------------------------
# The failure policy
# ---------------------------------------------------------------------------

class _Settings:
    backend_url = "http://backend"
    enforce_agent_entitlement = True
    entitlement_fallback = "allow"
    entitlement_cache_ttl = 300
    entitlement_lkg_ttl = 604800


def _caches(monkeypatch, *, hot=None, lkg=None):
    """Install fake Redis reads/writes and record what gets written."""
    written: dict[str, set[str]] = {}

    async def fake_read(key):
        if key.startswith("ent:lkg:"):
            return set(lkg) if lkg is not None else None
        return set(hot) if hot is not None else None

    async def fake_write(user_id, entitlements):
        written[user_id] = set(entitlements)

    monkeypatch.setattr(ent, "_cache_read", fake_read)
    monkeypatch.setattr(ent, "_cache_write", fake_write)
    monkeypatch.setattr(ent, "get_settings", lambda: _Settings())
    return written


@pytest.mark.asyncio
async def test_the_kill_switch_short_circuits_everything(monkeypatch):
    """Shipped OFF: every account currently resolves to free, so enforcing first
    would disable the Agent button for everyone including the owner."""
    calls = []

    class _Off(_Settings):
        enforce_agent_entitlement = False

    async def fetch(_t, _u):
        calls.append(1)
        return set()

    monkeypatch.setattr(ent, "get_settings", lambda: _Off())
    monkeypatch.setattr(ent, "_fetch", fetch)

    assert await ent.may_run_block_agents("tok", "u1") is True
    assert calls == [], "with the switch off there should be no HTTP call at all"


@pytest.mark.asyncio
async def test_a_cache_hit_makes_no_http_call(monkeypatch):
    calls = []

    async def fetch(_t, _u):
        calls.append(1)
        return set()

    _caches(monkeypatch, hot={"block_agent"})
    monkeypatch.setattr(ent, "_fetch", fetch)

    assert await ent.get_entitlements("tok", "u1") == {"block_agent"}
    assert calls == []


@pytest.mark.asyncio
async def test_an_outage_falls_back_to_last_known_good(monkeypatch):
    """A paying customer must not lose agents because the backend blipped."""
    _caches(monkeypatch, hot=None, lkg={"block_agent"})

    async def unreachable(_t, _u):
        return None

    monkeypatch.setattr(ent, "_fetch", unreachable)

    assert await ent.get_entitlements("tok", "u1") == {"block_agent"}


@pytest.mark.asyncio
async def test_an_outage_with_no_cache_at_all_uses_the_fallback(monkeypatch):
    _caches(monkeypatch, hot=None, lkg=None)

    async def unreachable(_t, _u):
        return None

    monkeypatch.setattr(ent, "_fetch", unreachable)
    assert await ent.get_entitlements("tok", "u1") == {"block_agent"}

    class _Deny(_Settings):
        entitlement_fallback = "deny"

    monkeypatch.setattr(ent, "get_settings", lambda: _Deny())
    assert await ent.get_entitlements("tok", "u1") == set()


@pytest.mark.asyncio
async def test_an_explicit_no_overwrites_the_last_known_good(monkeypatch):
    """Otherwise a cancelled subscription keeps working for the LKG's full week."""
    written = _caches(monkeypatch, hot=None, lkg={"block_agent"})

    async def says_no(_t, _u):
        return set()

    monkeypatch.setattr(ent, "_fetch", says_no)

    assert await ent.get_entitlements("tok", "u1") == set()
    assert written["u1"] == set(), "the stale yes must have been overwritten"


@pytest.mark.asyncio
async def test_a_denial_is_rechecked_past_the_cache(monkeypatch):
    """The common case for a denial is a user who has just paid."""
    answers = [set(), {"block_agent"}]

    _caches(monkeypatch, hot=None, lkg=None)

    async def fetch(_t, _u):
        return answers.pop(0)

    monkeypatch.setattr(ent, "_fetch", fetch)

    assert await ent.may_run_block_agents("tok", "u1") is True
    assert answers == [], "both the cached check and the forced refresh should run"
