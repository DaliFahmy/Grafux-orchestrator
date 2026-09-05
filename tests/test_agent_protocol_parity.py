"""Every key the Qt client reads, the server must actually write.

This is the guard for the bug that made the Agent button do nothing: the client
read ``agent_block_id`` off every ``agent_action`` and rejected the action when it
did not match a live block, while the server wrote that key nowhere at all. Three
readers, no writer, 346 green tests, and an agent that could not touch the canvas.

Nothing static-analyses the server here. A real ``_OrchestratorSession`` drives a
real ``BlockAgentLoop`` with a scripted model, every message it emits is captured,
and those are checked against the keys parsed out of the C++ that consumes them.
The messages are the evidence, so the test cannot drift from what is really sent.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from app.core.llm import ToolCall, ToolTurn
from app.modules.session import block_agent as ba
from app.modules.session.router import SESSION_FEATURES, _OrchestratorSession

CLIENT = (
    Path(__file__).resolve().parents[2]
    / "Grafux-app/src/clients/grafux-orchestrator/orchestratorclient.cpp"
)

# Keys the client reads but the server is not required to send, with the reason.
# Adding one here is a decision; the test exists to make it a deliberate one.
_OPTIONAL: dict[str, set[str]] = {
    # Advertised only by a server new enough to have the feature; an older one
    # sends session_ready with no features, which is read as "supports nothing".
    "session_ready": {"features"},
}


def _client_reads() -> dict[str, set[str]]:
    """{message type -> keys the C++ pulls off it} parsed from the dispatcher."""
    source = CLIENT.read_text(encoding="utf-8", errors="replace")
    reads: dict[str, set[str]] = {}
    # if (type == QStringLiteral("agent_action")) { ...root.value(QStringLiteral("x"))... }
    pattern = re.compile(
        r'if \(type == QStringLiteral\("(\w+)"\)\) \{(.*?)\n    \}',
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        msg_type, body = match.group(1), match.group(2)
        keys = set(re.findall(r'root\.value\(QStringLiteral\("(\w+)"\)\)', body))
        if keys:
            reads[msg_type] = keys
    return reads


class _Client:
    """Captures everything the server sends, and answers like the Qt app."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._task: asyncio.Task | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def attach(self, session: _OrchestratorSession) -> None:
        self._session = session
        self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _pump(self) -> None:
        seen = 0
        while True:
            await asyncio.sleep(0.005)
            while seen < len(self.sent):
                message = self.sent[seen]
                seen += 1
                if message.get("type") == "read_port":
                    await self._reply({"type": "port_content", "found": True,
                                       "content": "spec text"}, message)
                elif message.get("type") == "agent_action":
                    await self._reply({
                        "type": "agent_action_result", "ok": True, "status": "done",
                        "observation": {"block": "sync_fifo", "status": "Success",
                                        "ports": []},
                    }, message)
                elif message.get("type") == "agent_question":
                    await self._reply({"type": "agent_user_message", "text": "yes"},
                                      message)

    async def _reply(self, payload: dict, request: dict) -> None:
        await self._session._dispatch_text(
            json.dumps({**payload, "request_id": request["request_id"]})
        )


async def _capture_every_agent_message() -> list[dict]:
    """Run one agent through every message it can produce, and collect them."""
    client = _Client()
    session = _OrchestratorSession(client, "sid", "uid", "pid")
    client.attach(session)

    script = [
        [ToolCall("t1", "read_port_value", {"target_block": "fifo_spec",
                                            "direction": "output", "port_name": "spec"})],
        [ToolCall("t2", "set_port_value", {"target_block": "sync_fifo",
                                           "direction": "input", "port_name": "top",
                                           "value": "sync_fifo"})],
        [ToolCall("t3", "ask_user", {"question": "Run the simulation?"})],
        [ToolCall("t4", "run_block", {"target_block": "sync_fifo"})],
        [ToolCall("t5", "finish", {"summary": "done", "goal_met": "true"})],
    ]
    turns = 0

    async def fake_llm(system, messages, declarations, *, model=None, **kw):
        nonlocal turns
        calls = script[turns] if turns < len(script) else []
        turns += 1
        return ToolTurn("thinking about it", calls,
                        "tool_use" if calls else "end", {}, "anthropic", "claude-opus-5")

    import app.modules.session.block_agent as mod
    original = mod.call_llm_tools
    mod.call_llm_tools = fake_llm
    try:
        await session._start_block_agent({
            "block_id": "b7", "block_name": "sync_fifo", "block_type": "code_hdl",
            "canvas_state": {"blocks": [{"name": "sync_fifo", "type": "code_hdl",
                                         "status": "Idle", "ports": []}]},
        })
        await asyncio.wait_for(session._agent_tasks["b7"], timeout=15)
    finally:
        mod.call_llm_tools = original
        await client.stop()

    # session_ready is sent by the endpoint, not the session, so add it here with
    # the shape ws_session actually builds.
    return [{"type": "session_ready", "session_id": "sid", "features": SESSION_FEATURES},
            *client.sent]


@pytest.mark.asyncio
async def test_the_server_writes_every_key_the_client_reads():
    if not CLIENT.exists():
        pytest.skip("Grafux-app is not checked out alongside the orchestrator")

    reads = _client_reads()
    assert "agent_action" in reads, "could not parse the C++ dispatcher"

    sent = await _capture_every_agent_message()
    by_type: dict[str, list[dict]] = {}
    for message in sent:
        by_type.setdefault(message.get("type", ""), []).append(message)

    missing: list[str] = []
    for msg_type, keys in reads.items():
        for message in by_type.get(msg_type, []):
            for key in keys - _OPTIONAL.get(msg_type, set()):
                if key not in message:
                    missing.append(f"{msg_type}.{key}")

    assert not missing, (
        "the Qt client reads these keys off messages the server sends without "
        f"them: {sorted(set(missing))}. Either write them, or add them to "
        "_OPTIONAL with the reason they may be absent."
    )


@pytest.mark.asyncio
async def test_an_agent_action_is_attributed_to_its_agent():
    """The specific regression: an unattributed action is refused by the client."""
    sent = await _capture_every_agent_message()
    actions = [m for m in sent if m["type"] == "agent_action"]
    assert actions, "the agent never asked for a canvas action"
    for message in actions:
        assert message.get("agent_block_id") == "b7", (
            "agent_action must name the agent that asked for it; the client "
            "rejects an action it cannot attribute to a live agent"
        )


@pytest.mark.asyncio
async def test_every_agent_message_the_client_knows_is_actually_produced():
    """A handler for a message nobody sends is dead code; catch it early."""
    if not CLIENT.exists():
        pytest.skip("Grafux-app is not checked out alongside the orchestrator")

    sent = await _capture_every_agent_message()
    produced = {m.get("type") for m in sent}
    agent_handlers = {t for t in _client_reads() if t.startswith("agent_")}

    assert agent_handlers <= produced, (
        f"the client handles {sorted(agent_handlers - produced)} but a full agent "
        "run never produces them"
    )


def test_the_server_advertises_the_block_agent_feature():
    """The client refuses to start an agent on a server that does not claim it."""
    assert "block_agent" in SESSION_FEATURES
