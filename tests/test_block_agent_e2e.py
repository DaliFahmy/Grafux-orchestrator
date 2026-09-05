"""A real agent loop, through the real session, over the real wire protocol.

The unit tests either fake the canvas (test_block_agent_loop) or drive the
session directly (test_block_agent_actions). This one wires them together and
fakes only the model: a genuine BlockAgentLoop talks to a genuine
_OrchestratorSession, whose messages are answered the way the Qt client answers
them. What it proves is the thing the whole feature rests on -- that pressing
Agent produces a loop which reads the canvas, acts on it, SEES the result, and
reports an honest verdict.

The scenario is the one from the plan: a code_hdl block whose spec input is
empty, with a spec_hdl block sitting upstream that already has the answer.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from app.core.llm import ToolCall, ToolTurn
from app.modules.session import block_agent as ba
from app.modules.session.router import _OrchestratorSession

SPEC = "1. The FIFO holds 8 entries of 16 bits.\n2. Writes while full are ignored."
RTL = "module sync_fifo #(parameter W=16) (input clk, ...); endmodule"


class _FakeClient:
    """Stands in for the Qt app: answers the server's requests like it does."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.applied: list[dict] = []
        self.rejected: list[dict] = []
        # Mirrors ChatPanelWidget::m_agentBlocks: which agents the user has live.
        self.active_agents: set[str] = {"b7"}
        self._ports = {("fifo_spec", "output", "spec"): SPEC}
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
        """Answer every server request, the way the client's slots do."""
        seen = 0
        while True:
            await asyncio.sleep(0.005)
            while seen < len(self.sent):
                message = self.sent[seen]
                seen += 1
                await self._handle(message)

    async def _handle(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "read_port":
            key = (message["target_block"], message["direction"], message["port_name"])
            content = self._ports.get(key, "")
            await self._session._dispatch_text(json.dumps({
                "type": "port_content", "request_id": message["request_id"],
                "found": bool(content), "content": content,
            }))
        elif kind == "agent_action":
            await self._apply(message)

    async def _apply(self, message: dict) -> None:
        request_id = message["request_id"]

        # The real client refuses an action it cannot attribute to a live agent
        # (ChatPanelWidget::onAgentActionRequested). A fake that applies
        # everything is MORE PERMISSIVE than the thing it stands in for, which is
        # exactly how the missing agent_block_id shipped past a green suite.
        agent_block_id = message.get("agent_block_id", "")
        if agent_block_id and agent_block_id not in self.active_agents:
            await self._session._dispatch_text(json.dumps({
                "type": "agent_action_result", "request_id": request_id,
                "ok": False, "status": "rejected",
                "detail": "that block's agent has been stopped",
            }))
            self.rejected.append(message)
            return

        action = message["action"]
        self.applied.append(action)

        if action["type"] == "regenerate_block":
            # A run reports twice, exactly as BlockAgentController::runAndObserve
            # does: "started" first, then the real observation.
            await self._session._dispatch_text(json.dumps({
                "type": "agent_action_result", "request_id": request_id,
                "status": "started", "ok": True,
            }))
            await asyncio.sleep(0.01)
            await self._session._dispatch_text(json.dumps({
                "type": "agent_action_result", "request_id": request_id,
                "ok": True, "status": "done",
                "observation": {
                    "block": "sync_fifo", "status": "Success",
                    "ports": [{"name": "code", "direction": "output", "changed": True,
                               "value": RTL}],
                },
            }))
            return

        await self._session._dispatch_text(json.dumps({
            "type": "agent_action_result", "request_id": request_id,
            "ok": True, "status": "done",
            "observation": {"block": "sync_fifo", "status": "Idle", "ports": []},
        }))


def _canvas() -> dict:
    return {"blocks": [
        {"name": "fifo_spec", "type": "spec_hdl", "status": "Success",
         "ports": [{"name": "spec", "is_output": True, "value": SPEC[:40],
                    "truncated": True, "full_length": len(SPEC)}]},
        {"name": "sync_fifo", "type": "code_hdl", "status": "Idle",
         "ports": [{"name": "spec", "is_output": False, "value": ""},
                   {"name": "top", "is_output": False, "value": ""},
                   {"name": "code", "is_output": True, "value": ""}]},
    ]}


@pytest.mark.asyncio
async def test_an_agent_fills_its_empty_spec_and_reports_what_it_did(monkeypatch):
    """The Stage 1 demo, end to end.

    The block cannot produce good RTL without a spec, and its spec port is empty
    while the answer sits on an upstream block. A working agent notices, reads
    it, wires it, regenerates, checks the result and says so.
    """
    client = _FakeClient()
    session = _OrchestratorSession(client, "sid", "uid", "pid")
    client.attach(session)

    # The model's side of the conversation. Each turn is chosen by what the
    # PREVIOUS observation said, which is the behaviour under test.
    script = [
        # 1. Look first: the canvas says spec is empty and marks fifo_spec truncated.
        [ToolCall("t1", "read_port_value", {"target_block": "fifo_spec",
                                            "direction": "output", "port_name": "spec"})],
        # 2. Fix the input by wiring, not copying.
        [ToolCall("t2", "connect_ports", {"from_block": "fifo_spec", "from_port": "spec",
                                          "to_block": "sync_fifo", "to_port": "spec"}),
         ToolCall("t3", "set_port_value", {"target_block": "sync_fifo", "direction": "input",
                                           "port_name": "top", "value": "sync_fifo"})],
        # 3. Do the work.
        [ToolCall("t4", "regenerate_block", {"target_block": "sync_fifo"})],
        # 4. Check, then report.
        [ToolCall("t5", "read_port_value", {"target_block": "sync_fifo",
                                            "direction": "output", "port_name": "code"})],
        [ToolCall("t6", "finish", {
            "summary": "Wired fifo_spec.spec into sync_fifo.spec, set top, and "
                       "regenerated: 8-entry 16-bit FIFO with writes-while-full ignored.",
            "goal_met": "true",
        })],
    ]
    turns_seen: list[list[dict]] = []

    async def fake_llm(system, messages, declarations, *, model=None, **kw):
        turns_seen.append(list(messages))
        calls = script[len(turns_seen) - 1] if len(turns_seen) <= len(script) else []
        return ToolTurn("", calls, "tool_use" if calls else "end", {},
                        "anthropic", "claude-opus-5")

    monkeypatch.setattr(ba, "call_llm_tools", fake_llm)

    await session._start_block_agent({
        "block_id": "b7", "block_name": "sync_fifo", "block_type": "code_hdl",
        "model": "claude-opus-5", "canvas_state": _canvas(),
        "active_blocks": [_canvas()["blocks"][1]],
    })
    await asyncio.wait_for(session._agent_tasks["b7"], timeout=10)
    await client.stop()

    agent = session._agents["b7"]
    assert agent.state == "finished"
    assert agent.goal_met is True

    assert not client.rejected, (
        "the client refused an action it could not attribute to a live agent: "
        f"{client.rejected}"
    )

    # It really wired the blocks rather than pasting the text across.
    kinds = [a["type"] for a in client.applied]
    assert kinds == ["connect_ports", "set_port_value", "regenerate_block"]
    wire = client.applied[0]
    assert (wire["from_block"], wire["to_block"]) == ("fifo_spec", "sync_fifo")

    # And it OBSERVED: the regenerated code came back into its transcript, which
    # is what the chat path could never do.
    tool_results = [m for turn in turns_seen for m in turn if m.get("role") == "tool"]
    assert any(RTL in m["content"] for m in tool_results)
    assert any(SPEC in m["content"] for m in tool_results)

    # The user is told what happened, in order.
    steps = [m for m in client.sent if m["type"] == "agent_step"]
    assert [s["kind"] for s in steps][:2] == ["tool_call", "tool_result"]
    assert any("Wired fifo_spec.spec" in s["text"] for s in steps)
    final = [m for m in client.sent if m["type"] == "agent_state"][-1]
    assert final["state"] == "finished" and final["goal_met"] is True


@pytest.mark.asyncio
async def test_the_canvas_the_model_sees_is_refreshed_every_step(monkeypatch):
    """The agent is changing the canvas, so a snapshot taken once would go stale."""
    client = _FakeClient()
    session = _OrchestratorSession(client, "sid", "uid", "pid")
    client.attach(session)

    contexts: list[str] = []

    async def fake_llm(system, messages, declarations, *, model=None, **kw):
        contexts.append(messages[-1]["content"])
        calls = ([ToolCall("t1", "set_port_value",
                           {"target_block": "sync_fifo", "direction": "input",
                            "port_name": "top", "value": "sync_fifo"})]
                 if len(contexts) == 1
                 else [ToolCall("f", "finish", {"summary": "done", "goal_met": "true"})])
        return ToolTurn("", calls, "tool_use", {}, "anthropic", "claude-opus-5")

    monkeypatch.setattr(ba, "call_llm_tools", fake_llm)

    await session._start_block_agent({
        "block_id": "b7", "block_name": "sync_fifo", "block_type": "code_hdl",
        "canvas_state": _canvas(),
    })
    await asyncio.wait_for(session._agent_tasks["b7"], timeout=10)
    await client.stop()

    assert len(contexts) >= 2
    for text in contexts:
        assert text.startswith("CURRENT CANVAS")
        assert "sync_fifo" in text


@pytest.mark.asyncio
async def test_a_disconnect_mid_action_unwinds_the_agent_instead_of_hanging(monkeypatch):
    """The failure this rules out: a loop parked forever on a dead socket."""

    class _DeafClient(_FakeClient):
        async def _handle(self, message: dict) -> None:
            return                          # never answers anything

    client = _DeafClient()
    session = _OrchestratorSession(client, "sid", "uid", "pid")
    client.attach(session)

    async def fake_llm(system, messages, declarations, *, model=None, **kw):
        return ToolTurn("", [ToolCall("t1", "set_port_value", {
            "target_block": "sync_fifo", "direction": "input",
            "port_name": "top", "value": "x"})], "tool_use", {},
            "anthropic", "claude-opus-5")

    monkeypatch.setattr(ba, "call_llm_tools", fake_llm)

    await session._start_block_agent({
        "block_id": "b7", "block_name": "sync_fifo", "block_type": "code_hdl",
        "canvas_state": _canvas(),
    })
    task = session._agent_tasks["b7"]
    await asyncio.sleep(0.05)               # let it get as far as the round-trip

    await session.cleanup()                 # the socket dies

    await asyncio.wait_for(asyncio.shield(task), timeout=3)
    await client.stop()
    assert task.done()
