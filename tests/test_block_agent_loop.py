"""The block agent loop, driven against a fake canvas and a scripted model.

Everything here runs with no WebSocket, no client and no LLM: the loop reaches
the world only through CanvasPort, which is the reason it is a protocol. What
these tests are really guarding is that the agent OBSERVES -- that a tool result
comes back into the transcript, that a denial is something the model can read and
react to, and that no path lets the loop stop without telling the user why.
"""

from __future__ import annotations

import pytest
from app.core.llm import ToolCall, ToolTurn
from app.modules.session import block_agent as ba
from app.modules.session.block_agent import AgentPolicy, BlockAgentLoop


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeCanvas:
    """Records what the agent did, and replays canned observations."""

    def __init__(self, observations=None, port_values=None, answer="yes"):
        self.actions: list[dict] = []
        self.emitted: list[dict] = []
        self.reads: list[tuple[str, str, str]] = []
        self.questions: list[str] = []
        self._observations = list(observations or [])
        self._port_values = port_values or {}
        self._answer = answer

    async def act(self, action, *, expensive=False):
        self.actions.append({**action, "_expensive": expensive})
        if self._observations:
            return self._observations.pop(0)
        return {"ok": True, "detail": "applied"}

    async def read_port(self, target_block, direction, port_name):
        self.reads.append((target_block, direction, port_name))
        return self._port_values.get(port_name, "")

    async def emit(self, message):
        self.emitted.append(message)

    async def ask(self, block_id, question):
        self.questions.append(question)
        return self._answer

    def diagram_context(self, *, budget=6000):
        return "Block \"sync_fifo\" (code_hdl, Idle)  [ACTIVE]"

    # helpers
    def steps(self, kind=None):
        return [m for m in self.emitted
                if m["type"] == "agent_step" and (kind is None or m["kind"] == kind)]

    def states(self):
        return [m for m in self.emitted if m["type"] == "agent_state"]


def _script(monkeypatch, turns):
    """Install a call_llm_tools that returns `turns` in order, recording inputs."""
    seen: list[dict] = []
    queue = list(turns)

    async def fake(system, messages, declarations, *, model=None, **kw):
        seen.append({
            "system": system,
            "messages": list(messages),
            "tools": [d["name"] for d in declarations],
            "model": model,
        })
        if queue:
            return queue.pop(0)
        return ToolTurn("nothing left to do", [], "end", {}, "anthropic", "claude-opus-5")

    monkeypatch.setattr(ba, "call_llm_tools", fake)
    return seen


def _turn(text="", calls=(), model="claude-opus-5", stop=None):
    calls = list(calls)
    return ToolTurn(
        text=text,
        tool_calls=calls,
        stop_reason=stop or ("tool_use" if calls else "end"),
        usage={},
        provider="anthropic",
        model=model,
    )


def _finish(summary="done", goal_met="true"):
    return ToolCall("f1", "finish", {"summary": summary, "goal_met": goal_met})


def _agent(canvas, **kw):
    kw.setdefault("block_type", "code_hdl")
    return BlockAgentLoop(
        block_id="b7", block_name="sync_fifo", canvas=canvas, **kw
    )


# ── The loop observes ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tool_result_goes_back_into_the_transcript(monkeypatch):
    """The whole point: what happened must reach the model's next turn."""
    canvas = _FakeCanvas(observations=[{
        "ok": True, "block": "sync_fifo", "status": "Success",
        "ports": [{"name": "code", "direction": "output", "changed": True,
                   "value": "module sync_fifo;"}],
    }])
    seen = _script(monkeypatch, [
        _turn("Regenerating.", [ToolCall("t1", "regenerate_block", {"target_block": "sync_fifo"})]),
        _turn("", [_finish()]),
    ])
    agent = _agent(canvas)

    await agent.run()

    second_turn = seen[1]["messages"]
    tool_msg = next(m for m in second_turn if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "t1"
    assert "sync_fifo is now Success" in tool_msg["content"]
    assert "module sync_fifo;" in tool_msg["content"]
    assert tool_msg["is_error"] is False


@pytest.mark.asyncio
async def test_a_failed_action_comes_back_as_a_tool_error(monkeypatch):
    canvas = _FakeCanvas(observations=[{"ok": False, "detail": "block not found on canvas"}])
    seen = _script(monkeypatch, [
        _turn("", [ToolCall("t1", "run_block", {"target_block": "ghost"})]),
        _turn("", [_finish(goal_met="false")]),
    ])

    await _agent(canvas).run()

    tool_msg = next(m for m in seen[1]["messages"] if m.get("role") == "tool")
    assert tool_msg["is_error"] is True
    assert "FAILED" in tool_msg["content"]
    assert "block not found" in tool_msg["content"]


@pytest.mark.asyncio
async def test_read_port_does_not_touch_the_canvas(monkeypatch):
    canvas = _FakeCanvas(port_values={"spec": "1. The FIFO holds 8 entries."})
    _script(monkeypatch, [
        _turn("", [ToolCall("t1", "read_port_value",
                            {"target_block": "fifo_spec", "direction": "output",
                             "port_name": "spec"})]),
        _turn("", [_finish()]),
    ])

    await _agent(canvas).run()

    assert canvas.reads == [("fifo_spec", "output", "spec")]
    assert canvas.actions == []          # a read is not a canvas mutation


@pytest.mark.asyncio
async def test_read_defaults_to_the_agents_own_block(monkeypatch):
    canvas = _FakeCanvas(port_values={"code": "x"})
    _script(monkeypatch, [
        _turn("", [ToolCall("t1", "read_port_value", {"port_name": "code"})]),
        _turn("", [_finish()]),
    ])

    await _agent(canvas).run()

    assert canvas.reads == [("sync_fifo", "output", "code")]


# ── Termination ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finish_ends_the_run_and_reports_the_verdict(monkeypatch):
    canvas = _FakeCanvas()
    _script(monkeypatch, [
        _turn("", [ToolCall("f1", "finish", {
            "summary": "Wired the spec and regenerated.",
            "goal_met": "false",
            "blocking": "the simulation has not been run",
        })]),
    ])
    agent = _agent(canvas)

    await agent.run()

    assert agent.state == "finished"
    assert agent.goal_met is False
    assert agent.summary == "Wired the spec and regenerated."
    final = canvas.states()[-1]
    assert final["state"] == "finished"
    assert "the simulation has not been run" in canvas.steps("note")[-1]["text"]


@pytest.mark.asyncio
async def test_a_model_that_just_stops_is_finished_without_a_verdict(monkeypatch):
    """Never leave the user staring at an agent that quietly stopped."""
    canvas = _FakeCanvas()
    _script(monkeypatch, [_turn("I think that is everything.")])
    agent = _agent(canvas)

    await agent.run()

    assert agent.state == "finished"
    assert agent.goal_met is False
    assert agent.summary == "I think that is everything."


@pytest.mark.asyncio
async def test_stop_is_honoured_at_the_next_checkpoint(monkeypatch):
    canvas = _FakeCanvas()
    agent = _agent(canvas)

    async def fake(system, messages, declarations, **kw):
        agent.stop()                      # the user presses Stop mid-step
        return _turn("", [ToolCall("t1", "run_block", {"target_block": "sync_fifo"})])

    monkeypatch.setattr(ba, "call_llm_tools", fake)
    await agent.run()

    assert agent.state == "stopped"
    assert canvas.states()[-1]["state"] == "stopped"


@pytest.mark.asyncio
async def test_an_llm_error_is_reported_not_swallowed(monkeypatch):
    canvas = _FakeCanvas()

    async def boom(*a, **kw):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(ba, "call_llm_tools", boom)
    agent = _agent(canvas)

    await agent.run()

    assert agent.state == "error"
    assert "upstream exploded" in agent.summary
    assert canvas.steps("error")


@pytest.mark.asyncio
async def test_a_refusal_stops_the_run_rather_than_being_read_as_an_answer(monkeypatch):
    canvas = _FakeCanvas()
    _script(monkeypatch, [_turn("cannot help with that", stop="refusal")])
    agent = _agent(canvas)

    await agent.run()

    assert agent.state == "error"
    assert canvas.actions == []


# ── Budgets ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_step_budget_ends_with_a_summary_not_a_silent_halt(monkeypatch):
    """A budget that just stops leaves a changed canvas and no explanation."""
    canvas = _FakeCanvas()
    calls: list[list] = []

    async def fake(system, messages, declarations, **kw):
        calls.append([d["name"] for d in declarations])
        if len(calls) > 2:               # the wrap-up call, made with no tools
            return _turn("I set the spec, but never ran the simulation.")
        return _turn("", [ToolCall(f"t{len(calls)}", "set_port_value",
                                   {"port_name": "top", "direction": "input",
                                    "value": "sync_fifo"})])

    monkeypatch.setattr(ba, "call_llm_tools", fake)
    agent = _agent(canvas, policy=AgentPolicy(max_steps=2))

    await agent.run()

    assert agent.state == "finished"
    assert agent.goal_met is False
    assert calls[-1] == []                       # the last call offered no tools
    assert "never ran the simulation" in agent.summary


@pytest.mark.asyncio
async def test_the_run_budget_denies_further_runs_in_words_the_model_can_act_on(monkeypatch):
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [
        _turn("", [ToolCall("t1", "run_block", {"target_block": "sync_fifo"})]),
        _turn("", [ToolCall("t2", "run_block", {"target_block": "sync_fifo"})]),
        _turn("", [_finish(goal_met="false")]),
    ])
    agent = _agent(canvas, block_type="code_hdl", policy=AgentPolicy(max_block_runs=1))

    await agent.run()

    assert len(canvas.actions) == 1              # the second run never reached the canvas
    denial = next(m for m in seen[2]["messages"]
                  if m.get("role") == "tool" and m["tool_call_id"] == "t2")
    assert denial["is_error"] is True
    assert "all 1 runs" in denial["content"]
    assert canvas.steps("denied")


@pytest.mark.asyncio
async def test_a_cloud_run_is_metered_separately(monkeypatch):
    """verilator/yosys/openroad/gpu cost real money, so they have their own budget."""
    canvas = _FakeCanvas()
    _script(monkeypatch, [
        _turn("", [ToolCall("t1", "run_block", {"target_block": "fifo_sim"})]),
        _turn("", [_finish()]),
    ])
    agent = BlockAgentLoop(
        block_id="v1", block_name="fifo_sim", block_type="verilator", canvas=canvas,
    )

    await agent.run()

    assert agent.expensive_runs_used == 1
    assert canvas.actions[0]["_expensive"] is True


@pytest.mark.asyncio
async def test_expensive_runs_can_be_refused_by_policy(monkeypatch):
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [
        _turn("", [ToolCall("t1", "run_block", {"target_block": "fifo_sim"})]),
        _turn("", [_finish(goal_met="false")]),
    ])
    agent = BlockAgentLoop(
        block_id="v1", block_name="fifo_sim", block_type="verilator", canvas=canvas,
        policy=AgentPolicy(may_run_expensive=False),
    )

    await agent.run()

    assert canvas.actions == []
    denial = next(m for m in seen[1]["messages"] if m.get("role") == "tool")
    assert "paid cloud machine" in denial["content"]
    assert "ask_user" in denial["content"]


# ── Policy shapes the tool list ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_forbidden_tool_is_never_offered(monkeypatch):
    """Withholding beats declaring-then-refusing: every refusal costs a step."""
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [_turn("", [_finish()])])

    await _agent(canvas).run()

    assert "delete_block" not in seen[0]["tools"]
    assert "finish" in seen[0]["tools"]
    assert "read_port_value" in seen[0]["tools"]


@pytest.mark.asyncio
async def test_delete_is_offered_only_when_policy_allows_it(monkeypatch):
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [_turn("", [_finish()])])

    await _agent(canvas, policy=AgentPolicy(may_delete=True)).run()

    assert "delete_block" in seen[0]["tools"]


@pytest.mark.asyncio
async def test_a_read_only_agent_offers_no_mutations(monkeypatch):
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [_turn("", [_finish()])])
    policy = AgentPolicy(may_mutate_ports=False, may_wire=False,
                         may_create=False, may_run=False)

    await _agent(canvas, policy=policy).run()

    assert set(seen[0]["tools"]) == {"read_port_value", "finish", "ask_user"}


# ── ask_user ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_user_round_trips_the_answer(monkeypatch):
    canvas = _FakeCanvas(answer="no, leave it alone")
    seen = _script(monkeypatch, [
        _turn("", [ToolCall("t1", "ask_user", {"question": "Delete the stale block?"})]),
        _turn("", [_finish(goal_met="false")]),
    ])

    await _agent(canvas).run()

    assert canvas.questions == ["Delete the stale block?"]
    answer = next(m for m in seen[1]["messages"] if m.get("role") == "tool")
    assert answer["content"] == "no, leave it alone"


@pytest.mark.asyncio
async def test_an_unanswered_question_does_not_strand_the_agent(monkeypatch):
    canvas = _FakeCanvas(answer="")
    seen = _script(monkeypatch, [
        _turn("", [ToolCall("t1", "ask_user", {"question": "Run it?"})]),
        _turn("", [_finish(goal_met="false")]),
    ])

    await _agent(canvas).run()

    answer = next(m for m in seen[1]["messages"] if m.get("role") == "tool")
    assert "assume no and continue" in answer["content"]


# ── Prompt assembly ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_cached_system_prompt_never_changes_between_steps(monkeypatch):
    """Volatile canvas text in the cached prefix would mean no caching at all."""
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [
        _turn("", [ToolCall("t1", "set_port_value",
                            {"port_name": "top", "direction": "input", "value": "x"})]),
        _turn("", [_finish()]),
    ])

    await _agent(canvas, goal="hit 500 MHz").run()

    assert seen[0]["system"] == seen[1]["system"]
    assert "sync_fifo" in seen[0]["system"]
    assert "hit 500 MHz" in seen[0]["system"]
    assert "Idle" not in seen[0]["system"]        # the canvas render is not in here


@pytest.mark.asyncio
async def test_the_canvas_rides_as_a_mid_conversation_system_message(monkeypatch):
    canvas = _FakeCanvas()
    seen = _script(monkeypatch, [_turn("", [_finish()])])

    await _agent(canvas).run()

    last = seen[0]["messages"][-1]
    assert last["role"] == "system"
    assert "CURRENT CANVAS" in last["content"]
    assert "(code_hdl, Idle)" in last["content"]


def test_the_objective_falls_back_rather_than_going_blank():
    """get_system_prompt returns "" for an unknown section; a blank objective
    would produce a confident, directionless agent."""
    assert ba.objective_section("code_hdl") != ba.objective_section("topics")
    assert ba.objective_section("topics")            # the default, not empty
    assert ba.objective_section("no_such_type") == ba.objective_section("topics")


def test_every_prompt_placeholder_is_filled():
    prompt = ba.build_system_prompt(
        block_name="sync_fifo", block_type="code_hdl",
        policy=AgentPolicy(), goal="",
    )
    for placeholder in ("{BLOCK_NAME}", "{BLOCK_TYPE}", "{BLOCK_OBJECTIVE}",
                        "{AUTONOMY}", "{TEAM_GOAL}"):
        assert placeholder not in prompt
    assert "may NOT delete blocks" in prompt


@pytest.mark.asyncio
async def test_a_model_substitution_is_announced(monkeypatch):
    """The Agent dropdown offers gemini ids that silently become OpenAI."""
    canvas = _FakeCanvas()
    _script(monkeypatch, [_turn("", [_finish()], model="gpt-4o")])

    await _agent(canvas, model="gemini-2.5-flash").run()

    notes = [s["text"] for s in canvas.steps("note")]
    assert any("gemini-2.5-flash is not available here" in n for n in notes)


# ── Observation rendering ────────────────────────────────────────────────────


def test_observation_keeps_the_truncation_wording_the_model_already_knows():
    text = ba._render_observation({
        "ok": True, "block": "sync_fifo", "status": "Success",
        "ports": [{"name": "code", "direction": "output", "changed": True,
                   "value": "x" * 20, "truncated": True, "full_length": 4211}],
    })
    assert "truncated: showing 20 of 4211 chars" in text
    assert "read_port_value" in text


def test_observation_says_when_nothing_changed():
    text = ba._render_observation({"ok": True, "status": "Success", "ports": []})
    assert "No output port changed." in text


def test_observation_surfaces_a_reported_error():
    text = ba._render_observation({"ok": True, "status": "Error", "error": "syntax error"})
    assert "Reported error: syntax error" in text


# ── Attribution, detachment, resumption ──────────────────────────────────────


@pytest.mark.asyncio
async def test_every_action_says_which_agent_asked_for_it(monkeypatch):
    """The client refuses an action it cannot attribute to a live agent.

    Without this the id arrives empty, the client's ownership guard rejects
    every action, and the fallback target degrades to whichever tab the user
    happens to be looking at -- so one agent's edit lands on another's block.
    """
    canvas = _FakeCanvas()
    _script(monkeypatch, [
        _turn(calls=[ToolCall("c1", "set_port_value", {
            "target_block": "sync_fifo", "direction": "input",
            "port_name": "top", "value": "counter",
        })]),
        _turn(calls=[_finish()]),
    ])
    await _agent(canvas).run("wire it up")

    assert canvas.actions, "the agent never acted"
    assert all(a.get("agent_block_id") == "b7" for a in canvas.actions)


@pytest.mark.asyncio
async def test_a_detached_loop_goes_quiet(monkeypatch):
    """A replaced loop must not narrate over the agent that replaced it.

    stop() is cooperative, so the old loop still unwinds through run()'s finally
    and reports a final state. Landing after the new loop said "running", that
    stale "Stopped." tears down a live agent's UI on the client.
    """
    canvas = _FakeCanvas()
    _script(monkeypatch, [_turn(calls=[_finish()])])
    agent = _agent(canvas)
    agent.detach()

    await agent.run("do the thing")

    assert canvas.emitted == []
    assert agent._stopping is True          # detaching stops it too


@pytest.mark.asyncio
async def test_resume_keeps_the_transcript_and_the_sequence(monkeypatch):
    """The budget is per instruction; the conversation is not."""
    canvas = _FakeCanvas()
    _script(monkeypatch, [_turn(calls=[_finish(summary="first pass done")])])
    agent = _agent(canvas)
    await agent.run("make it synchronous")

    messages_before = len(agent._messages)
    seq_before = agent._seq
    assert agent.steps_used > 0

    agent.resume_budget()

    assert agent.steps_used == 0
    assert agent.runs_used == 0
    assert agent.expensive_runs_used == 0
    assert agent.goal_met is False
    assert agent.summary == ""
    assert agent._stopping is False
    assert len(agent._messages) == messages_before   # it still remembers
    assert agent._seq == seq_before                  # and keeps counting from here


@pytest.mark.asyncio
async def test_step_sequence_is_strictly_increasing(monkeypatch):
    """The client de-duplicates on seq, so a repeat would silently drop a step."""
    canvas = _FakeCanvas()
    _script(monkeypatch, [
        _turn(calls=[ToolCall("c1", "set_port_value", {
            "target_block": "sync_fifo", "direction": "input",
            "port_name": "top", "value": "counter",
        })]),
        _turn(calls=[_finish()]),
    ])
    await _agent(canvas).run("go")

    seqs = [m["seq"] for m in canvas.steps()]
    assert seqs == sorted(set(seqs)) and len(seqs) == len(set(seqs))
