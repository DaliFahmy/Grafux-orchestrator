"""The client round-trip that makes an agent able to OBSERVE.

Everything an agent does to the canvas goes out over the socket and comes back
as a result. These tests cover the ways that can go wrong -- a client that never
answers, a run so long no fixed timeout could cover it, and a socket that dies
mid-loop -- because each of them, handled badly, ends with an agent parked
forever on a future nobody will ever resolve.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from app.modules.session.block_agent import AgentPolicy
from app.modules.session.router import _OrchestratorSession


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _session() -> tuple[_OrchestratorSession, _FakeWS]:
    ws = _FakeWS()
    return _OrchestratorSession(ws, "sid", "uid", "pid"), ws


def _last(ws: _FakeWS, msg_type: str) -> dict:
    return next(m for m in reversed(ws.sent) if m["type"] == msg_type)


async def _reply_to(sess, ws, msg_type, payload, *, after=0.01):
    """Answer the client request of `msg_type` once it has been sent."""
    for _ in range(200):
        match = next((m for m in ws.sent if m["type"] == msg_type), None)
        if match:
            await asyncio.sleep(after)
            await sess._dispatch_text(json.dumps({**payload, "request_id": match["request_id"]}))
            return match
        await asyncio.sleep(0.005)
    raise AssertionError(f"no {msg_type} was ever sent")


# ── The happy path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_action_round_trips_into_an_observation():
    sess, ws = _session()
    action = {"type": "set_port_value", "target_block": "A",
              "direction": "input", "port_name": "top", "value": "counter"}

    task = asyncio.create_task(sess.act(action))
    await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "ok": True, "status": "done",
        "observation": {"block": "A", "status": "Success",
                        "ports": [{"name": "top", "direction": "input", "changed": True,
                                   "value": "counter"}]},
    })
    observation = await task

    sent = _last(ws, "agent_action")
    assert sent["action"] == action
    assert sent["expects"] == "result"
    assert observation["ok"] is True
    assert observation["status"] == "Success"
    assert observation["ports"][0]["value"] == "counter"


@pytest.mark.asyncio
async def test_a_rejected_action_is_reported_as_a_failure():
    sess, ws = _session()

    task = asyncio.create_task(sess.act({"type": "run_block", "target_block": "ghost"}))
    await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "ok": False, "status": "rejected",
        "detail": "no block named ghost is on the canvas",
    })
    observation = await task

    assert observation["ok"] is False
    assert "ghost" in observation["detail"]


# ── The two clocks ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_silent_client_times_out_into_a_readable_failure():
    """An older app that does not understand agent_action simply never replies."""
    sess, _ws = _session()
    request_id, fut = await sess._dispatch_client({"type": "agent_action", "action": {}})

    result = await sess._await_client(
        request_id, fut, idle_timeout=0.05, hard_timeout=0.05, what="test action",
    )

    assert result["ok"] is False
    assert "did not respond to the test action" in result["detail"]
    assert request_id not in sess._pending          # and nothing is left leaking


@pytest.mark.asyncio
async def test_progress_re_arms_the_idle_deadline():
    """A long run outlives any idle window; heartbeats are what keep it alive.

    Without this, no single timeout could serve both a 200 ms port read and a
    90-minute place-and-route.
    """
    sess, ws = _session()
    request_id, fut = await sess._dispatch_client({"type": "agent_action", "action": {}})

    async def heartbeat_then_finish():
        # Six beats over ~0.6s, each well inside the 0.3s idle window but
        # together far longer than it. Windows' timer granularity is ~16ms, so
        # the margins here are deliberately generous.
        for _ in range(6):
            await asyncio.sleep(0.1)
            await sess._dispatch_text(json.dumps({
                "type": "agent_action_progress",
                "request_id": request_id, "sub_status": "Round 2 of 3...",
            }))
        await sess._dispatch_text(json.dumps({
            "type": "agent_action_result", "request_id": request_id,
            "ok": True, "status": "done", "observation": {"status": "Success"},
        }))

    beat = asyncio.create_task(heartbeat_then_finish())
    result = await sess._await_client(
        request_id, fut, idle_timeout=0.3, hard_timeout=10.0, what="long run",
    )
    await beat

    assert result["ok"] is True
    assert result["status"] == "Success"


@pytest.mark.asyncio
async def test_the_hard_timeout_is_never_re_armed():
    """Heartbeats prove liveness, not progress. Something has to end an endless run."""
    sess, _ws = _session()
    request_id, fut = await sess._dispatch_client({"type": "agent_action", "action": {}})

    async def forever():
        for _ in range(200):
            await asyncio.sleep(0.02)
            await sess._dispatch_text(json.dumps({
                "type": "agent_action_progress", "request_id": request_id,
            }))

    beat = asyncio.create_task(forever())
    result = await sess._await_client(
        request_id, fut, idle_timeout=10.0, hard_timeout=0.3, what="stuck run",
    )
    beat.cancel()

    assert result["ok"] is False
    assert "did not respond" in result["detail"]


@pytest.mark.asyncio
async def test_started_is_an_acknowledgement_not_an_outcome():
    """"started" means the run began; resolving on it would observe nothing."""
    sess, ws = _session()
    request_id, fut = await sess._dispatch_client({"type": "agent_action", "action": {}})

    await sess._dispatch_text(json.dumps({
        "type": "agent_action_result", "request_id": request_id, "status": "started",
    }))
    assert not fut.done()

    await sess._dispatch_text(json.dumps({
        "type": "agent_action_result", "request_id": request_id,
        "ok": True, "status": "done", "observation": {"status": "Success"},
    }))
    result = await sess._await_client(request_id, fut, idle_timeout=1, hard_timeout=1)
    assert result["status"] == "Success"


# ── A run does not hold the canvas lock ──────────────────────────────────────


@pytest.mark.asyncio
async def test_a_long_run_does_not_block_other_canvas_edits():
    """Holding the lock across a 90-minute route would freeze every other agent."""
    sess, ws = _session()

    run = asyncio.create_task(sess.act({"type": "run_block", "target_block": "sim"},
                                       expensive=True))
    await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "status": "started",
    })

    # While that run is still outstanding, an ordinary edit must get through.
    edit = asyncio.create_task(sess.act({"type": "set_port_value", "target_block": "A",
                                         "direction": "input", "port_name": "p",
                                         "value": "v"}))
    edit_msg = None
    for _ in range(200):
        edit_msg = next((m for m in ws.sent if m["type"] == "agent_action"
                         and m["action"]["type"] == "set_port_value"), None)
        if edit_msg:
            break
        await asyncio.sleep(0.005)
    assert edit_msg, "the edit never reached the client - the run is holding the lock"

    await sess._dispatch_text(json.dumps({
        "type": "agent_action_result", "request_id": edit_msg["request_id"],
        "ok": True, "status": "done", "observation": {},
    }))
    assert (await edit)["ok"] is True

    run_msg = next(m for m in ws.sent if m["type"] == "agent_action"
                   and m["action"]["type"] == "run_block")
    await sess._dispatch_text(json.dumps({
        "type": "agent_action_result", "request_id": run_msg["request_id"],
        "ok": True, "status": "done", "observation": {"status": "Success"},
    }))
    assert (await run)["status"] == "Success"


# ── Disconnect ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_resolves_outstanding_requests_so_loops_unwind():
    """The failure this prevents is a loop parked forever on a dead socket."""
    sess, _ws = _session()
    request_id, fut = await sess._dispatch_client({"type": "agent_action", "action": {}})

    waiting = asyncio.create_task(
        sess._await_client(request_id, fut, idle_timeout=30, hard_timeout=30)
    )
    await asyncio.sleep(0)
    await sess.cleanup()
    result = await asyncio.wait_for(waiting, timeout=1.0)

    assert result["ok"] is False
    assert "disconnected" in result["detail"]
    assert sess._pending == {}


@pytest.mark.asyncio
async def test_cleanup_stops_every_agent():
    sess, _ws = _session()
    await sess._start_block_agent({
        "block_id": "b1", "block_name": "sync_fifo", "block_type": "code_hdl",
    })
    assert "b1" in sess._agents

    await sess.cleanup()

    assert sess._agents == {}
    assert sess._agent_tasks == {}


# ── Agent lifecycle ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_registers_an_agent_and_honours_the_policy():
    sess, _ws = _session()

    await sess._start_block_agent({
        "block_id": "b1", "block_name": "sync_fifo", "block_type": "CODE_HDL",
        "model": "claude-opus-5", "goal": "hit 500 MHz",
        "autonomy": {"delete": False, "run_expensive": False},
    })

    agent = sess._agents["b1"]
    assert agent.block_type == "code_hdl"           # normalized
    assert agent.model == "claude-opus-5"
    assert agent.goal == "hit 500 MHz"
    assert agent.policy.may_delete is False
    assert agent.policy.may_run_expensive is False
    assert agent.policy.may_create is True          # unnamed keys keep their default
    await sess.cleanup()


@pytest.mark.asyncio
async def test_start_without_a_block_is_an_error_not_a_crash():
    sess, ws = _session()
    await sess._start_block_agent({"block_id": "", "block_name": ""})
    assert _last(ws, "error")["message"].startswith("start_block_agent needs")
    assert sess._agents == {}


@pytest.mark.asyncio
async def test_stop_reports_the_agent_as_stopped():
    sess, ws = _session()
    await sess._start_block_agent({
        "block_id": "b1", "block_name": "sync_fifo", "block_type": "code_hdl",
    })

    await sess._stop_block_agent("b1")

    assert sess._agents == {}
    assert _last(ws, "agent_state")["state"] == "stopped"


@pytest.mark.asyncio
async def test_stop_with_no_block_id_stops_all_of_them():
    sess, _ws = _session()
    for i in (1, 2, 3):
        await sess._start_block_agent({
            "block_id": f"b{i}", "block_name": f"n{i}", "block_type": "code_hdl",
        })

    await sess._stop_block_agent("")

    assert sess._agents == {}


@pytest.mark.asyncio
async def test_pressing_agent_again_replaces_the_running_agent():
    """Matches the client's existing rule that a second Run replaces the first."""
    sess, _ws = _session()
    payload = {"block_id": "b1", "block_name": "sync_fifo", "block_type": "code_hdl"}

    await sess._start_block_agent(payload)
    first = sess._agents["b1"]
    await sess._start_block_agent(payload)
    second = sess._agents["b1"]

    assert first is not second
    assert len(sess._agents) == 1
    await sess.cleanup()


# ── Questions and instructions ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_user_waits_for_the_answer():
    sess, ws = _session()

    task = asyncio.create_task(sess.ask("b1", "Delete the stale block?"))
    await _reply_to(sess, ws, "agent_question", {
        "type": "agent_user_message", "text": "no, keep it",
    })
    answer = await task

    assert _last(ws, "agent_question")["question"] == "Delete the stale block?"
    assert answer == "no, keep it"


@pytest.mark.asyncio
async def test_an_instruction_for_a_block_with_no_agent_says_so():
    sess, ws = _session()
    await sess._on_agent_user_message({"block_id": "nope", "text": "do a thing"})
    assert "no active agent" in _last(ws, "error")["message"]


@pytest.mark.asyncio
async def test_an_instruction_to_a_busy_agent_is_refused_rather_than_dropped():
    """Steering a live loop would mutate a transcript another task is reading."""
    sess, ws = _session()
    await sess._start_block_agent({
        "block_id": "b1", "block_name": "sync_fifo", "block_type": "code_hdl",
    })
    sess._agents["b1"].state = "running"

    await sess._on_agent_user_message({"block_id": "b1", "text": "also do this"})

    message = _last(ws, "error")["message"]
    assert "still working" in message
    assert "Stop" in message
    await sess.cleanup()


@pytest.mark.asyncio
async def test_an_instruction_to_an_idle_agent_resumes_the_same_loop():
    """A follow-up continues the conversation instead of starting a new one.

    The user sees one unbroken chat, so the model must too. Rebuilding the loop
    here would hand it an empty transcript and drop the client's autonomy along
    with it.
    """
    sess, _ws = _session()
    await sess._start_block_agent({
        "block_id": "b1", "block_name": "sync_fifo", "block_type": "code_hdl",
        "model": "claude-opus-5",
        "autonomy": {"run": False, "run_expensive": False},
    })
    first = sess._agents["b1"]
    first.state = "finished"
    first.steps_used = 7
    first._messages.append({"role": "assistant", "content": "made the fifo synchronous"})
    first._seq = 12

    await sess._on_agent_user_message({"block_id": "b1", "text": "now optimise for area"})

    agent = sess._agents["b1"]
    assert agent is first                            # the same loop, not a replacement
    assert agent.model == "claude-opus-5"
    assert agent.policy.may_run is False             # autonomy survived the follow-up
    assert agent.steps_used == 0                     # a fresh budget for a fresh task
    assert agent._seq == 12                          # but seq stays monotonic
    assert any("made the fifo synchronous" in str(m.get("content", ""))
               for m in agent._messages)             # and it remembers
    assert sess._team_goal == "now optimise for area"
    await sess.cleanup()


@pytest.mark.asyncio
async def test_the_action_envelope_names_the_agent_that_asked():
    """The client reads the owner off the envelope, before it parses the action.

    It refuses anything it cannot attribute to a block whose agent it still
    believes is running -- so an unstamped envelope means every action comes
    back rejected and the agent burns its whole budget on failures.
    """
    sess, ws = _session()
    action = {"type": "set_port_value", "target_block": "A", "direction": "input",
              "port_name": "top", "value": "counter", "agent_block_id": "b1"}

    task = asyncio.create_task(sess.act(action))
    sent = await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "ok": True, "status": "done", "detail": "ok",
    })
    await task

    assert sent["agent_block_id"] == "b1"
    assert sent["action"]["agent_block_id"] == "b1"
    await sess.cleanup()


@pytest.mark.asyncio
async def test_two_agents_can_be_waiting_on_the_user_at_once():
    """Each question is answered by request_id, so order does not matter.

    The panel used to keep ONE pending question, so a second agent's question
    overwrote the first: the first agent then waited out its full timeout, and
    whatever the user typed next was recorded against whichever tab was visible
    rather than the agent it was sent to. The server side has to be unambiguous
    for the fix to hold, so pin it.
    """
    sess, ws = _session()

    ask_a = asyncio.create_task(sess.ask("b1", "how wide?"))
    ask_b = asyncio.create_task(sess.ask("b2", "how deep?"))
    for _ in range(200):
        questions = [m for m in ws.sent if m["type"] == "agent_question"]
        if len(questions) == 2:
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError("both agents should have asked")

    by_block = {q["block_id"]: q["request_id"] for q in questions}
    assert set(by_block) == {"b1", "b2"}          # each names its own block

    # Answered out of order, and each answer reaches the agent that asked.
    await sess._dispatch_text(json.dumps({
        "type": "agent_user_message", "block_id": "b2",
        "text": "16 deep", "request_id": by_block["b2"],
    }))
    await sess._dispatch_text(json.dumps({
        "type": "agent_user_message", "block_id": "b1",
        "text": "8 wide", "request_id": by_block["b1"],
    }))

    assert await ask_a == "8 wide"
    assert await ask_b == "16 deep"
    await sess.cleanup()


# ── Policy parsing ───────────────────────────────────────────────────────────


def test_a_malformed_autonomy_object_falls_back_to_defaults():
    """A bad policy must never be the reason the Agent button does nothing."""
    assert AgentPolicy.from_client(None) == AgentPolicy()
    assert AgentPolicy.from_client("nonsense") == AgentPolicy()
    assert AgentPolicy.from_client({"delete": "yes please"}).may_delete is False
    assert AgentPolicy.from_client({"unknown_key": True}) == AgentPolicy()


# ── A failed run must arrive as a failure ────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_run_is_not_reported_as_a_success():
    """The observation must not be able to overwrite the client's verdict.

    BlockObservation::toJson writes its own "ok", so splatting the observation
    over the envelope let a run that ended in Error arrive as ok=true. The loop
    marks a tool result as an error from exactly that flag, so the agent was told
    its failed run had succeeded -- and had no reason to repair anything.
    """
    sess, ws = _session()

    task = asyncio.create_task(sess.act({"type": "run_block", "target_block": "sim"}))
    await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "ok": False, "status": "done",
        "detail": "the run ended in Error",
        "observation": {"ok": True, "block": "sim", "status": "Error",
                        "error": "3 assertions failed", "ports": []},
    })
    observation = await task

    assert observation["ok"] is False
    assert observation["status"] == "Error"
    assert observation["error"] == "3 assertions failed"
    assert "Error" in observation["detail"]


@pytest.mark.asyncio
async def test_a_successful_run_still_arrives_as_a_success():
    sess, ws = _session()

    task = asyncio.create_task(sess.act({"type": "run_block", "target_block": "sim"}))
    await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "ok": True, "status": "done",
        "observation": {"ok": True, "block": "sim", "status": "Success", "ports": []},
    })

    assert (await task)["ok"] is True


@pytest.mark.asyncio
async def test_a_long_run_is_given_the_long_ceiling_whoever_owns_the_block():
    """A code_hdl agent running its downstream verilator is the prescribed flow.

    `expensive` answers a POLICY question (may this agent spend money) and is
    false for a neighbour's block. Using it to pick the timeout too abandoned
    exactly those runs after ten minutes while they were still going.
    """
    sess, ws = _session()
    seen: dict = {}

    original = sess._await_client

    async def spy(request_id, fut, *, idle_timeout, hard_timeout, what="request"):
        seen["hard_timeout"] = hard_timeout
        return await original(request_id, fut, idle_timeout=idle_timeout,
                              hard_timeout=hard_timeout, what=what)

    sess._await_client = spy
    task = asyncio.create_task(
        sess.act({"type": "run_block", "target_block": "fifo_sim"}, expensive=False)
    )
    await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_result", "ok": True, "status": "done", "observation": {},
    })
    await task

    assert seen["hard_timeout"] == 2700.0


@pytest.mark.asyncio
async def test_progress_reaches_the_user_as_a_step():
    """A 90-minute run must not be a blackout in the panel."""
    sess, ws = _session()

    task = asyncio.create_task(
        sess.act({"type": "run_block", "target_block": "sim",
                  "agent_block_id": "b7"}, expensive=True)
    )
    action = await _reply_to(sess, ws, "agent_action", {
        "type": "agent_action_progress", "sub_status": "Placing cells...",
    })
    await asyncio.sleep(0.05)
    await sess._dispatch_text(json.dumps({
        "type": "agent_action_result", "request_id": action["request_id"],
        "ok": True, "status": "done", "observation": {},
    }))
    await task

    steps = [m for m in ws.sent if m["type"] == "agent_step"]
    assert any("Placing cells" in s["text"] for s in steps), (
        "the sub-status never reached the panel"
    )
    assert all(s["block_id"] == "b7" for s in steps)
