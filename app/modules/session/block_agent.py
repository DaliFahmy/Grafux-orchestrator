"""The per-block AI agent: the loop behind a block's Agent button.

Pressing Agent on a block used to open a chat scoped to it and then wait for the
user to type. This module is what turns that into an agent: it plans, calls
canvas tools, **observes what actually happened**, and iterates until the block's
objective is met or its budget runs out.

The observation is the whole point. The chat path emits its canvas changes as a
trailing ``##ACTIONS##`` string that is shipped to the client and forgotten, so
the model never learns whether a single one of them worked. Here every tool call
is a real round-trip through ``CanvasPort.act`` and its result goes back into the
transcript, which is what lets an agent notice that a port is still empty, that a
run failed, or that regenerating changed nothing.

Layering:

* the LOOP is here, and reaches the canvas only through the ``CanvasPort``
  protocol -- so it can be exercised against a fake with no WebSocket, no client
  and no LLM (see tests/test_block_agent_loop.py);
* the TRANSPORT is ``_OrchestratorSession`` in router.py, which implements that
  protocol over the existing ``/ws/session`` socket;
* the TOOL VOCABULARY is canvas_tools.py, shared with the voice and chat paths so
  there is one description of what an AI may do to a canvas, not three.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.constants import (
    BLOCK_AGENT_DEFAULT_SECTION,
    BLOCK_AGENT_SECTION,
    EXPENSIVE_BLOCK_TYPES,
)
from app.core.llm import ToolCall, ToolTurn, call_llm_tools
from app.core.logging import get_logger
from app.modules.session import canvas_tools
from app.prompts import get_system_prompt

log = get_logger("session.block_agent")

# Tools that only READ. Never budget-limited, never blocked by policy, and
# resolved without touching the canvas.
_READ_TOOLS = frozenset({"read_port_value"})

# Tools handled inside the loop rather than sent to the client.
_LIFECYCLE_TOOLS = frozenset({"finish", "ask_user"})

# Tools that start a block running (as opposed to editing it).
_RUN_TOOLS = frozenset({"run_block", "regenerate_block"})


# ── Policy ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentPolicy:
    """What this agent may do, and how much of it.

    Defaults encode the product decision: an agent creates and wires freely
    because those are additive and connections are undoable through the scene's
    undo stack, but it never deletes a block on its own -- deleting removes the
    block's folder and port files from disk, and nothing undoes that.
    """

    may_mutate_ports: bool = True
    may_wire: bool = True
    may_create: bool = True
    may_delete: bool = False
    may_run: bool = True
    may_run_expensive: bool = True

    max_steps: int = 24
    max_wall_clock_s: float = 1800.0
    max_block_runs: int = 6
    max_expensive_runs: int = 2

    @classmethod
    def from_client(cls, raw: Any) -> AgentPolicy:
        """Build a policy from the client's ``autonomy`` object, ignoring junk.

        Unknown keys and wrong types fall back to the default rather than
        raising: a policy is a preference, and a malformed one must not be the
        reason a user's Agent button does nothing.
        """
        if not isinstance(raw, dict):
            return cls()
        defaults = cls()
        def flag(key: str, current: bool) -> bool:
            value = raw.get(key)
            return bool(value) if isinstance(value, bool) else current
        return cls(
            may_mutate_ports=flag("mutate_ports", defaults.may_mutate_ports),
            may_wire=flag("wire", defaults.may_wire),
            may_create=flag("create", defaults.may_create),
            may_delete=flag("delete", defaults.may_delete),
            may_run=flag("run", defaults.may_run),
            may_run_expensive=flag("run_expensive", defaults.may_run_expensive),
        )

    def describes(self) -> str:
        """The policy as prose for the system prompt.

        The model is told what it may do AND what it may not, because a tool that
        is simply absent leaves it with a goal it cannot reach and no vocabulary
        to say so -- at which point it improvises something worse.
        """
        allowed: list[str] = []
        if self.may_mutate_ports:
            allowed.append("read and write port values, and add, rename or remove ports")
        if self.may_wire:
            allowed.append("connect and disconnect wires between blocks")
        if self.may_create:
            allowed.append("create new blocks and load saved ones")
        if self.may_run:
            allowed.append("run and regenerate blocks")
        lines = [
            "You may do the following WITHOUT asking: " + "; ".join(allowed) + "."
            if allowed else "You may not change the canvas.",
        ]
        if not self.may_delete:
            lines.append(
                "You may NOT delete blocks. Deleting a block permanently removes its "
                "folder and all of its port files from disk. If deleting one is "
                "genuinely the right move, call ask_user and let them decide."
            )
        if self.may_run and not self.may_run_expensive:
            lines.append(
                "Running a verilator, yosys, openroad or gpu block provisions a paid "
                "cloud machine, so ask_user before running one."
            )
        lines.append(
            f"You have at most {self.max_steps} steps. Spend them on the objective, "
            "and call finish as soon as it is met."
        )
        return "\n".join(lines)


# ── Steps ────────────────────────────────────────────────────────────────────


@dataclass
class AgentStep:
    """One thing the agent did, as the chat panel renders it."""

    kind: str            # thought | tool_call | tool_result | denied | error | note
    text: str
    tool: dict[str, Any] | None = None


# ── The seam between the loop and the world ──────────────────────────────────


class CanvasPort(Protocol):
    """Everything the loop needs from the outside world.

    Deliberately four methods. The loop is the part worth testing and the part
    most likely to change, so it depends on this rather than on
    ``_OrchestratorSession``, a WebSocket, or a client that may have gone away.
    """

    async def act(self, action: dict[str, Any], *, expensive: bool = False) -> dict[str, Any]:
        """Apply one canvas action and return an observation of what happened."""

    async def read_port(self, target_block: str, direction: str, port_name: str) -> str:
        """Return a port file's full content as text for the model."""

    async def emit(self, message: dict[str, Any]) -> None:
        """Push one message to the client."""

    async def ask(self, block_id: str, question: str) -> str:
        """Ask the user a question and wait for their answer."""

    def diagram_context(self, *, budget: int) -> str:
        """The canvas as model-readable text."""


# ── Prompt assembly ──────────────────────────────────────────────────────────


def objective_section(block_type: str) -> str:
    """The Msg_config section holding this block type's agent objective.

    ``get_system_prompt`` returns ``""`` for a section that does not exist, so a
    missing or misspelled entry would silently produce a blank objective and a
    confidently useless agent. The fallback is explicit for that reason.
    """
    section = BLOCK_AGENT_SECTION.get((block_type or "").strip().lower())
    if section:
        text = get_system_prompt(section)
        if text:
            return text
        log.warning("block_agent_section_empty", block_type=block_type, section=section)
    return get_system_prompt(BLOCK_AGENT_DEFAULT_SECTION)


def build_system_prompt(
    *,
    block_name: str,
    block_type: str,
    policy: AgentPolicy,
    goal: str,
) -> str:
    """The FROZEN half of the prompt -- byte-identical for every step of a run.

    Nothing volatile belongs here. Caching is a prefix match, so folding the
    canvas render into this block would invalidate the cache on every single step
    of an agent whose entire job is to change the canvas: it would look like
    caching without being it. Volatile context rides as a mid-conversation system
    message instead (see ``_context_message``).
    """
    shell = get_system_prompt("block_agent")
    return (
        shell
        .replace("{BLOCK_NAME}", block_name)
        .replace("{BLOCK_TYPE}", block_type)
        .replace("{BLOCK_OBJECTIVE}", objective_section(block_type))
        .replace("{AUTONOMY}", policy.describes())
        .replace("{TEAM_GOAL}", goal.strip() or "(none given - work to your block's objective.)")
    )


# ── The loop ─────────────────────────────────────────────────────────────────


@dataclass
class BlockAgentLoop:
    """One block's agent.

    Owns its own transcript. The session's chat history is deliberately not
    reused: an agent's transcript is a working log of tool calls and
    observations, and interleaving it with the user's conversation would make
    both worse.
    """

    block_id: str
    block_name: str
    block_type: str
    canvas: CanvasPort
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    model: str | None = None
    goal: str = ""

    state: str = "idle"          # idle | running | finished | stopped | error
    summary: str = ""
    goal_met: bool = False
    steps_used: int = 0
    runs_used: int = 0
    expensive_runs_used: int = 0

    _messages: list[dict[str, Any]] = field(default_factory=list)
    _seq: int = 0
    _started_at: float = 0.0
    _stopping: bool = False
    _detached: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Ask the loop to stop at its next checkpoint.

        Cooperative rather than immediate: a hard cancel mid-round-trip would
        leave a block running on the canvas with nobody listening for its result.
        """
        self._stopping = True

    def detach(self) -> None:
        """Cut this loop off from the client for good.

        A stopped or replaced loop still has to unwind: ``run()``'s ``finally``
        reports its final state, and a cancelled round-trip can raise its way out
        through another ``_step``. Once the session has handed this block to a new
        loop, those trailing messages are lies about the agent the user is now
        watching -- a stale "Stopped." verdict that dismantles a live agent's UI.
        Detaching drops them at the source, so the client needs no epoch guard of
        its own. Stopping is cooperative; this is not.
        """
        self._stopping = True
        self._detached = True

    def resume_budget(self) -> None:
        """Ready this loop for a fresh instruction, keeping its transcript.

        The budget is per instruction, the transcript is per conversation -- two
        of the three clocks, kept apart. Rebuilding the loop instead (which is
        what a follow-up used to do) throws away ``_messages``, so the model
        forgets a conversation the user can still read on screen, and restarts
        ``_seq`` under a client that is de-duplicating on it.
        """
        self._stopping = False
        self.steps_used = 0
        self.runs_used = 0
        self.expensive_runs_used = 0
        self.goal_met = False
        self.summary = ""

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at if self._started_at else 0.0

    def _budget_exhausted(self) -> str:
        if self.steps_used >= self.policy.max_steps:
            return f"the {self.policy.max_steps}-step budget"
        if self.elapsed >= self.policy.max_wall_clock_s:
            return f"the {int(self.policy.max_wall_clock_s)}-second time budget"
        return ""

    # ── reporting ────────────────────────────────────────────────────────────

    async def _step(self, kind: str, text: str, tool: dict[str, Any] | None = None) -> None:
        if self._detached:
            return
        self._seq += 1
        await self.canvas.emit({
            "type": "agent_step",
            "block_id": self.block_id,
            "seq": self._seq,
            "kind": kind,
            "text": text,
            "tool": tool or {},
        })

    async def _report_state(self) -> None:
        if self._detached:
            return
        await self.canvas.emit({
            "type": "agent_state",
            "block_id": self.block_id,
            "state": self.state,
            "summary": self.summary,
            "goal": self.goal,
            "goal_met": self.goal_met,
            "steps_used": self.steps_used,
            "steps_max": self.policy.max_steps,
        })

    # ── prompt ───────────────────────────────────────────────────────────────

    def _context_message(self) -> dict[str, Any]:
        """The volatile half of the prompt, as a mid-conversation system message.

        Kept out of the cached top-level system block on purpose -- see
        ``build_system_prompt``.
        """
        return {
            "role": "system",
            "content": (
                "CURRENT CANVAS (refreshed each step; your block is marked ACTIVE):\n"
                + (self.canvas.diagram_context(budget=6000) or "(the canvas is empty)")
            ),
        }

    def _tool_names(self) -> list[str]:
        """The tools this agent is allowed to call, per its policy.

        A forbidden tool is omitted rather than declared-and-refused: every
        refusal costs a step, and a model that keeps being told "no" starts
        improvising equivalents out of the tools it does have.
        """
        names = ["read_port_value", "finish", "ask_user"]
        if self.policy.may_mutate_ports:
            names += ["set_port_value", "add_port", "remove_port", "rename_port",
                      "open_port", "set_description"]
        if self.policy.may_wire:
            names += ["connect_ports", "disconnect_ports"]
        if self.policy.may_create:
            names += ["create_block", "load_block"]
        if self.policy.may_delete:
            names.append("delete_block")
        if self.policy.may_run:
            names += ["run_block", "regenerate_block"]
        return names

    # ── run ──────────────────────────────────────────────────────────────────

    async def run(self, instruction: str = "") -> None:
        """Drive the agent to completion, a budget, or a stop."""
        self._started_at = time.monotonic()
        self.state = "running"
        await self._report_state()

        opening = instruction.strip() or (
            f'Work on the "{self.block_name}" block ({self.block_type}). '
            "Start by reading what it and its neighbours already contain, then do "
            "whatever is needed to fulfil its objective well."
        )
        self._messages.append({"role": "user", "content": opening})

        try:
            await self._loop()
        except Exception as exc:                      # never let a loop die silently
            log.error("block_agent_failed", block=self.block_name, error=str(exc))
            self.state = "error"
            self.summary = f"The agent stopped after an internal error: {exc}"
            await self._step("error", self.summary)
        finally:
            if self.state == "running":
                self.state = "stopped"
            await self._report_state()

    async def _loop(self) -> None:
        system = build_system_prompt(
            block_name=self.block_name,
            block_type=self.block_type,
            policy=self.policy,
            goal=self.goal,
        )
        declarations = canvas_tools.to_declarations(self._tool_names())

        while True:
            if self._stopping:
                self.state = "stopped"
                self.summary = "Stopped at your request."
                await self._step("note", self.summary)
                return

            exhausted = self._budget_exhausted()
            if exhausted:
                await self._wrap_up(exhausted, system)
                return

            self.steps_used += 1
            turn = await call_llm_tools(
                system,
                [*self._messages, self._context_message()],
                declarations,
                model=self.model,
            )
            await self._announce_model(turn)

            if turn.stop_reason == "refusal":
                self.state = "error"
                self.summary = turn.text
                await self._step("error", turn.text)
                return

            if turn.text:
                await self._step("thought", turn.text)

            if not turn.tool_calls:
                # The model stopped talking without calling finish. Treat that as
                # done rather than looping on an empty turn, but record that it
                # never gave a verdict.
                self.state = "finished"
                self.summary = turn.text or "The agent stopped without a summary."
                self.goal_met = False
                return

            self._messages.append({
                "role": "assistant",
                "content": turn.text,
                "tool_calls": turn.tool_calls,
                # Kept so the Anthropic branch can replay this turn unchanged --
                # see ToolTurn.raw_content. Without it the turn is rebuilt from
                # text + tool_calls, which drops the thinking blocks the API
                # requires back verbatim.
                "raw_content": turn.raw_content,
            })

            results: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                if call.name == "finish":
                    await self._finish(call)
                    return
                results.append(await self._dispatch(call))
            self._messages.extend(results)

    async def _announce_model(self, turn: ToolTurn) -> None:
        """Say which model actually ran, once, on the first step.

        The Agent dropdown offers ids the server cannot route natively (the
        gemini ones fall back to OpenAI). The fallback is fine; being unable to
        tell it happened is not.
        """
        if self.steps_used != 1:
            return
        requested = (self.model or "").strip()
        if requested and requested != turn.model:
            await self._step(
                "note",
                f"{requested} is not available here - running on {turn.model} instead.",
            )

    async def _finish(self, call: ToolCall) -> None:
        args = call.arguments
        self.state = "finished"
        self.summary = str(args.get("summary", "")).strip() or "Done."
        self.goal_met = str(args.get("goal_met", "")).strip().lower() in ("true", "yes", "1")
        blocking = str(args.get("blocking", "")).strip()
        text = self.summary
        if not self.goal_met and blocking:
            text += f"\n\nStill outstanding: {blocking}"
        await self._step("note", text)

    async def _wrap_up(self, budget: str, system: str) -> None:
        """Spend one last, tool-less call saying what happened.

        A budget that simply halts the loop leaves the user with a canvas that
        changed for no stated reason. The agent always gets to report.
        """
        await self._step("note", f"Reached {budget}; summarising what I did.")
        self._messages.append({
            "role": "user",
            "content": (
                f"You have reached {budget} and must stop now. Do not call any tools. "
                "In a few sentences: what you changed, what you verified, what is "
                "still unproven, and the single next step you would take."
            ),
        })
        try:
            turn = await call_llm_tools(system, self._messages, [], model=self.model)
            self.summary = turn.text.strip() or f"Reached {budget}."
        except Exception as exc:
            log.warning("block_agent_wrapup_failed", block=self.block_name, error=str(exc))
            self.summary = f"Reached {budget} (and could not summarise: {exc})."
        self.state = "finished"
        self.goal_met = False
        await self._step("note", self.summary)

    # ── tool dispatch ────────────────────────────────────────────────────────

    def _result(self, call: ToolCall, content: str, *, is_error: bool = False) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": content,
            "is_error": is_error,
        }

    async def _dispatch(self, call: ToolCall) -> dict[str, Any]:
        """Run one tool call and return the transcript entry describing its result."""
        await self._step("tool_call", self._describe(call),
                         {"name": call.name, "args": call.arguments})

        if call.name == "ask_user":
            return await self._dispatch_ask(call)
        if call.name in _READ_TOOLS:
            return await self._dispatch_read(call)

        denial = self._deny(call)
        if denial:
            await self._step("denied", denial)
            return self._result(call, denial, is_error=True)

        action = canvas_tools.function_call_to_action(
            call.name, call.arguments, [{"name": self.block_name}]
        )
        if action is None:
            return self._result(call, f"{call.name} is not a canvas action.", is_error=True)

        # Who is asking. The client refuses an action it cannot attribute to a
        # block whose agent it still believes is running, and picks the target
        # block from this when the tool call named none -- without it, one
        # agent's edit lands on whichever tab the user happens to be looking at.
        action["agent_block_id"] = self.block_id

        expensive = self._is_expensive(call, action)
        if call.name in _RUN_TOOLS:
            self.runs_used += 1
            if expensive:
                self.expensive_runs_used += 1

        observation = await self.canvas.act(action, expensive=expensive)
        text = _render_observation(observation)
        ok = bool(observation.get("ok", True))
        await self._step("tool_result", text)
        return self._result(call, text, is_error=not ok)

    async def _dispatch_ask(self, call: ToolCall) -> dict[str, Any]:
        question = str(call.arguments.get("question", "")).strip()
        if not question:
            return self._result(call, "ask_user needs a question.", is_error=True)
        answer = await self.canvas.ask(self.block_id, question)
        await self._step("tool_result", f"The user said: {answer}" if answer
                         else "The user did not answer.")
        return self._result(call, answer or "(no answer - assume no and continue.)")

    async def _dispatch_read(self, call: ToolCall) -> dict[str, Any]:
        args = call.arguments
        target = str(args.get("target_block", "")).strip() or self.block_name
        direction = str(args.get("direction", "")).strip() or "output"
        port = str(args.get("port_name", "")).strip()
        if not port:
            return self._result(call, "read_port_value needs a port_name.", is_error=True)
        content = await self.canvas.read_port(target, direction, port)
        await self._step("tool_result", f"Read {target}.{port} ({len(content)} chars).")
        return self._result(call, content or "(the port is empty)")

    def _deny(self, call: ToolCall) -> str:
        """Why this call may not proceed, or an empty string.

        Enforced here rather than trusted to the prompt. Budget rules in
        particular are exactly the ones a model under pressure talks itself out
        of, and a denial the model can read is more useful than a silent drop.
        """
        if call.name in _RUN_TOOLS:
            if not self.policy.may_run:
                return "You may not run blocks."
            if self.runs_used >= self.policy.max_block_runs:
                return (
                    f"You have used all {self.policy.max_block_runs} runs. Finish with "
                    "what you have, and say what is still unverified."
                )
            if self._is_expensive(call, None):
                if not self.policy.may_run_expensive:
                    return (
                        "That block runs on a paid cloud machine and you may not start "
                        "one. Call ask_user if it needs to run."
                    )
                if self.expensive_runs_used >= self.policy.max_expensive_runs:
                    return (
                        f"You have used all {self.policy.max_expensive_runs} cloud runs. "
                        "Report what you have rather than starting another."
                    )
        if call.name == "delete_block" and not self.policy.may_delete:
            return (
                "Deleting a block is permanent and is not yours to do. Call ask_user "
                "if you believe it should go."
            )
        return ""

    def _is_expensive(self, call: ToolCall, action: dict[str, Any] | None) -> bool:
        """True when this call would start a paid cloud run.

        The type is taken from the agent's own block when the call targets it,
        which is the common case; a call aimed elsewhere is judged conservatively
        as ordinary, since only the client knows another block's real type.
        """
        if call.name not in _RUN_TOOLS:
            return False
        target = str((action or call.arguments).get("target_block", "")).strip()
        if target and target != self.block_name:
            return False
        return self.block_type in EXPENSIVE_BLOCK_TYPES

    def _describe(self, call: ToolCall) -> str:
        """A one-line, human-readable rendering of a tool call for the chat panel."""
        a = call.arguments
        if call.name in ("run_block", "regenerate_block"):
            verb = "Running" if call.name == "run_block" else "Regenerating"
            return f'{verb} "{a.get("target_block") or self.block_name}"'
        if call.name == "set_port_value":
            return f'Setting {a.get("direction", "")} port "{a.get("port_name", "")}"'
        if call.name == "read_port_value":
            return f'Reading {a.get("target_block") or self.block_name}.{a.get("port_name", "")}'
        if call.name == "connect_ports":
            return (f'Wiring {a.get("from_block", "")}.{a.get("from_port", "")} to '
                    f'{a.get("to_block", "")}.{a.get("to_port", "")}')
        if call.name == "create_block":
            return f'Creating a {a.get("block_type", "")} block "{a.get("block_name", "")}"'
        if call.name == "ask_user":
            return "Asking you a question"
        return call.name.replace("_", " ")


# ── Observation rendering ────────────────────────────────────────────────────


def _render_observation(observation: dict[str, Any]) -> str:
    """Turn the client's observation into the text the model reads back.

    Port values keep the same truncation wording ``render_port`` uses in the
    canvas context, so the model already knows that a truncated value means
    "call read_port_value if you need the rest" -- one convention, learned once.
    """
    if not observation.get("ok", True):
        detail = str(observation.get("detail", "")).strip()
        return f"FAILED: {detail or 'the action could not be applied.'}"

    lines: list[str] = []
    detail = str(observation.get("detail", "")).strip()
    if detail:
        lines.append(detail)

    status = str(observation.get("status", "")).strip()
    if status:
        block = str(observation.get("block", "")).strip()
        line = f"{block} is now {status}" if block else f"Status: {status}"
        sub = str(observation.get("sub_status", "")).strip()
        if sub:
            line += f" ({sub})"
        lines.append(line)

    changed = [p for p in (observation.get("ports") or [])
               if isinstance(p, dict) and p.get("changed")]
    if changed:
        lines.append("Ports that changed:")
        for port in changed:
            value = str(port.get("value", "")) or "(empty)"
            line = f'  [{port.get("direction", "out")}] {port.get("name", "")}: {value}'
            if port.get("truncated"):
                line += (
                    f' ...(truncated: showing {len(value)} of '
                    f'{port.get("full_length", "?")} chars - call read_port_value '
                    "to read the full file)"
                )
            lines.append(line)
    elif status:
        lines.append("No output port changed.")

    error = str(observation.get("error", "")).strip()
    if error:
        lines.append(f"Reported error: {error}")

    return "\n".join(lines) if lines else "Done."


# Type alias used by the session when registering a loop factory.
LoopFactory = Callable[..., Awaitable[None]]
