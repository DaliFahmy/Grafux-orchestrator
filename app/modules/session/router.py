from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.core.security import verify_jwt
from app.modules.session.canvas_context import (
    READ_PORT_TOOL,
)
from app.modules.session.canvas_context import (
    format_library_entry as _format_library_entry,
)
from app.modules.session.canvas_context import (
    parse_actions_tag as _parse_actions_tag,
)
from app.modules.session.canvas_context import (
    render_canvas_context as _render_canvas_context,
)
from app.modules.session.canvas_context import (
    render_port as _render_port,  # noqa: F401 — re-exported for tests
)
from app.modules.session.block_agent import AgentPolicy, BlockAgentLoop
from app.modules.session.enrichment import (
    enrich_actions_streaming,
    is_enrichable,
)

log = get_logger("session.router")

router = APIRouter()

# What this server supports, advertised on session_ready. Add a name here when a
# capability lands, so a client can tell "not deployed yet" from "broken".
SESSION_FEATURES: list[str] = ["block_agent"]


async def _send(websocket: WebSocket, data: dict[str, Any]) -> None:
    try:
        await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        pass  # client already gone — nothing to deliver
    except Exception as exc:
        # Closed/broken socket mid-turn is expected; don't crash the turn.
        log.debug("ws_send_failed", error=str(exc))


@router.websocket("/ws/session")
async def ws_session(
    websocket: WebSocket,
    token: str = Query(default=""),
    project_id: str = Query(default=""),
) -> None:
    """Main orchestrator WebSocket session — text AI and voice relay for the desktop app."""
    payload = await verify_jwt(token)
    if not payload:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    user_id: str = (
        payload.get("user_id")
        or payload.get("sub")
        or payload.get("data", {}).get("user_id")
        or "unknown"
    )
    session_id = str(uuid.uuid4())

    await websocket.accept()
    log.info(
        "session_connected",
        session_id=session_id,
        user_id=user_id,
        project_id=project_id,
    )

    # The feature list is how a client finds out what this server can actually
    # do. Without it an app newer than its server sends start_block_agent, the
    # server drops it as an unknown type, and BOTH ends stay silent -- which is
    # exactly how "the Agent button does nothing" became so hard to diagnose.
    await _send(websocket, {
        "type": "session_ready",
        "session_id": session_id,
        "features": SESSION_FEATURES,
    })

    session = _OrchestratorSession(
        websocket=websocket,
        session_id=session_id,
        user_id=user_id,
        project_id=project_id,
    )
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("session_error", session_id=session_id, error=str(exc))
    finally:
        await session.cleanup()
        log.info("session_disconnected", session_id=session_id)


class _OrchestratorSession:
    """Manages a single client session: text AI turns and optional Gemini voice relay."""

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        project_id: str,
    ) -> None:
        self._ws = websocket
        self._session_id = session_id
        self._user_id = user_id
        self._project_id = project_id
        # Chat history keyed by conversation. The diagram chat and each block
        # agent tab are separate conversations, and the client already keeps them
        # apart (ChatPanelWidget::m_agentHistoriesByBlockId); a single shared list
        # here meant every block tab was answered in the context of every other.
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._canvas_state: dict[str, Any] = {}
        self._active_blocks: list[Any] = []
        self._saved_library: list[Any] = []
        self._voice_active = False
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._voice_task: asyncio.Task | None = None
        # Set when canvas/active-block state changes during a live voice session,
        # so the relay can re-seed Gemini with fresh port data. _last_voice_context
        # holds the last diagram context seeded into the live session.
        self._canvas_dirty = asyncio.Event()
        self._last_voice_context: str | None = None
        # Every outstanding request TO the client, keyed by request_id. Port
        # reads, agent actions and agent questions all resolve through this one
        # map so there is a single timeout policy and a single place that has to
        # be drained when the socket dies.
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # Liveness for long-running requests. The timestamp is authoritative --
        # the event only nudges the waiter awake. Driving the deadline off the
        # event alone loses any beat that lands between one wait returning and
        # the next clear(), and the cost of that is a live run declared dead.
        self._progress: dict[str, asyncio.Event] = {}
        self._last_seen: dict[str, float] = {}
        # The block whose agent owns each outstanding request, and the last
        # sub-status it reported ("Compiling...", "Round 2 of 3..."). Without
        # these a 90-minute run is a blackout: the user watches an idle panel and
        # the agent's transcript records nothing between "started" and the result.
        self._request_owner: dict[str, str] = {}
        self._request_status: dict[str, str] = {}
        # Block agents, keyed by block id, plus the tasks running their loops.
        self._agents: dict[str, BlockAgentLoop] = {}
        self._agent_tasks: dict[str, asyncio.Task] = {}
        # Serializes canvas mutations. The Qt client applies actions on one
        # thread; two agents interleaving connect_ports would push overlapping
        # commands onto its undo stack. Held over dispatch, released before
        # waiting on a long run.
        self._canvas_lock = asyncio.Lock()
        self._team_goal = ""
        # Background enrichment tasks (block content generated after turn_complete is
        # sent, then streamed to the client as action_enriched). Tracked so they can be
        # cancelled on cleanup/stop and don't outlive the session.
        self._enrich_tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        while True:
            msg = await self._ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] != "websocket.receive":
                continue

            raw_bytes = msg.get("bytes")
            raw_text = msg.get("text")

            if raw_bytes:
                if self._voice_active:
                    await self._audio_queue.put(raw_bytes)
            elif raw_text:
                await self._dispatch_text(raw_text)

    async def cleanup(self) -> None:
        # Order matters. Resolving the outstanding requests first lets every
        # agent loop wake up, see "client disconnected" and unwind on its own;
        # cancelling the tasks first would leave them parked on futures that can
        # never be resolved, and cleanup would hang waiting for them.
        self._fail_pending("the client disconnected")
        await self._stop_all_agents()
        await self._cancel_enrichment()
        if self._voice_task and not self._voice_task.done():
            self._voice_active = False
            self._canvas_dirty.set()  # wake the refresh loop so it exits
            self._voice_task.cancel()
            await self._audio_queue.put(None)
            try:
                await asyncio.wait_for(self._voice_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                pass

    def _fail_pending(self, reason: str) -> None:
        """Resolve every outstanding client request as failed."""
        for request_id, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_result({"ok": False, "found": False, "detail": reason, "content": ""})
            self._pending.pop(request_id, None)
        self._progress.clear()
        self._last_seen.clear()
        self._request_owner.clear()
        self._request_status.clear()

    async def _stop_all_agents(self) -> None:
        """Ask every agent to stop, then wait briefly for its loop to unwind."""
        for loop_obj in self._agents.values():
            loop_obj.stop()
        tasks = [t for t in self._agent_tasks.values() if not t.done()]
        self._agent_tasks.clear()
        self._agents.clear()
        for task in tasks:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                task.cancel()

    async def _cancel_enrichment(self) -> None:
        """Cancel any in-flight background enrichment tasks (best-effort, swallow errors)."""
        tasks = list(self._enrich_tasks)
        self._enrich_tasks.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ── Message dispatch ──────────────────────────────────────────────────────

    async def _dispatch_text(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return

        msg_type = data.get("type", "")

        if msg_type == "ping":
            await _send(self._ws, {"type": "pong"})

        elif msg_type == "text_message":
            if data.get("canvas_state"):
                self._canvas_state = data["canvas_state"]
            if data.get("active_blocks"):
                self._active_blocks = data["active_blocks"]
            if data.get("saved_library") is not None:
                self._saved_library = data["saved_library"]
            text = data.get("text", "").strip()
            if text:
                # Older clients send no conversation_id and keep the single
                # shared "diagram" thread they have always had.
                conversation_id = str(data.get("conversation_id") or "diagram")
                await self._run_text_turn(text, conversation_id)

        elif msg_type == "canvas_update":
            self._canvas_state = data.get("canvas_state", self._canvas_state)
            if data.get("saved_library") is not None:
                self._saved_library = data["saved_library"]
            if self._voice_active:
                self._canvas_dirty.set()  # re-seed live voice with fresh port data

        elif msg_type == "set_active_blocks":
            self._active_blocks = data.get("blocks", self._active_blocks)
            if self._voice_active:
                self._canvas_dirty.set()

        elif msg_type == "port_content":
            self._resolve_pending(data.get("request_id", ""), {
                "found": bool(data.get("found")),
                "content": data.get("content", ""),
            })

        elif msg_type == "agent_action_result":
            # "started" is an acknowledgement, not an outcome: the client is
            # saying the run has begun and it will report again at the terminal
            # point. Treated as progress so the idle deadline is re-armed rather
            # than the request being resolved with nothing to observe.
            if str(data.get("status", "done")) == "started":
                self._note_progress(data.get("request_id", ""),
                                    str(data.get("detail", "")))
            else:
                # The observation is merged UNDER the envelope, not over it.
                # BlockObservation::toJson writes its own "ok", so splatting it
                # last let a failed run arrive as ok=true -- and the loop marks a
                # tool result as an error from exactly that flag, so the agent
                # was told its failed run had succeeded.
                observation = data.get("observation") or {}
                self._resolve_pending(data.get("request_id", ""), {
                    **observation,
                    "ok": bool(data.get("ok", True)) and bool(observation.get("ok", True)),
                    "detail": data.get("detail", "") or observation.get("detail", ""),
                    "status_text": data.get("status", ""),
                })

        elif msg_type == "agent_action_progress":
            self._note_progress(data.get("request_id", ""),
                                str(data.get("sub_status", "")))

        elif msg_type == "start_block_agent":
            await self._start_block_agent(data)

        elif msg_type == "stop_block_agent":
            await self._stop_block_agent(str(data.get("block_id", "")))

        elif msg_type == "set_team_goal":
            self._team_goal = str(data.get("goal", "")).strip()

        elif msg_type == "agent_user_message":
            await self._on_agent_user_message(data)

        elif msg_type == "start_voice":
            await self._start_voice()

        elif msg_type == "stop_voice":
            await self._stop_voice()

        else:
            # Never silent. An unknown type means the client is newer than this
            # server, and a dropped message with no trace is indistinguishable
            # from a bug in the feature itself.
            log.warning(
                "unknown_client_message",
                session_id=self._session_id,
                msg_type=msg_type or "(missing)",
            )

    # ── On-demand port read (round-trip to the client) ───────────────────────

    def _resolve_pending(self, request_id: str, payload: dict[str, Any]) -> None:
        """Complete an outstanding client request, if it is still waiting."""
        fut = self._pending.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    def _note_progress(self, request_id: str, sub_status: str = "") -> None:
        """Record that a long-running request is still alive, and what it is doing."""
        if request_id not in self._progress:
            return
        self._last_seen[request_id] = time.monotonic()
        self._progress[request_id].set()
        sub_status = (sub_status or "").strip()
        if not sub_status or sub_status == self._request_status.get(request_id):
            return
        self._request_status[request_id] = sub_status
        # Surface it as a step so the user sees a long run moving. Fire-and-forget:
        # a progress note must never be able to fail the run it is describing.
        owner = self._request_owner.get(request_id, "")
        if owner:
            task = asyncio.create_task(_send(self._ws, {
                "type": "agent_step", "block_id": owner, "seq": 0,
                "kind": "tool_result", "text": sub_status, "tool": {},
            }))
            self._enrich_tasks.add(task)
            task.add_done_callback(self._enrich_tasks.discard)

    async def _dispatch_client(self, message: dict[str, Any],
                               *, owner_block_id: str = "") -> tuple[str, asyncio.Future]:
        """Send a request to the client and return its (request_id, future).

        Split from the wait so a caller can hold the canvas lock over the send
        and release it before waiting, which is what stops one agent\'s
        90-minute place-and-route from freezing every other agent\'s edits.
        """
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut
        self._progress[request_id] = asyncio.Event()
        self._last_seen[request_id] = time.monotonic()
        if owner_block_id:
            self._request_owner[request_id] = owner_block_id
        await _send(self._ws, {**message, "request_id": request_id})
        return request_id, fut

    async def _await_client(
        self,
        request_id: str,
        fut: asyncio.Future,
        *,
        idle_timeout: float,
        hard_timeout: float,
        what: str = "request",
    ) -> dict[str, Any]:
        """Wait for the client\'s reply, tolerating long silences it explains.

        Two clocks, because they answer different questions. ``idle_timeout``
        asks "is the client still there?" and is re-armed by every progress
        message; without that, no single timeout could cover both a 200 ms port
        read and a 90-minute OpenROAD run. ``hard_timeout`` asks "has this gone
        on absurdly long?" and is never re-armed.

        A client too old to understand the message never replies at all, so it
        times out into a readable failure rather than an exception: the same
        graceful degradation the port read has always had.
        """
        progress = self._progress.setdefault(request_id, asyncio.Event())
        self._last_seen.setdefault(request_id, time.monotonic())
        deadline = time.monotonic() + hard_timeout
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                idle_left = idle_timeout - (now - self._last_seen[request_id])
                if idle_left <= 0:
                    break          # silent for a whole idle window: give up
                progress.clear()
                waiter = asyncio.ensure_future(progress.wait())
                try:
                    done, _ = await asyncio.wait(
                        {fut, waiter},
                        timeout=min(idle_left, deadline - now),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    waiter.cancel()
                if fut in done:
                    if fut.cancelled():
                        return {"ok": False, "found": False, "content": "",
                                "detail": "cancelled"}
                    return fut.result()
                # Either progress arrived or the window lapsed; _last_seen decides
                # which, on the next pass. Re-checking it rather than trusting the
                # event is what makes a lost set() harmless instead of fatal.
            log.warning("client_request_timeout", session_id=self._session_id, what=what)
            return {
                "ok": False,
                "found": False,
                "content": "",
                "detail": f"the app did not respond to the {what} in time",
            }
        finally:
            self._pending.pop(request_id, None)
            self._progress.pop(request_id, None)
            self._last_seen.pop(request_id, None)
            self._request_owner.pop(request_id, None)
            self._request_status.pop(request_id, None)

    async def _request_port_content(
        self,
        target_block: str,
        direction: str,
        port_name: str,
        timeout: float = 12.0,
    ) -> dict[str, Any]:
        """Ask the client to read a port file and return {found, content}.

        Port files live on the client\'s filesystem, so the server cannot read
        them directly.
        """
        request_id, fut = await self._dispatch_client({
            "type": "read_port",
            "target_block": target_block,
            "direction": direction,
            "port_name": port_name,
        })
        return await self._await_client(
            request_id, fut,
            idle_timeout=timeout, hard_timeout=timeout,
            what=f"read of {target_block}.{port_name}",
        )

    # -- Block agents -----------------------------------------------------------
    #
    # The session is the agents' hands: it owns the socket, the canvas snapshot
    # and the request map, and exposes exactly the four operations the loop needs
    # (act / read_port / emit / ask) plus the canvas render. Everything else about
    # an agent -- its transcript, budget, tools and prompt -- lives in
    # block_agent.py, where it can be tested without any of this.

    RUN_ACTIONS = ("run_block", "regenerate_block")

    async def _start_block_agent(self, data: dict[str, Any]) -> None:
        """Begin (or restart) the agent for one block."""
        block_id = str(data.get("block_id", "")).strip()
        block_name = str(data.get("block_name", "")).strip()
        if not block_id or not block_name:
            await _send(self._ws, {"type": "error", "message": "start_block_agent needs a block"})
            return

        if data.get("canvas_state"):
            self._canvas_state = data["canvas_state"]
        if data.get("active_blocks"):
            self._active_blocks = data["active_blocks"]
        if data.get("saved_library") is not None:
            self._saved_library = data["saved_library"]
        goal = str(data.get("goal", "")).strip()
        if goal and not self._team_goal:
            self._team_goal = goal

        # A second press replaces the first rather than racing it, matching how
        # a second Run replaces a running verification loop on the client.
        await self._stop_block_agent(block_id, notify=False)

        agent = BlockAgentLoop(
            block_id=block_id,
            block_name=block_name,
            block_type=str(data.get("block_type", "")).strip().lower(),
            canvas=self,
            policy=AgentPolicy.from_client(data.get("autonomy")),
            model=str(data.get("model", "")).strip() or None,
            goal=self._team_goal,
        )
        self._agents[block_id] = agent
        task = asyncio.create_task(agent.run(str(data.get("instruction", ""))))
        self._agent_tasks[block_id] = task
        task.add_done_callback(lambda _t, bid=block_id: self._agent_tasks.pop(bid, None))
        log.info(
            "block_agent_started",
            session_id=self._session_id,
            block=block_name,
            block_type=agent.block_type,
        )

    async def _stop_block_agent(self, block_id: str, *, notify: bool = True) -> None:
        """Stop one agent, or every agent when block_id is empty."""
        targets = [block_id] if block_id else list(self._agents)
        for bid in targets:
            agent = self._agents.pop(bid, None)
            if agent is None:
                continue
            # Detach, not just stop: this loop no longer speaks for the block, and
            # its unwind still wants to report a final state. Whatever it says
            # after this point would describe an agent the user has replaced.
            agent.detach()
            task = self._agent_tasks.pop(bid, None)
            if task is not None and not task.done():
                # Give the loop a moment to notice the flag and report its own
                # state; only then take it apart. A hard cancel first would leave
                # the user with a block that silently stopped mid-run.
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    task.cancel()
            if notify:
                await _send(self._ws, {
                    "type": "agent_state",
                    "block_id": bid,
                    "state": "stopped",
                    "summary": "Stopped.",
                    "goal": self._team_goal,
                    "goal_met": False,
                    "steps_used": agent.steps_used,
                    "steps_max": agent.policy.max_steps,
                })

    async def _on_agent_user_message(self, data: dict[str, Any]) -> None:
        """Route a chat message typed while an agent is active.

        Two jobs, told apart by request_id: answering a question the agent asked,
        or giving a running/finished agent a fresh instruction.
        """
        request_id = str(data.get("request_id", "")).strip()
        text = str(data.get("text", "")).strip()
        if request_id and request_id in self._pending:
            self._resolve_pending(request_id, {"ok": True, "answer": text})
            return

        block_id = str(data.get("block_id", "")).strip()
        agent = self._agents.get(block_id)
        if agent is None:
            await _send(self._ws, {
                "type": "error",
                "message": "That block has no active agent - press Agent on it first.",
            })
            return

        # The first goal typed while agents are running becomes the shared one.
        if text and not self._team_goal:
            self._team_goal = text
        if agent.state == "running":
            # Steering a loop mid-flight would mean mutating a transcript another
            # task is reading. The instruction is refused rather than silently
            # dropped, so the user knows to stop it or wait.
            await _send(self._ws, {
                "type": "error",
                "message": (
                    f'"{agent.block_name}" is still working. Press Stop to interrupt it, '
                    "or wait for it to finish and send this again."
                ),
            })
            return

        self._resume_block_agent(agent, text)

    def _resume_block_agent(self, agent: BlockAgentLoop, instruction: str) -> None:
        """Give a finished agent a fresh instruction without forgetting the chat.

        Rebuilding the loop here (which is what this used to do) started a new
        BlockAgentLoop with an empty transcript, so the model had no memory of a
        conversation the user can still scroll through -- and it silently dropped
        the autonomy the client granted, because the rebuild passed no
        ``autonomy`` and AgentPolicy.from_client(None) falls back to defaults.
        Reusing the object keeps both, and keeps ``_seq`` monotonic for the
        client's de-duplication.

        Pressing the Agent button again still routes through _start_block_agent
        and is still a deliberate, complete reset.
        """
        agent.resume_budget()
        agent.goal = self._team_goal
        block_id = agent.block_id
        task = asyncio.create_task(agent.run(instruction))
        self._agent_tasks[block_id] = task
        task.add_done_callback(lambda _t, bid=block_id: self._agent_tasks.pop(bid, None))
        log.info(
            "block_agent_resumed",
            session_id=self._session_id,
            block=agent.block_name,
            steps_max=agent.policy.max_steps,
        )

    # -- CanvasPort ------------------------------------------------------------

    async def act(self, action: dict[str, Any], *, expensive: bool = False) -> dict[str, Any]:
        """Apply one canvas action on the client and report what happened."""
        is_run = action.get("type") in self.RUN_ACTIONS
        # create_block is enriched HERE rather than through the client's
        # pending/action_enriched dance: the agent is already asynchronous, so it
        # can simply wait, and it then gets an observation naming the ports that
        # were really created instead of a placeholder.
        if action.get("type") == "create_block" and is_enrichable(action):
            try:
                await enrich_actions_streaming([action], self._session_id).__anext__()
            except StopAsyncIteration:
                pass
            except Exception as exc:
                log.warning("agent_enrich_failed", session_id=self._session_id, error=str(exc))

        async with self._canvas_lock:
            request_id, fut = await self._dispatch_client({
                "type": "agent_action",
                "action": action,
                "expects": "result",
                # Stamped by BlockAgentLoop._dispatch_tool. Carried on the
                # envelope as well as in the action so the client can decide
                # whether this agent is still live before it parses the action.
                "agent_block_id": action.get("agent_block_id", ""),
            }, owner_block_id=action.get("agent_block_id", ""))
            if not is_run:
                return await self._await_client(
                    request_id, fut,
                    idle_timeout=30.0, hard_timeout=120.0,
                    what=f'{action.get("type")} action',
                )

        # A run is awaited OUTSIDE the lock. Holding it for the 30-90 minutes an
        # OpenROAD place-and-route can take would block every other edit on the
        # canvas; the client already refuses to start a block that is running, so
        # the run itself needs no lock of its own.
        # EVERY run gets the long ceiling, not just one the agent owns. The
        # `expensive` flag answers a policy question -- may this agent spend money
        # -- and it is false when the target is a NEIGHBOUR, which is precisely the
        # workflow the chip-design prompts prescribe (a code_hdl agent running its
        # downstream verilator). Using it to pick the timeout too abandoned those
        # runs at ten minutes while the block was still going. One flag, one job.
        return await self._await_client(
            request_id, fut,
            idle_timeout=180.0,
            hard_timeout=2700.0,
            what=f'{action.get("type")} on {action.get("target_block", "the block")}',
        )

    async def read_port(self, target_block: str, direction: str, port_name: str) -> str:
        return await self._resolve_read_tool({
            "target_block": target_block,
            "direction": direction,
            "port_name": port_name,
        })

    async def emit(self, message: dict[str, Any]) -> None:
        await _send(self._ws, message)

    async def ask(self, block_id: str, question: str) -> str:
        """Put a question to the user and wait for their reply.

        Reuses the same pending map as every other client round-trip, so a user
        who simply never answers times out into "no answer" and a socket that
        dies resolves it like everything else, instead of parking the agent
        forever.
        """
        request_id, fut = await self._dispatch_client({
            "type": "agent_question",
            "block_id": block_id,
            "question": question,
        })
        result = await self._await_client(
            request_id, fut,
            idle_timeout=600.0, hard_timeout=600.0,
            what="question",
        )
        return str(result.get("answer", "")).strip()

    def diagram_context(self, *, budget: int = 6000) -> str:
        return self._build_diagram_context(budget=budget, lib_limit=30, empty="")

    # ── Diagram context ──────────────────────────────────────────────────────

    def _build_diagram_context(
        self,
        *,
        budget: int | None = None,
        lib_limit: int | None = None,
        empty: str = "No diagram loaded.",
    ) -> str:
        """Render the current canvas + saved library as readable text.

        Blocks and their port-file contents are rendered as text (active blocks
        first) so the model can read and talk about port values directly. Shared
        by the text turn and the voice relay (initial seed + mid-session refresh)
        so both see the same up-to-date `self._canvas_state`.
        """
        parts: list[str] = []
        canvas_text = (
            _render_canvas_context(self._canvas_state, self._active_blocks, budget=budget)
            if budget is not None
            else _render_canvas_context(self._canvas_state, self._active_blocks)
        )
        if canvas_text:
            parts.append("Current canvas blocks:\n" + canvas_text)
        if self._saved_library:
            lib = self._saved_library[:lib_limit] if lib_limit else self._saved_library
            parts.append(
                "Saved block library:\n"
                + "\n".join(_format_library_entry(e) for e in lib)
            )
        return "\n\n".join(parts) if parts else empty

    # ── Action streaming / enrichment ─────────────────────────────────────────

    # Keys an enricher may fill on an action; only these are forwarded as the patch.
    # "memory_mode" is here because the memory enricher NORMALISES it (an unknown
    # mode becomes snapshot) and the ports it scaffolds only make sense alongside
    # the mode the client then persists on the block.
    _PATCH_KEYS = ("output_ports", "input_ports", "code", "language", "memory_mode")

    @staticmethod
    def _stamp_actions(actions: list[Any]) -> None:
        """Give every create_block action a stable id and mark the enrichable ones pending.

        ``action_id`` is the primary key the client correlates the follow-up
        ``action_enriched`` message against; ``pending`` tells the client to drop in a
        placeholder block now and wait for the enriched ports/code.
        """
        for action in actions:
            if not isinstance(action, dict) or action.get("type") != "create_block":
                continue
            action.setdefault("action_id", uuid.uuid4().hex[:8])
            if is_enrichable(action):
                action["pending"] = True

    @classmethod
    def _extract_patch(cls, action: dict[str, Any]) -> dict[str, Any]:
        """The enricher-produced keys to ship to the client (omit anything not generated)."""
        return {k: action[k] for k in cls._PATCH_KEYS if k in action}

    def _spawn_enrichment(self, actions: list[Any]) -> None:
        """Start background enrichment for a turn's actions without blocking the turn."""
        if not any(is_enrichable(a) for a in actions):
            return
        task = asyncio.create_task(self._enrich_and_stream(actions))
        self._enrich_tasks.add(task)
        task.add_done_callback(self._enrich_tasks.discard)

    async def _enrich_and_stream(self, actions: list[Any]) -> None:
        """Enrich each action and push an ``action_enriched`` message as each one finishes."""
        try:
            async for action, status, detail in enrich_actions_streaming(actions, self._session_id):
                await _send(self._ws, {
                    "type": "action_enriched",
                    "action_id": action.get("action_id"),
                    "block_name": action.get("block_name"),
                    "enrichment_status": "failed" if status == "failed" else "ok",
                    "enrichment_error": detail if status == "failed" else "",
                    "patch": self._extract_patch(action),
                })
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let a background task crash silently
            log.warning("enrich_stream_failed", session_id=self._session_id, error=str(exc))

    # ── Text / AI turn ────────────────────────────────────────────────────────

    async def _run_text_turn(self, text: str, conversation_id: str = "diagram") -> None:
        from app.config import get_settings
        settings = get_settings()

        if not settings.openai_api_key:
            await _send(self._ws, {"type": "error", "message": "AI not configured — set OPENAI_API_KEY"})
            return

        history = self._histories.setdefault(conversation_id, [])
        history.append({"role": "user", "content": text})

        from app.prompts import get_system_prompt
        raw_prompt = get_system_prompt("chat_assistant")

        # Build DIAGRAM_CONTEXT from whatever the client has sent.
        diagram_context = self._build_diagram_context(empty="No diagram loaded yet.")
        system_prompt = raw_prompt.replace("{DIAGRAM_CONTEXT}", diagram_context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history,
        ]

        try:
            from app.core.llm import get_async_openai
            client = get_async_openai()

            # Tool-use loop: the model may call read_port_value (resolved via a
            # round-trip to the client) before producing its final answer.
            full_text = ""
            for _iteration in range(4):
                stream = await client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    tools=[READ_PORT_TOOL],
                    stream=True,
                )
                segment_text = ""
                tool_calls: dict[int, dict[str, Any]] = {}
                finish_reason: str | None = None
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta and delta.content:
                        segment_text += delta.content
                        full_text += delta.content
                        await _send(self._ws, {"type": "text_chunk", "text": delta.content})
                    if delta and delta.tool_calls:
                        for tc in delta.tool_calls:
                            slot = tool_calls.setdefault(
                                tc.index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            if tc.id:
                                slot["id"] = tc.id
                            if tc.function and tc.function.name:
                                slot["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                if finish_reason != "tool_calls" or not tool_calls:
                    break

                # Resolve each read_port_value call against the client, then loop.
                assistant_tool_calls = [
                    {
                        "id": c["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    }
                    for i, c in sorted(tool_calls.items())
                ]
                messages.append({
                    "role": "assistant",
                    "content": segment_text or None,
                    "tool_calls": assistant_tool_calls,
                })
                for call in assistant_tool_calls:
                    try:
                        cargs = json.loads(call["function"]["arguments"] or "{}")
                    except Exception:
                        cargs = {}
                    result_text = await self._resolve_read_tool(cargs)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result_text,
                    })

            display_text, actions = _parse_actions_tag(full_text, self._session_id)
            # Stamp ids / mark pending, send the turn immediately, THEN enrich in the
            # background — the client renders placeholder blocks now and fills them in
            # via action_enriched instead of waiting on web grounding / codegen.
            self._stamp_actions(actions)
            history.append({"role": "assistant", "content": display_text})
            await _send(self._ws, {"type": "turn_complete", "full_text": display_text, "actions": actions})
            self._spawn_enrichment(actions)

        except Exception as exc:
            log.error("ai_turn_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": str(exc)})

    async def _resolve_read_tool(self, args: dict[str, Any]) -> str:
        """Resolve a read_port_value tool call into a text result for the model.

        ``target_block`` is optional, as the declaration says: with exactly one
        active block the model is already focused on it and need not name it.
        The voice path always defaulted it here; the text path used to reject the
        call instead, which is the drift that deriving READ_PORT_TOOL exposed.
        """
        from app.modules.session.canvas_tools import _default_target_block

        target_block = str(args.get("target_block", "")).strip()
        if not target_block:
            target_block = _default_target_block(self._active_blocks)
        direction = str(args.get("direction", "")).strip() or "output"
        port_name = str(args.get("port_name", "")).strip()
        if not target_block or not port_name:
            return (
                "Error: port_name is required, and target_block is required "
                "unless exactly one block is active."
            )
        result = await self._request_port_content(target_block, direction, port_name)
        if not result.get("found"):
            return (
                f'Could not read {direction} port "{port_name}" on block '
                f'"{target_block}" (file empty, missing, or client unavailable).'
            )
        return result.get("content", "") or "(the port file is empty)"

    async def _build_read_response(self, fc: Any) -> dict[str, Any]:
        """Build a Gemini function_response for a read_port_value call."""
        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "")
        fc_id = getattr(fc, "id", None) or (fc.get("id") if isinstance(fc, dict) else None)
        raw_args = getattr(fc, "args", None)
        if raw_args is None and isinstance(fc, dict):
            raw_args = fc.get("args", {})
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        content = await self._resolve_read_tool(args)
        entry: dict[str, Any] = {"name": name, "response": {"content": content}}
        if fc_id:
            entry["id"] = fc_id
        return entry

    def _failed_read_response(self, fc: Any) -> dict[str, Any]:
        """Graceful function_response for a read whose resolution raised.

        Keeps the shape ``_build_read_response`` returns so Gemini still gets a reply
        for the call (it errors out otherwise) without leaking the exception.
        """
        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "")
        fc_id = getattr(fc, "id", None) or (fc.get("id") if isinstance(fc, dict) else None)
        entry: dict[str, Any] = {
            "name": name,
            "response": {"content": "Could not read the port (an internal error occurred)."},
        }
        if fc_id:
            entry["id"] = fc_id
        return entry

    # ── Voice relay ───────────────────────────────────────────────────────────

    async def _start_voice(self) -> None:
        from app.config import get_settings
        settings = get_settings()

        if not settings.gemini_api_key:
            await _send(self._ws, {"type": "error", "message": "Voice not configured — set GEMINI_API_KEY"})
            return

        if self._voice_active:
            return

        self._voice_active = True
        self._canvas_dirty.clear()
        await _send(self._ws, {"type": "voice_started"})
        self._voice_task = asyncio.create_task(self._gemini_voice_relay())

    async def _stop_voice(self) -> None:
        if not self._voice_active:
            return
        self._voice_active = False
        self._canvas_dirty.set()  # wake the refresh loop so it exits
        await self._audio_queue.put(None)  # signal relay to exit
        if self._voice_task and not self._voice_task.done():
            try:
                await asyncio.wait_for(self._voice_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                self._voice_task.cancel()
        self._voice_task = None
        await _send(self._ws, {"type": "voice_stopped"})

    async def _gemini_voice_relay(self) -> None:
        """Relay PCM audio between the client and Gemini Live API."""
        from app.config import get_settings
        settings = get_settings()

        try:
            from google.genai import types as genai_types

            from app.core.llm import get_gemini_client

            client = get_gemini_client(settings.gemini_api_key)

            # Build Grafux context message for context seeding. Port-file
            # contents are rendered readably (active blocks first) so the voice
            # assistant can describe what each block/port contains.
            diagram_context = self._build_diagram_context(budget=9000, lib_limit=30)

            from app.modules.session.canvas_tools import get_live_tools_config

            grafux_instructions = (
                "You are a voice assistant embedded in Grafux, a visual block-diagram pipeline tool. "
                "Respond conversationally and concisely — you are speaking, not writing.\n\n"
                "You have FULL CONTROL over the canvas via the provided tools. "
                "When the user asks you to change the diagram (set a port value, run a block, connect ports, etc.), "
                "call the matching tool function. Speak a brief confirmation; do NOT read JSON aloud.\n\n"
                "Use exact block and port names from the canvas/active-block context. "
                "For category-based saved blocks, block_name is the leaf block (e.g. add1), "
                "not the category folder (e.g. general).\n"
                "The context below shows each block's input/output port-file contents. When the "
                "user asks what a port or block contains, or to summarize/explain port data, answer "
                "from that context. If a value is marked truncated or you need the full file, call "
                "read_port_value — it returns the content so you can describe it.\n"
                "Ask for confirmation before load_block, create_block, or delete_block.\n"
                "Execute set_port_value, run_block, add_port, connect_ports immediately when asked.\n\n"
                f"{diagram_context}"
            )

            live_config: dict = {
                "response_modalities": ["AUDIO"],
                "output_audio_transcription": {},
                "history_config": {"initial_history_in_client_content": True},
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": "Aoede"}
                    }
                },
                "tools": get_live_tools_config(),
            }

            live_model = settings.gemini_live_model
            async with client.aio.live.connect(
                model=live_model,
                config=live_config,
            ) as gemini:

                # Seed Grafux context as initial history before mic audio arrives.
                # turn_complete=False so Gemini does not speak a response to this.
                await gemini.send_client_content(
                    turns=[
                        {"role": "user", "parts": [{"text": grafux_instructions}]},
                        {"role": "model", "parts": [{"text": "Understood. I have the Grafux diagram context and I'm ready to assist via voice."}]},
                    ],
                    turn_complete=False,
                )
                self._last_voice_context = diagram_context

                async def _forward_to_gemini() -> None:
                    while True:
                        chunk = await self._audio_queue.get()
                        if chunk is None:
                            break
                        await gemini.send_realtime_input(
                            audio=genai_types.Blob(
                                data=chunk,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )

                async def _stream_from_gemini() -> None:
                    import base64

                    from app.modules.session.canvas_tools import (
                        build_tool_function_responses,
                        tool_calls_to_actions,
                    )

                    # NOTE: `turn_tool_actions` is declared OUTSIDE the while loop on
                    # purpose. `gemini.receive()` ends at the end of each turn (hence the
                    # outer loop), and a Gemini function call completes its OWN turn — the
                    # spoken confirmation arrives in the NEXT receive() turn. Resetting this
                    # per-iteration would drop the queued canvas action before the speech
                    # turn's turn_complete flushes it (the bug where voice said "done" but
                    # the port never changed). It is reset only after a successful flush.
                    turn_tool_actions: list[Any] = []
                    while self._voice_active:
                        turn_transcript = ""
                        async for response in gemini.receive():
                            if not self._voice_active:
                                return

                            # Tool calls — structured canvas actions (primary voice path)
                            tool_call = getattr(response, "tool_call", None)
                            if tool_call is None and response.server_content:
                                tool_call = getattr(response.server_content, "tool_call", None)
                            if tool_call:
                                function_calls = getattr(tool_call, "function_calls", None) or []
                                if function_calls:
                                    # read_port_value rounds-trips to the client and
                                    # returns real content; all other calls map to
                                    # queued canvas actions applied on the client.
                                    read_calls = [
                                        fc for fc in function_calls
                                        if (getattr(fc, "name", None)
                                            or (fc.get("name") if isinstance(fc, dict) else None))
                                        == "read_port_value"
                                    ]
                                    other_calls = [
                                        fc for fc in function_calls if fc not in read_calls
                                    ]

                                    turn_tool_actions.extend(
                                        tool_calls_to_actions(other_calls, self._active_blocks)
                                    )

                                    responses = build_tool_function_responses(other_calls)
                                    if read_calls:
                                        # Resolve concurrent reads in parallel, not serially.
                                        # return_exceptions so one failed read can't cancel its
                                        # siblings (and kill the relay) — failures degrade to a
                                        # graceful "could not read" response instead.
                                        read_results = await asyncio.gather(
                                            *(self._build_read_response(fc) for fc in read_calls),
                                            return_exceptions=True,
                                        )
                                        for fc, res in zip(read_calls, read_results, strict=False):
                                            if isinstance(res, BaseException):
                                                log.warning(
                                                    "voice_read_response_failed",
                                                    session_id=self._session_id,
                                                    error=str(res),
                                                )
                                                responses.append(
                                                    self._failed_read_response(fc)
                                                )
                                            else:
                                                responses.append(res)

                                    try:
                                        await gemini.send_tool_response(
                                            function_responses=responses
                                        )
                                    except Exception as exc:
                                        log.warning(
                                            "voice_tool_response_failed",
                                            session_id=self._session_id,
                                            error=str(exc),
                                        )

                            # Audio bytes — forward directly for playback
                            audio_bytes: bytes | None = None
                            if hasattr(response, "data") and isinstance(response.data, bytes):
                                audio_bytes = response.data
                            elif response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if hasattr(part, "inline_data") and part.inline_data:
                                        raw = part.inline_data.data
                                        audio_bytes = base64.b64decode(raw) if isinstance(raw, str) else raw
                                        break

                            if audio_bytes:
                                try:
                                    await self._ws.send_bytes(audio_bytes)
                                except Exception:
                                    return

                            # Accumulate transcript of audio output
                            if (
                                response.server_content
                                and hasattr(response.server_content, "output_transcription")
                                and response.server_content.output_transcription
                                and response.server_content.output_transcription.text
                            ):
                                frag = response.server_content.output_transcription.text
                                turn_transcript += frag
                                await _send(self._ws, {"type": "text_chunk", "text": frag})

                            if (
                                response.server_content
                                and response.server_content.turn_complete
                            ):
                                actions: list[Any] = list(turn_tool_actions)
                                display_text = turn_transcript
                                if not actions and turn_transcript:
                                    display_text, actions = _parse_actions_tag(
                                        turn_transcript,
                                        self._session_id,
                                    )
                                # Send the turn immediately and enrich in the background so
                                # gemini.receive() keeps flowing audio — blocking here on web
                                # grounding (5-40s) used to stall the whole voice loop.
                                self._stamp_actions(actions)
                                if display_text or actions:
                                    await _send(self._ws, {
                                        "type": "turn_complete",
                                        "full_text": display_text,
                                        "actions": actions,
                                    })
                                self._spawn_enrichment(actions)
                                turn_transcript = ""
                                turn_tool_actions = []

                async def _refresh_to_gemini() -> None:
                    """Re-seed Gemini with fresh diagram context when the canvas changes.

                    The client keeps pushing `canvas_update` (with updated port-file
                    contents) during a live session — e.g. after running a block or
                    regenerating a tool. Without this, the voice assistant keeps
                    answering from the snapshot taken at session start (the bug where
                    the user had to Stop/Live to refresh). Debounced and change-gated
                    so rapid edits don't flood the live session.
                    """
                    while self._voice_active:
                        await self._canvas_dirty.wait()
                        # Debounce: keep resetting the timer while edits keep arriving, so a
                        # burst re-seeds Gemini once it settles — not once per edit. 1.0s keeps
                        # the assistant's view fresh without flooding the live session.
                        while self._voice_active:
                            self._canvas_dirty.clear()
                            try:
                                await asyncio.wait_for(self._canvas_dirty.wait(), timeout=1.0)
                            except TimeoutError:
                                break  # quiet period elapsed → process the latest canvas state
                        if not self._voice_active:
                            return
                        new_ctx = self._build_diagram_context(budget=9000, lib_limit=30)
                        if new_ctx == self._last_voice_context:
                            continue  # nothing port-relevant actually changed
                        self._last_voice_context = new_ctx
                        refresh_msg = (
                            "The canvas has changed. Here is the UPDATED diagram context — "
                            "use these current port values from now on:\n\n" + new_ctx
                        )
                        try:
                            await gemini.send_client_content(
                                turns=[
                                    {"role": "user", "parts": [{"text": refresh_msg}]},
                                    {"role": "model", "parts": [{"text": "Got it — I've refreshed my view of the canvas."}]},
                                ],
                                turn_complete=False,  # update history without speaking
                            )
                        except Exception as exc:
                            log.warning(
                                "voice_context_refresh_failed",
                                session_id=self._session_id,
                                error=str(exc),
                            )

                await asyncio.gather(
                    _forward_to_gemini(),
                    _stream_from_gemini(),
                    _refresh_to_gemini(),
                )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("voice_relay_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": f"Voice error: {exc}"})
        finally:
            self._voice_active = False
            await _send(self._ws, {"type": "voice_stopped"})
