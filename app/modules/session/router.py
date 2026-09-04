from __future__ import annotations

import asyncio
import json
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
from app.modules.session.enrichment import (
    enrich_actions_streaming,
    is_enrichable,
)

log = get_logger("session.router")

router = APIRouter()


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

    await _send(websocket, {"type": "session_ready", "session_id": session_id})

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
        self._history: list[dict[str, Any]] = []
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
        # Pending on-demand port reads, keyed by request_id → future resolved
        # when the client replies with a "port_content" message.
        self._pending_reads: dict[str, asyncio.Future[dict[str, Any]]] = {}
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
                await self._run_text_turn(text)

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
            req_id = data.get("request_id", "")
            fut = self._pending_reads.get(req_id)
            if fut is not None and not fut.done():
                fut.set_result({
                    "found": bool(data.get("found")),
                    "content": data.get("content", ""),
                })

        elif msg_type == "start_voice":
            await self._start_voice()

        elif msg_type == "stop_voice":
            await self._stop_voice()

    # ── On-demand port read (round-trip to the client) ───────────────────────

    async def _request_port_content(
        self,
        target_block: str,
        direction: str,
        port_name: str,
        timeout: float = 12.0,
    ) -> dict[str, Any]:
        """Ask the client to read a port file and return {found, content}.

        Port files live on the client's filesystem, so the server cannot read
        them directly. Older clients that don't understand "read_port" simply
        never reply, so we time out gracefully and report the file as missing.
        """
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_reads[request_id] = fut
        await _send(self._ws, {
            "type": "read_port",
            "request_id": request_id,
            "target_block": target_block,
            "direction": direction,
            "port_name": port_name,
        })
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            log.warning(
                "port_read_timeout",
                session_id=self._session_id,
                target_block=target_block,
                port_name=port_name,
            )
            return {"found": False, "content": ""}
        finally:
            self._pending_reads.pop(request_id, None)

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

    async def _run_text_turn(self, text: str) -> None:
        from app.config import get_settings
        settings = get_settings()

        if not settings.openai_api_key:
            await _send(self._ws, {"type": "error", "message": "AI not configured — set OPENAI_API_KEY"})
            return

        self._history.append({"role": "user", "content": text})

        from app.prompts import get_system_prompt
        raw_prompt = get_system_prompt("chat_assistant")

        # Build DIAGRAM_CONTEXT from whatever the client has sent.
        diagram_context = self._build_diagram_context(empty="No diagram loaded yet.")
        system_prompt = raw_prompt.replace("{DIAGRAM_CONTEXT}", diagram_context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *self._history,
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
            self._history.append({"role": "assistant", "content": display_text})
            await _send(self._ws, {"type": "turn_complete", "full_text": display_text, "actions": actions})
            self._spawn_enrichment(actions)

        except Exception as exc:
            log.error("ai_turn_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": str(exc)})

    async def _resolve_read_tool(self, args: dict[str, Any]) -> str:
        """Resolve a read_port_value tool call into a text result for the model."""
        target_block = str(args.get("target_block", "")).strip()
        direction = str(args.get("direction", "")).strip() or "output"
        port_name = str(args.get("port_name", "")).strip()
        if not target_block or not port_name:
            return "Error: target_block and port_name are required."
        result = await self._request_port_content(target_block, direction, port_name)
        if not result.get("found"):
            return (
                f'Could not read {direction} port "{port_name}" on block '
                f'"{target_block}" (file empty, missing, or client unavailable).'
            )
        return result.get("content", "") or "(the port file is empty)"

    async def _build_read_response(self, fc: Any) -> dict[str, Any]:
        """Build a Gemini function_response for a read_port_value call."""
        from app.modules.session.canvas_tools import _default_target_block

        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "")
        fc_id = getattr(fc, "id", None) or (fc.get("id") if isinstance(fc, dict) else None)
        raw_args = getattr(fc, "args", None)
        if raw_args is None and isinstance(fc, dict):
            raw_args = fc.get("args", {})
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        if not str(args.get("target_block", "")).strip():
            args["target_block"] = _default_target_block(self._active_blocks)
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
