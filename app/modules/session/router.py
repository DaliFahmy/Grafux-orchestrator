from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.core.security import verify_jwt

log = get_logger("session.router")

router = APIRouter()


def _format_library_entry(entry: dict[str, Any]) -> str:
    """Format one saved-library entry for DIAGRAM_CONTEXT."""
    block_type = entry.get("block_type", "")
    block_name = entry.get("block_name", "")
    category = entry.get("category", "")
    if category:
        return (
            f'  block_type="{block_type}" category="{category}" block_name="{block_name}"'
        )
    return f'  block_type="{block_type}" block_name="{block_name}"'


# OpenAI tool schema for the on-demand port read (text path). Mutations stay on
# the ##ACTIONS## tag; only reads are a real round-trip tool.
READ_PORT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_port_value",
        "description": (
            "Read the FULL current content of a block's input or output port file. "
            "Use this when the user asks what a port or block contains, to summarize or "
            "explain port data, or when the canvas context marks a value as truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_block": {"type": "string", "description": "Block name on the canvas."},
                "direction": {"type": "string", "description": "Must be 'input' or 'output'."},
                "port_name": {"type": "string", "description": "Port name (snake_case)."},
            },
            "required": ["target_block", "direction", "port_name"],
        },
    },
}


def _render_port(port: dict[str, Any]) -> str:
    """Render one port object (from the client) as a readable line."""
    name = port.get("name", "")
    direction = "out" if port.get("is_output") else "in"
    value = port.get("value", "")
    if value == "" or value is None:
        value = "(empty)"
    line = f'    [{direction}] {name}: {value}'
    if port.get("truncated"):
        full_len = port.get("full_length")
        shown = len(str(port.get("value", "")))
        suffix = (
            f" …(truncated: showing {shown} of {full_len} chars — "
            "call read_port_value to read the full file)"
        )
        line += suffix
    return line


def _render_block(block: dict[str, Any], *, active: bool) -> str:
    """Render one block (name, type, status, description, ports, connections)."""
    name = block.get("name", "")
    btype = block.get("type", "")
    status = block.get("status", "")
    header = f'Block "{name}" ({btype}, {status})'
    if active:
        header += "  [ACTIVE]"
    lines = [header]
    desc = (block.get("description") or "").strip()
    if desc:
        lines.append(f"  description: {desc}")
    ports = block.get("ports") or []
    if ports:
        lines.append("  ports:")
        lines.extend(_render_port(p) for p in ports if isinstance(p, dict))
    conns = block.get("connections") or []
    if conns:
        for c in conns:
            if isinstance(c, dict):
                lines.append(
                    f'  connection: {c.get("from_port","")} -> '
                    f'{c.get("to_block","")}.{c.get("to_port","")}'
                )
    return "\n".join(lines)


def _render_canvas_context(
    canvas_state: dict[str, Any],
    active_blocks: list[Any],
    *,
    budget: int = 12000,
) -> str:
    """Render the canvas as readable text for the model, active blocks first.

    Active blocks are emitted before the rest (and never dropped) so the model
    always sees full context for what the user is focused on. Remaining blocks
    fill the character budget; any overflow is noted, not silently dropped.
    """
    blocks = (canvas_state or {}).get("blocks") or []
    active_names = {
        b.get("name") for b in (active_blocks or []) if isinstance(b, dict) and b.get("name")
    }

    # Active blocks: prefer the richer active_blocks payload, fall back to canvas.
    active_rendered: list[str] = []
    seen: set[str] = set()
    for b in (active_blocks or []):
        if isinstance(b, dict) and b.get("name"):
            active_rendered.append(_render_block(b, active=True))
            seen.add(b.get("name"))
    for b in blocks:
        if isinstance(b, dict) and b.get("name") in active_names and b.get("name") not in seen:
            active_rendered.append(_render_block(b, active=True))
            seen.add(b.get("name"))

    other_rendered: list[str] = [
        _render_block(b, active=False)
        for b in blocks
        if isinstance(b, dict) and b.get("name") not in active_names
    ]

    parts: list[str] = []
    used = 0
    # Active blocks are always included.
    for chunk in active_rendered:
        parts.append(chunk)
        used += len(chunk) + 2
    omitted = 0
    for chunk in other_rendered:
        if used + len(chunk) > budget:
            omitted += 1
            continue
        parts.append(chunk)
        used += len(chunk) + 2
    if omitted:
        parts.append(
            f"…({omitted} more block(s) not shown to save space; ask about them "
            "by name and call read_port_value for their port contents.)"
        )
    return "\n\n".join(parts) if parts else ""


def _parse_actions_tag(full_text: str, session_id: str) -> tuple[str, list[Any]]:
    """Extract ##ACTIONS## JSON from model text; log parse failures."""
    marker = "##ACTIONS##"
    idx = full_text.rfind(marker)
    if idx == -1:
        return full_text, []
    json_str = full_text[idx + len(marker):].strip()
    display_text = full_text[:idx].rstrip()
    try:
        parsed = json.loads(json_str)
        return display_text, parsed.get("actions", [])
    except Exception as exc:
        log.warning(
            "actions_parse_failed",
            session_id=session_id,
            error=str(exc),
            snippet=json_str[:200],
        )
        return display_text, []


async def _send(websocket: WebSocket, data: dict[str, Any]) -> None:
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass


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
        # Pending on-demand port reads, keyed by request_id → future resolved
        # when the client replies with a "port_content" message.
        self._pending_reads: dict[str, asyncio.Future[dict[str, Any]]] = {}

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
        if self._voice_task and not self._voice_task.done():
            self._voice_task.cancel()
            await self._audio_queue.put(None)
            try:
                await asyncio.wait_for(self._voice_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
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

        elif msg_type == "set_active_blocks":
            self._active_blocks = data.get("blocks", self._active_blocks)

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
        except asyncio.TimeoutError:
            log.warning(
                "port_read_timeout",
                session_id=self._session_id,
                target_block=target_block,
                port_name=port_name,
            )
            return {"found": False, "content": ""}
        finally:
            self._pending_reads.pop(request_id, None)

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

        # Build DIAGRAM_CONTEXT from whatever the client has sent. Blocks and
        # their port-file contents are rendered as readable text (active blocks
        # first), so the model can read and talk about port values directly.
        diagram_ctx_parts: list[str] = []
        canvas_text = _render_canvas_context(self._canvas_state, self._active_blocks)
        if canvas_text:
            diagram_ctx_parts.append("Current canvas blocks:\n" + canvas_text)
        if self._saved_library:
            lib_lines = [_format_library_entry(e) for e in self._saved_library]
            diagram_ctx_parts.append("Saved block library:\n" + "\n".join(lib_lines))
        diagram_context = (
            "\n\n".join(diagram_ctx_parts)
            if diagram_ctx_parts
            else "No diagram loaded yet."
        )
        system_prompt = raw_prompt.replace("{DIAGRAM_CONTEXT}", diagram_context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *self._history,
        ]

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)

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

            self._history.append({"role": "assistant", "content": display_text})
            await _send(self._ws, {"type": "turn_complete", "full_text": display_text, "actions": actions})

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
        await _send(self._ws, {"type": "voice_started"})
        self._voice_task = asyncio.create_task(self._gemini_voice_relay())

    async def _stop_voice(self) -> None:
        if not self._voice_active:
            return
        self._voice_active = False
        await self._audio_queue.put(None)  # signal relay to exit
        if self._voice_task and not self._voice_task.done():
            try:
                await asyncio.wait_for(self._voice_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._voice_task.cancel()
        self._voice_task = None
        await _send(self._ws, {"type": "voice_stopped"})

    async def _gemini_voice_relay(self) -> None:
        """Relay PCM audio between the client and Gemini Live API."""
        from app.config import get_settings
        settings = get_settings()

        try:
            from google import genai as google_genai
            from google.genai import types as genai_types

            client = google_genai.Client(api_key=settings.gemini_api_key)

            # Build Grafux context message for context seeding. Port-file
            # contents are rendered readably (active blocks first) so the voice
            # assistant can describe what each block/port contains.
            diagram_ctx_parts: list[str] = []
            canvas_text = _render_canvas_context(
                self._canvas_state, self._active_blocks, budget=9000
            )
            if canvas_text:
                diagram_ctx_parts.append("Current canvas blocks:\n" + canvas_text)
            if self._saved_library:
                lib_lines = [_format_library_entry(e) for e in self._saved_library[:30]]
                diagram_ctx_parts.append("Saved block library:\n" + "\n".join(lib_lines))
            diagram_context = "\n\n".join(diagram_ctx_parts) if diagram_ctx_parts else "No diagram loaded."

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

                    while self._voice_active:
                        turn_transcript = ""
                        turn_tool_actions: list[Any] = []
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
                                    for fc in read_calls:
                                        responses.append(
                                            await self._build_read_response(fc)
                                        )

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
                                if display_text or actions:
                                    await _send(self._ws, {
                                        "type": "turn_complete",
                                        "full_text": display_text,
                                        "actions": actions,
                                    })
                                turn_transcript = ""
                                turn_tool_actions = []

                await asyncio.gather(_forward_to_gemini(), _stream_from_gemini())

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("voice_relay_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": f"Voice error: {exc}"})
        finally:
            self._voice_active = False
            await _send(self._ws, {"type": "voice_stopped"})
