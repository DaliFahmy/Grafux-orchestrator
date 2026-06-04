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

        elif msg_type == "start_voice":
            await self._start_voice()

        elif msg_type == "stop_voice":
            await self._stop_voice()

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

        # Build DIAGRAM_CONTEXT from whatever the client has sent
        diagram_ctx_parts: list[str] = []
        if self._canvas_state:
            diagram_ctx_parts.append(
                f"Current canvas blocks:\n{json.dumps(self._canvas_state)[:3000]}"
            )
        if self._saved_library:
            lib_lines = [_format_library_entry(e) for e in self._saved_library]
            diagram_ctx_parts.append("Saved block library:\n" + "\n".join(lib_lines))
        diagram_context = (
            "\n\n".join(diagram_ctx_parts)
            if diagram_ctx_parts
            else "No diagram loaded yet."
        )
        system_prompt = raw_prompt.replace("{DIAGRAM_CONTEXT}", diagram_context)

        system_parts = [system_prompt]
        if self._active_blocks:
            system_parts.append(f"Active blocks: {json.dumps(self._active_blocks)[:500]}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            *self._history,
        ]

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            stream = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                stream=True,
            )
            full_text = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_text += delta
                    await _send(self._ws, {"type": "text_chunk", "text": delta})

            display_text, actions = _parse_actions_tag(full_text, self._session_id)

            self._history.append({"role": "assistant", "content": display_text})
            await _send(self._ws, {"type": "turn_complete", "full_text": display_text, "actions": actions})

        except Exception as exc:
            log.error("ai_turn_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": str(exc)})

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

            # Build Grafux context message for context seeding
            diagram_ctx_parts: list[str] = []
            if self._canvas_state:
                diagram_ctx_parts.append(
                    f"Current canvas blocks:\n{json.dumps(self._canvas_state)[:2500]}"
                )
            if self._active_blocks:
                diagram_ctx_parts.append(
                    f"Active blocks:\n{json.dumps(self._active_blocks)[:400]}"
                )
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
                                    mapped = tool_calls_to_actions(
                                        function_calls,
                                        self._active_blocks,
                                    )
                                    turn_tool_actions.extend(mapped)
                                    try:
                                        await gemini.send_tool_response(
                                            function_responses=build_tool_function_responses(
                                                function_calls
                                            )
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
