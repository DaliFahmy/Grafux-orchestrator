from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

# region agent log
import time as _time, pathlib as _pathlib
_DEBUG_LOG = _pathlib.Path("debug-859e2c.log")
def _dlog(msg: str, data: dict, hyp: str) -> None:
    entry = json.dumps({"sessionId":"859e2c","timestamp":int(_time.time()*1000),"location":"session/router.py","message":msg,"data":data,"hypothesisId":hyp})
    try:
        with _DEBUG_LOG.open("a", encoding="utf-8") as _f:
            _f.write(entry + "\n")
    except Exception:
        pass
# endregion

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.core.security import verify_jwt

log = get_logger("session.router")

router = APIRouter()


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
            lib_lines = [
                f'  block_type="{e.get("block_type","")}" block_name="{e.get("block_name","")}"'
                for e in self._saved_library
            ]
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

        # region agent log
        _dlog("system_prompt_built", {
            "has_canvas": bool(self._canvas_state),
            "has_saved_library": bool(self._saved_library),
            "has_active_blocks": bool(self._active_blocks),
            "diagram_context_placeholder_present": "{DIAGRAM_CONTEXT}" in raw_prompt,
            "diagram_context_snippet": diagram_context[:200],
        }, "H-B")
        # endregion

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

            # region agent log
            _dlog("full_text_before_parse", {
                "has_actions_tag": "##ACTIONS##" in full_text,
                "tail_200": full_text[-200:] if len(full_text) > 200 else full_text,
                "length": len(full_text),
            }, "H-A")
            # endregion

            # Parse ##ACTIONS## tag from the end of the model response
            display_text = full_text
            actions: list[Any] = []
            marker = "##ACTIONS##"
            idx = full_text.rfind(marker)
            if idx != -1:
                json_str = full_text[idx + len(marker):].strip()
                display_text = full_text[:idx].rstrip()
                try:
                    parsed = json.loads(json_str)
                    actions = parsed.get("actions", [])
                except Exception:
                    pass

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

        # region agent log
        _dlog("voice_start", {
            "has_canvas": bool(self._canvas_state),
            "has_active_blocks": bool(self._active_blocks),
            "response_modalities": ["AUDIO"],
            "system_prompt_sent": True,
        }, "H-C")
        # endregion

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

            # Build voice system prompt — concise Grafux context for Gemini Live
            from app.prompts import get_system_prompt
            voice_context_parts: list[str] = []
            if self._canvas_state:
                voice_context_parts.append(
                    f"Current canvas blocks:\n{json.dumps(self._canvas_state)[:2000]}"
                )
            if self._active_blocks:
                voice_context_parts.append(
                    f"Active blocks:\n{json.dumps(self._active_blocks)[:400]}"
                )
            if self._saved_library:
                lib_lines = [
                    f'  block_type="{e.get("block_type","")}" block_name="{e.get("block_name","")}"'
                    for e in self._saved_library
                ]
                voice_context_parts.append("Saved block library:\n" + "\n".join(lib_lines[:20]))
            voice_context = (
                "\n\n".join(voice_context_parts) if voice_context_parts else "No diagram loaded."
            )
            voice_system = (
                "You are a voice assistant embedded in Grafux, a visual block-diagram pipeline tool. "
                "Answer questions about the diagram conversationally. "
                "Be concise — you are speaking, not writing.\n\n"
                + voice_context
            )

            # region agent log
            _dlog("voice_relay_start", {
                "has_canvas": bool(self._canvas_state),
                "has_active_blocks": bool(self._active_blocks),
                "response_modalities": ["AUDIO"],
                "system_prompt_sent": True,
                "system_snippet": voice_system[:150],
            }, "H-C")
            # endregion

            live_config = genai_types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=genai_types.Content(
                    parts=[genai_types.Part(text=voice_system)],
                ),
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name="Aoede"
                        )
                    )
                ),
            )

            # Model name is configurable; default is the current GA Live model.
            live_model = settings.gemini_live_model

            # region agent log
            _dlog("gemini_connect_attempt", {
                "model": live_model,
                "has_system_instruction": True,
                "system_len": len(voice_system),
            }, "H-C")
            # endregion

            async with client.aio.live.connect(
                model=live_model,
                config=live_config,
            ) as gemini:

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
                    # receive() yields until turn_complete; loop for continuous streaming
                    while self._voice_active:
                        async for response in gemini.receive():
                            if not self._voice_active:
                                return
                            audio_bytes: bytes | None = None

                            # Direct .data attribute
                            if hasattr(response, "data") and isinstance(response.data, bytes):
                                audio_bytes = response.data
                            # Parts inside server_content.model_turn
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

                await asyncio.gather(_forward_to_gemini(), _stream_from_gemini())

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("voice_relay_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": f"Voice error: {exc}"})
        finally:
            self._voice_active = False
            await _send(self._ws, {"type": "voice_stopped"})
