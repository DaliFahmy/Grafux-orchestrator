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
            text = data.get("text", "").strip()
            if text:
                await self._run_text_turn(text)

        elif msg_type == "canvas_update":
            self._canvas_state = data.get("canvas_state", self._canvas_state)

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

        system_parts = [
            "You are Grafux, an AI assistant for a visual programming canvas. "
            "Help users create, modify, and understand their canvas blocks and workflows."
        ]
        if self._canvas_state:
            system_parts.append(f"Current canvas: {json.dumps(self._canvas_state)[:2000]}")
        if self._active_blocks:
            system_parts.append(f"Active blocks: {json.dumps(self._active_blocks)[:500]}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": " ".join(system_parts)},
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

            self._history.append({"role": "assistant", "content": full_text})
            await _send(self._ws, {"type": "turn_complete", "full_text": full_text, "actions": []})

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
            live_config = genai_types.LiveConnectConfig(
                response_modalities=["AUDIO"],
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
            async with client.aio.live.connect(
                model=live_model,
                config=live_config,
            ) as gemini:

                async def _forward_to_gemini() -> None:
                    while True:
                        chunk = await self._audio_queue.get()
                        if chunk is None:
                            break
                        await gemini.send(
                            input={
                                "realtime_input": {
                                    "media_chunks": [
                                        {
                                            "data": chunk,
                                            "mime_type": "audio/pcm;rate=16000",
                                        }
                                    ]
                                }
                            }
                        )

                async def _stream_from_gemini() -> None:
                    async for response in gemini.receive():
                        if not self._voice_active:
                            break
                        # Audio data comes as bytes on the response
                        audio_bytes: bytes | None = None
                        if hasattr(response, "data"):
                            audio_bytes = response.data
                        elif hasattr(response, "server_content"):
                            sc = response.server_content
                            if hasattr(sc, "model_turn") and sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if hasattr(part, "inline_data") and part.inline_data:
                                        audio_bytes = part.inline_data.data
                                        break
                        if audio_bytes:
                            try:
                                await self._ws.send_bytes(audio_bytes)
                            except Exception:
                                break

                await asyncio.gather(_forward_to_gemini(), _stream_from_gemini())

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("voice_relay_error", session_id=self._session_id, error=str(exc))
            await _send(self._ws, {"type": "error", "message": f"Voice error: {exc}"})
        finally:
            self._voice_active = False
            await _send(self._ws, {"type": "voice_stopped"})
