from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from .catalogue import CatalogueService
from .config import Config
from .prompt import PromptBuilder
from .reasoning import Reasoning
from .session import Session
from .voice import VoiceSession

log = logging.getLogger("orchestrator.handler")

_CATALOGUE_WAIT_TIMEOUT = 5.0
_CANVAS_DEBOUNCE_DELAY = 0.4


class SessionHandler:
    """Owns the WebSocket message loop for one connected client.

    Instantiated once per connection.  Orchestrates Session, CatalogueService,
    Reasoning, and VoiceSession for the lifetime of that connection.
    """

    def __init__(
        self,
        websocket: WebSocket,
        user_id: str,
        username: str,
        project_id: str,
        catalogue_service: CatalogueService,
        reasoning: Reasoning,
        prompt_builder: PromptBuilder,
        config: Config,
    ) -> None:
        self._ws = websocket
        self._catalogue_service = catalogue_service
        self._reasoning = reasoning
        self._prompt_builder = prompt_builder
        self._config = config

        self._session = Session(
            ws=websocket,
            user_id=user_id,
            username=username,
            project_id=project_id,
        )

        self._catalogue_task: asyncio.Task | None = None
        self._canvas_update_pending = False
        self._canvas_update_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Accept the connection, pre-fetch the catalogue, and run the message loop."""
        session = self._session

        self._catalogue_task = asyncio.create_task(
            self._catalogue_service.load(
                session.user_id, session.project_id, session.username
            )
        )

        log.info("Session opened: user=%s project=%s", session.user_id, session.project_id)
        await self._ws.send_text(json.dumps({"type": "session_ready"}))

        try:
            await self._message_loop()
        except WebSocketDisconnect:
            log.info("Session disconnected: user=%s", session.user_id)
        except Exception as exc:
            log.exception("Session error for user=%s: %s", session.user_id, exc)
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Private: message loop
    # ------------------------------------------------------------------

    async def _message_loop(self) -> None:
        session = self._session
        ws = self._ws

        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                log.info(
                    "Client disconnected: user=%s code=%s",
                    session.user_id,
                    msg.get("code"),
                )
                break

            if msg["type"] == "websocket.receive" and msg.get("bytes") is not None:
                if session.voice_active:
                    await session.audio_to_gemini.put(msg["bytes"])
                continue

            raw_text = msg.get("text", "")
            if not raw_text:
                continue

            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                continue

            await self._dispatch(data)

    async def _dispatch(self, data: dict) -> None:
        """Route an incoming JSON message to the appropriate handler method."""
        msg_type = data.get("type", "")

        if msg_type == "ping":
            await self._handle_ping()
        elif msg_type == "canvas_update":
            await self._handle_canvas_update(data)
        elif msg_type == "set_active_blocks":
            self._handle_set_active_blocks(data)
        elif msg_type == "text_message":
            await self._handle_text_message(data)
        elif msg_type == "start_voice":
            await self._handle_start_voice()
        elif msg_type == "stop_voice":
            await self._handle_stop_voice()
        else:
            log.debug("Unknown message type: %s", msg_type)

    # ------------------------------------------------------------------
    # Private: individual message handlers
    # ------------------------------------------------------------------

    async def _handle_ping(self) -> None:
        await self._ws.send_text(json.dumps({"type": "pong"}))

    async def _handle_canvas_update(self, data: dict) -> None:
        self._session.canvas_state = data.get("canvas_state", {})
        if not self._canvas_update_pending:
            self._canvas_update_pending = True
            self._canvas_update_task = asyncio.create_task(
                self._debounced_canvas_update()
            )

    def _handle_set_active_blocks(self, data: dict) -> None:
        self._session.active_blocks = data.get("blocks", [])

    async def _handle_text_message(self, data: dict) -> None:
        session = self._session

        if "canvas_state" in data:
            session.canvas_state = data["canvas_state"]
        if "active_blocks" in data:
            session.active_blocks = data["active_blocks"]

        await self._ensure_catalogue_loaded()

        user_text = data.get("text", "").strip()
        if user_text:
            asyncio.create_task(self._reasoning.stream_text(session, user_text))

    async def _handle_start_voice(self) -> None:
        session = self._session
        if session.voice_active:
            return

        await self._ensure_catalogue_loaded()

        session.voice_active = True
        self._flush_audio_queues()

        voice = VoiceSession(
            session=session,
            reasoning=self._reasoning,
            config=self._config,
            prompt_builder=self._prompt_builder,
        )
        session.gemini_task = asyncio.create_task(voice.run())

    async def _handle_stop_voice(self) -> None:
        session = self._session
        session.voice_active = False
        await session.audio_to_gemini.put(None)
        if session.gemini_task:
            session.gemini_task.cancel()
            session.gemini_task = None
        await self._ws.send_text(json.dumps({"type": "voice_stopped"}))

    # ------------------------------------------------------------------
    # Private: canvas debounce
    # ------------------------------------------------------------------

    async def _debounced_canvas_update(self) -> None:
        """Wait briefly, then push the latest canvas state into the voice session."""
        await asyncio.sleep(_CANVAS_DEBOUNCE_DELAY)
        self._canvas_update_pending = False

        session = self._session
        if not session.voice_active:
            return

        voice = VoiceSession(
            session=session,
            reasoning=self._reasoning,
            config=self._config,
            prompt_builder=self._prompt_builder,
        )
        await voice.send_canvas_context_update()

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    async def _ensure_catalogue_loaded(self) -> None:
        """Await the background catalogue task if it hasn't finished yet."""
        session = self._session
        task = self._catalogue_task
        if task is None:
            return
        if not task.done():
            try:
                session.catalogue = await asyncio.wait_for(
                    asyncio.shield(task), timeout=_CATALOGUE_WAIT_TIMEOUT
                )
            except (asyncio.TimeoutError, Exception):
                session.catalogue = "(catalogue loading…)"
        else:
            session.catalogue = task.result()

    def _flush_audio_queues(self) -> None:
        session = self._session
        while not session.audio_to_gemini.empty():
            session.audio_to_gemini.get_nowait()
        while not session.audio_from_gemini.empty():
            session.audio_from_gemini.get_nowait()

    async def _cleanup(self) -> None:
        session = self._session
        session.voice_active = False
        await session.audio_to_gemini.put(None)
        if session.gemini_task:
            session.gemini_task.cancel()
        if self._canvas_update_task:
            self._canvas_update_task.cancel()
        if self._catalogue_task:
            self._catalogue_task.cancel()
        log.info("Session cleaned up: user=%s", session.user_id)
