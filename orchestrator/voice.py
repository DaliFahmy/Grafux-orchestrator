from __future__ import annotations

import asyncio
import base64
import json
import logging

import websockets

from .config import Config
from .prompt import PromptBuilder
from .reasoning import Reasoning
from .session import Session

log = logging.getLogger("orchestrator.voice")

_GEMINI_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
_GEMINI_VOICE = "Aoede"
_CONTEXT_SENTINEL = "__context__"


class VoiceSession:
    """Manages a bidirectional Gemini Live audio session for one client.

    Audio flow:
      C++ mic PCM  →  session.audio_to_gemini queue  →  Gemini
      Gemini audio →  C++ client (sent as binary WebSocket frames)
    """

    def __init__(
        self,
        session: Session,
        reasoning: Reasoning,
        config: Config,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._session = session
        self._reasoning = reasoning
        self._config = config
        self._prompt_builder = prompt_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Open the Gemini Live WebSocket and relay audio until the session ends."""
        session = self._session
        ws = session.ws

        if not self._config.gemini_api_key:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "GEMINI_API_KEY not configured on server.",
            }))
            return

        url = f"{self._config.gemini_live_url}?key={self._config.gemini_api_key}"
        setup_msg = self._build_setup_message()

        try:
            async with websockets.connect(url, max_size=20 * 1024 * 1024) as gemini_ws:
                log.info("Gemini Live connected for user=%s", session.user_id)
                await gemini_ws.send(json.dumps(setup_msg))
                await ws.send_text(json.dumps({"type": "voice_started"}))

                send_task = asyncio.create_task(self._send_audio_loop(gemini_ws))
                recv_task = asyncio.create_task(self._receive_gemini_loop(gemini_ws))

                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()

        except websockets.exceptions.WebSocketException as exc:
            log.error("Gemini Live WS error: %s", exc)
            try:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"Gemini Live: {exc}",
                }))
            except Exception:
                pass
        except Exception as exc:
            log.error("Gemini Live unexpected error: %s", exc)
        finally:
            self._session.voice_active = False
            try:
                await ws.send_text(json.dumps({"type": "voice_stopped"}))
            except Exception:
                pass
            log.info("Gemini Live session ended for user=%s", session.user_id)

    async def send_canvas_context_update(self) -> None:
        """Push the current canvas state into the active Gemini voice session.

        Encodes the canvas as a clientContent turn so Gemini has up-to-date
        block/port values without interrupting the audio stream.
        """
        session = self._session
        if not session.voice_active:
            return

        blocks = session.canvas_state.get("blocks", [])
        if not blocks:
            return

        lines = ["[Canvas state update] Current state of all canvas blocks:"]
        for blk in blocks:
            name = blk.get("name", "?")
            lines.append(f'\nBlock "{name}" (type: {blk.get("type", "?")}):')
            ports = blk.get("ports", [])
            if not ports:
                lines.append("  (no ports)")
            else:
                for p in ports:
                    direction = "output" if p.get("is_output") else "input"
                    value = p.get("value") or "(empty)"
                    lines.append(f'  [{direction}] {p.get("name", "?")}: {value}')
        lines.append("\nPlease update your understanding of all canvas blocks accordingly.")

        client_content_msg = json.dumps({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "\n".join(lines)}]}],
                "turnComplete": False,
            }
        })
        await session.audio_to_gemini.put((_CONTEXT_SENTINEL, client_content_msg))

    # ------------------------------------------------------------------
    # Private: audio send loop
    # ------------------------------------------------------------------

    async def _send_audio_loop(self, gemini_ws) -> None:
        """Drain session.audio_to_gemini and forward frames to Gemini."""
        session = self._session
        while session.voice_active:
            try:
                pcm = await asyncio.wait_for(session.audio_to_gemini.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if pcm is None:
                break

            if isinstance(pcm, tuple) and pcm[0] == _CONTEXT_SENTINEL:
                try:
                    await gemini_ws.send(pcm[1])
                except Exception:
                    break
                continue

            b64 = base64.b64encode(pcm).decode()
            msg = {
                "realtimeInput": {
                    "mediaChunks": [{"mimeType": "audio/pcm;rate=16000", "data": b64}]
                }
            }
            try:
                await gemini_ws.send(json.dumps(msg))
            except Exception:
                break

    # ------------------------------------------------------------------
    # Private: Gemini receive loop
    # ------------------------------------------------------------------

    async def _receive_gemini_loop(self, gemini_ws) -> None:
        """Receive frames from Gemini and forward audio/text to the C++ client."""
        session = self._session
        ws = session.ws
        transcript_buffer = ""

        async for raw in gemini_ws:
            if not session.voice_active:
                break

            data = self._parse_gemini_frame(raw)
            if data is None:
                if isinstance(raw, (bytes, bytearray)):
                    await ws.send_bytes(bytes(raw))
                continue

            sc = data.get("serverContent", {})
            transcript_buffer = await self._handle_server_content(
                sc, ws, transcript_buffer
            )

    async def _handle_server_content(
        self, sc: dict, ws, transcript_buffer: str
    ) -> str:
        """Process a serverContent payload; return the updated transcript buffer."""
        session = self._session

        # Output transcription from native-audio preview models
        out_tr_text = sc.get("outputTranscription", {}).get("text", "")
        if out_tr_text:
            transcript_buffer += out_tr_text
            await ws.send_text(json.dumps({"type": "text_chunk", "text": out_tr_text}))

        # Inline audio and text parts from modelTurn
        for part in sc.get("modelTurn", {}).get("parts", []):
            transcript_buffer = await self._handle_part(part, ws, transcript_buffer)

        if sc.get("turnComplete"):
            transcript_buffer = await self._finalize_turn(ws, transcript_buffer)

        if sc.get("interrupted"):
            log.debug("Gemini interrupted current turn")

        return transcript_buffer

    async def _handle_part(self, part: dict, ws, transcript_buffer: str) -> str:
        """Forward one modelTurn part (audio or text) to the client."""
        inline = part.get("inlineData", {})
        if inline.get("mimeType", "").startswith("audio/"):
            audio_b64 = inline.get("data", "")
            if audio_b64:
                await ws.send_bytes(base64.b64decode(audio_b64))

        text_part = part.get("text", "")
        if text_part:
            transcript_buffer += text_part
            await ws.send_text(json.dumps({"type": "text_chunk", "text": text_part}))

        return transcript_buffer

    async def _finalize_turn(self, ws, transcript_buffer: str) -> str:
        """Parse actions from the completed transcript and send turn_complete."""
        session = self._session
        clean_text, actions = self._reasoning.parse_actions(transcript_buffer)

        if not actions and transcript_buffer.strip():
            actions = await self._reasoning.extract_actions_from_voice(
                session, transcript_buffer
            )

        if clean_text:
            session.history.append({"role": "assistant", "content": clean_text})
            max_entries = self._config.history_max_turns * 2
            if len(session.history) > max_entries:
                session.history = session.history[-max_entries:]

        await ws.send_text(json.dumps({
            "type": "turn_complete",
            "full_text": clean_text,
            "actions": actions,
        }))

        return ""  # reset buffer

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _build_setup_message(self) -> dict:
        system_prompt = self._prompt_builder.build(self._session)
        return {
            "setup": {
                # Must match the model used by the C++ GeminiLiveClient.
                # gemini-2.0-flash-live-001 is rejected with 1008 on v1beta.
                "model": _GEMINI_MODEL,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": _GEMINI_VOICE}
                        },
                    },
                },
                "systemInstruction": {
                    "parts": [{"text": system_prompt}],
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }
        }

    @staticmethod
    def _parse_gemini_frame(raw) -> dict | None:
        """Decode a raw WebSocket frame from Gemini into a dict, or None on failure."""
        try:
            text = raw if isinstance(raw, str) else raw.decode()
            return json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
