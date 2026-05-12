from __future__ import annotations

import json
import logging

import httpx

from .config import Config
from .prompt import PromptBuilder
from .session import Session

log = logging.getLogger("orchestrator.reasoning")

_ACTIONS_MARKER = "##ACTIONS##"

_VOICE_ACTION_VOCABULARY = (
    "Action vocabulary (use snake_case for all names):\n"
    '  run_block:        {"type":"run_block","target_block":"Name"}\n'
    '  regenerate_block: {"type":"regenerate_block","target_block":"Name"}\n'
    '  add_port:         {"type":"add_port","target_block":"Name","direction":"input","port_name":"name"}\n'
    '  remove_port:      {"type":"remove_port","target_block":"Name","direction":"output","port_name":"name"}\n'
    '  rename_port:      {"type":"rename_port","target_block":"Name","direction":"input","old_port_name":"a","new_port_name":"b"}\n'
    '  set_port_value:   {"type":"set_port_value","target_block":"Name","direction":"input","port_name":"name","value":"val"}\n'
    '  connect_ports:    {"type":"connect_ports","from_block":"A","from_port":"out","to_block":"B","to_port":"in"}\n'
    '  disconnect_ports: {"type":"disconnect_ports","from_block":"A","from_port":"out","to_block":"B","to_port":"in"}\n'
    '  set_description:  {"type":"set_description","target_block":"Name","description":"text"}\n'
    '  rename_block:     {"type":"rename_block","target_block":"OldName","new_name":"NewName"}\n'
    '  delete_block:     {"type":"delete_block","target_block":"Name"}\n\n'
    'If the AI was only speaking (no canvas action performed), return: {"actions": []}'
)


class Reasoning:
    """AI reasoning layer: action parsing, GPT text streaming, and voice-action extraction."""

    def __init__(self, config: Config, prompt_builder: PromptBuilder) -> None:
        self._config = config
        self._prompt_builder = prompt_builder

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_actions(self, text: str) -> tuple[str, list]:
        """Extract the ##ACTIONS## block from an LLM response.

        Returns (clean_text, actions_list).  clean_text has the marker and
        everything after it removed.
        """
        idx = text.find(_ACTIONS_MARKER)
        if idx == -1:
            return text, []

        clean_text = text[:idx].strip()
        json_str = text[idx + len(_ACTIONS_MARKER):].strip()
        json_str = self._strip_wrapping(json_str)

        data = self._try_parse_json(json_str)
        if data is None and "'" in json_str:
            # Last-resort: swap single quotes to double quotes, but only when
            # standard parse failed (swapping blindly corrupts Python code fields).
            data = self._try_parse_json(json_str.replace("'", '"'))

        if data is None:
            log.warning("Could not parse ##ACTIONS## JSON: %s", json_str[:200])
            return clean_text, []

        actions = data.get("actions", []) if isinstance(data, dict) else []
        return clean_text, actions

    # ------------------------------------------------------------------
    # Voice-action fallback extraction
    # ------------------------------------------------------------------

    async def extract_actions_from_voice(
        self, session: Session, voice_transcript: str
    ) -> list:
        """Use GPT-4o-mini to extract structured actions from a voice transcript.

        Gemini Live produces natural-language audio output that never contains
        ##ACTIONS## JSON.  This method calls a fast GPT model with the transcript
        and a compact canvas context so it can recover the implied actions.
        """
        if not self._config.openai_api_key or not voice_transcript.strip():
            return []

        canvas_context = self._build_compact_canvas_context(session)
        prompt = (
            f"Canvas blocks:\n{canvas_context}\n\n"
            "The Grafux AI voice assistant just said:\n"
            f'"{voice_transcript.strip()}"\n\n'
            "Based on what the AI described doing, extract any canvas actions performed.\n"
            "Return ONLY a JSON object — no markdown, no explanation:\n"
            '{"actions": [...]}\n\n'
            + _VOICE_ACTION_VOCABULARY
        )

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._config.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                        "temperature": 0,
                    },
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    content = self._strip_wrapping(content)
                    parsed = self._try_parse_json(content)
                    if parsed:
                        actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
                        log.info(
                            "Voice action extraction: %d action(s) from transcript",
                            len(actions),
                        )
                        return actions
        except Exception as exc:
            log.warning("Voice action extraction failed: %s", exc)

        return []

    # ------------------------------------------------------------------
    # GPT text streaming
    # ------------------------------------------------------------------

    async def stream_text(self, session: Session, user_text: str) -> None:
        """Stream a GPT response for a text turn.

        Sends ``text_chunk`` frames as tokens arrive, then a final
        ``turn_complete`` frame containing the cleaned text and extracted actions.
        """
        ws = session.ws

        if not self._config.openai_api_key:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "OpenAI API key not configured on the Orchestrator server.",
            }))
            return

        messages = self._build_chat_messages(session, user_text)
        full_text = await self._stream_gpt(session, messages)
        if full_text is None:
            return  # error already sent to client

        clean_text, actions = self.parse_actions(full_text)

        # GPT-4o intermittently omits ##ACTIONS## despite the prompt.  When the
        # response text clearly describes an action, use the voice extraction
        # fallback to recover the structured actions.
        if not actions and clean_text.strip():
            actions = await self.extract_actions_from_voice(session, clean_text)

        self._append_to_history(session, user_text, clean_text)

        await ws.send_text(json.dumps({
            "type": "turn_complete",
            "full_text": clean_text,
            "actions": actions,
        }))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_chat_messages(self, session: Session, user_text: str) -> list[dict]:
        system_prompt = self._prompt_builder.build(session)
        messages = [{"role": "system", "content": system_prompt}]
        for turn in session.history[-self._config.history_max_turns:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _stream_gpt(
        self, session: Session, messages: list[dict]
    ) -> str | None:
        """Stream from the OpenAI API and forward chunks to the client.

        Returns the full accumulated text, or None if an HTTP/request error
        occurred (the error frame has already been sent to the client).
        """
        ws = session.ws
        full_text = ""

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._config.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._config.openai_model,
                        "messages": messages,
                        "stream": True,
                        "max_tokens": 2048,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": (
                                f"OpenAI error {resp.status_code}: "
                                f"{body[:200].decode()}"
                            ),
                        }))
                        return None

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                                await ws.send_text(
                                    json.dumps({"type": "text_chunk", "text": delta})
                                )
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

        except httpx.RequestError as exc:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            return None

        return full_text

    def _append_to_history(
        self, session: Session, user_text: str, assistant_text: str
    ) -> None:
        session.history.append({"role": "user", "content": user_text})
        session.history.append({"role": "assistant", "content": assistant_text})
        max_entries = self._config.history_max_turns * 2
        if len(session.history) > max_entries:
            session.history = session.history[-max_entries:]

    def _build_compact_canvas_context(self, session: Session) -> str:
        blocks = (
            session.active_blocks
            if session.active_blocks
            else session.canvas_state.get("blocks", [])
        )
        lines = []
        for b in blocks[:8]:
            ports = b.get("ports", [])
            port_str = ", ".join(
                f'{"[out]" if p.get("is_output") else "[in]"} {p.get("name", "?")}'
                for p in ports
            )
            lines.append(
                f'  "{b.get("name", "?")}" '
                f'(type:{b.get("type", "?")}) '
                f'ports: {port_str or "(none)"}'
            )
        return "\n".join(lines) if lines else "  (canvas empty)"

    @staticmethod
    def _strip_wrapping(text: str) -> str:
        """Remove optional <json>...</json> wrappers and markdown fences."""
        if text.lower().startswith("<json>"):
            text = text[6:]
            close = text.lower().find("</json>")
            if close != -1:
                text = text[:close]
            text = text.strip()

        if text.startswith("```"):
            end = text.find("```", 3)
            text = (text[3:end] if end != -1 else text[3:]).strip()
            nl = text.find("\n")
            if nl != -1 and "{" not in text[:nl]:
                text = text[nl + 1:].strip()

        return text

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
