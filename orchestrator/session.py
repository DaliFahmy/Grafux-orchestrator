from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from fastapi import WebSocket


@dataclass
class Session:
    """All mutable state for a single connected client."""

    ws: WebSocket
    user_id: str = "anonymous"
    username: str = "anonymous"
    project_id: str = ""

    # Canvas state mirrored from the client
    canvas_state: dict = field(default_factory=dict)
    active_blocks: list = field(default_factory=list)

    # Conversation history: list of {"role": str, "content": str}
    history: list = field(default_factory=list)

    # S3-loaded block catalogue (formatted text)
    catalogue: str = "(catalogue not loaded)"

    # Voice session state
    voice_active: bool = False
    gemini_task: asyncio.Task | None = None
    audio_to_gemini: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_from_gemini: asyncio.Queue = field(default_factory=asyncio.Queue)
