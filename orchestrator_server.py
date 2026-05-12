#!/usr/bin/env python3
"""
Grafux Orchestrator Server
--------------------------
A FastAPI WebSocket server that acts as the AI intelligence layer for the
Grafux desktop application.  Each connected client gets its own session that
manages:
  • Conversation history (last 20 turns)
  • Canvas state (blocks, ports, connections, values)
  • Active-blocks context (agent-button blocks)
  • Text conversation via OpenAI GPT streaming
  • Live voice conversation via Gemini BidiGenerateContent audio relay
  • Saved-block catalogue loaded from AWS S3

Protocol (one WebSocket per client session)
-------------------------------------------
Text frames are JSON.  Binary frames are raw PCM audio.

C++ → Server (text):
  { "type": "text_message",      "text": "...", "canvas_state": {...}, "active_blocks": [...] }
  { "type": "canvas_update",     "canvas_state": {...} }
  { "type": "set_active_blocks", "blocks": [...] }
  { "type": "start_voice" }
  { "type": "stop_voice"  }
  { "type": "ping"        }

C++ → Server (binary):  raw PCM audio (16 kHz, Int16, mono) for Gemini Live

Server → C++ (text):
  { "type": "session_ready" }
  { "type": "text_chunk",    "text": "..." }
  { "type": "turn_complete", "full_text": "...", "actions": [...] }
  { "type": "voice_started" }
  { "type": "voice_stopped" }
  { "type": "error",         "message": "..." }
  { "type": "pong"           }

Server → C++ (binary):  raw PCM audio (24 kHz, Int16, mono) from Gemini Live
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Query, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware

from orchestrator.auth import AuthService
from orchestrator.catalogue import CatalogueService
from orchestrator.config import Config
from orchestrator.handler import SessionHandler
from orchestrator.prompt import PromptBuilder
from orchestrator.reasoning import Reasoning

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Shared service instances (singletons for the lifetime of the process)
# ---------------------------------------------------------------------------

config = Config()
auth = AuthService(config)
catalogue = CatalogueService(config)
prompt_builder = PromptBuilder()
reasoning = Reasoning(config, prompt_builder)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Grafux Orchestrator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.websocket("/ws/session")
async def session_endpoint(
    websocket: WebSocket,
    token: str = Query(default=""),
    project_id: str = Query(default=""),
) -> None:
    """Main orchestrator WebSocket endpoint — one session per connected client."""
    payload = auth.decode_token(token)
    if payload is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()

    user_id = payload.get("sub") or payload.get("user_id") or "anonymous"
    username = payload.get("username") or payload.get("email") or user_id

    handler = SessionHandler(
        websocket=websocket,
        user_id=user_id,
        username=username,
        project_id=project_id,
        catalogue_service=catalogue,
        reasoning=reasoning,
        prompt_builder=prompt_builder,
        config=config,
    )
    await handler.run()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "grafux-orchestrator",
        "gemini": bool(config.gemini_api_key),
        "openai": bool(config.openai_api_key),
        "s3": bool(config.s3_bucket),
    }


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
