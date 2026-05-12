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
  { "type": "text_message",   "text": "...", "canvas_state": {...}, "active_blocks": [...] }
  { "type": "canvas_update",  "canvas_state": {...} }
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

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError, BotoCoreError
import jwt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from fastapi.middleware.cors import CORSMiddleware
import httpx
import websockets

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
# Environment / config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
JWT_SECRET       = os.environ.get("JWT_SECRET", "")
S3_BUCKET        = os.environ.get("S3_BUCKET", "")
AWS_REGION       = os.environ.get("AWS_REGION", "us-east-1")
OPENAI_MODEL     = os.environ.get("OPENAI_MODEL", "gpt-4o")

GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com"
    "/ws/google.ai.generativelanguage.v1beta"
    ".GenerativeService.BidiGenerateContent"
)

HISTORY_MAX_TURNS = 20

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Grafux Orchestrator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token.  Returns payload or None on failure."""
    if not JWT_SECRET or not token:
        if not JWT_SECRET:
            log.warning("JWT_SECRET not set – allowing unauthenticated connections")
            return {"sub": "anonymous", "username": "anonymous"}
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        log.warning("JWT decode failed: %s", exc)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# S3 catalogue helpers
# ─────────────────────────────────────────────────────────────────────────────

_s3_client = None

def _get_s3() -> Any:
    global _s3_client
    if _s3_client is None and (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        try:
            _s3_client = boto3.client("s3", region_name=AWS_REGION)
        except Exception as exc:
            log.warning("Could not create S3 client: %s", exc)
    return _s3_client


async def load_s3_catalogue(user_id: str, project_id: str, username: str = "") -> str:
    """Load the saved-block catalogue from S3.  Returns formatted text or fallback."""
    s3 = _get_s3()
    if not s3 or not S3_BUCKET:
        return "(S3 catalogue not available)"

    prefix = f"users/{user_id}/{username}/{project_id}/" if username else f"{user_id}/{project_id}/"
    block_types = [
        "tools", "topics", "commands", "procedures",
        "components", "memory", "selection", "filter", "devices",
    ]

    def _list_and_load():
        results = []
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key: str = obj["Key"]
                    # Match: {prefix}{block_type}/{block_name}/{block_name}.json
                    parts = key[len(prefix):].split("/")
                    if len(parts) >= 3 and parts[0] in block_types and parts[-1].endswith(".json"):
                        block_type = parts[0]
                        block_name = parts[-1][:-5]  # strip .json
                        try:
                            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                            data = json.loads(resp["Body"].read())
                            desc = data.get("description", "")
                            if not desc:
                                tcs = data.get("tool_calls", [])
                                if tcs:
                                    desc = tcs[0].get("params", {}).get("description", "")
                            line = f'  {block_name} | block_type="{block_type}" block_name="{block_name}"'
                            if desc:
                                line += f" — {desc[:80]}"
                            results.append(line)
                        except Exception:
                            results.append(
                                f'  {block_name} | block_type="{block_type}" block_name="{block_name}"'
                            )
        except (ClientError, BotoCoreError) as exc:
            log.warning("S3 catalogue load failed: %s", exc)
        return results

    loop = asyncio.get_event_loop()
    try:
        entries = await loop.run_in_executor(None, _list_and_load)
    except Exception as exc:
        log.warning("S3 executor error: %s", exc)
        return "(S3 catalogue load error)"

    if not entries:
        return "(No saved blocks found in this project)"
    return f"SAVED BLOCKS CATALOGUE ({len(entries)} blocks):\n" + "\n".join(entries)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    ws: WebSocket
    user_id: str = "anonymous"
    username: str = "anonymous"
    project_id: str = ""
    canvas_state: dict = field(default_factory=dict)
    active_blocks: list = field(default_factory=list)
    history: list = field(default_factory=list)   # [{role, content}, ...]
    catalogue: str = "(catalogue not loaded)"
    voice_active: bool = False
    gemini_task: asyncio.Task | None = None
    audio_to_gemini: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_from_gemini: asyncio.Queue = field(default_factory=asyncio.Queue)

# ─────────────────────────────────────────────────────────────────────────────
# System-prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(session: Session) -> str:
    """Build the full system prompt from canvas state, active blocks and catalogue."""
    canvas = session.canvas_state
    active = session.active_blocks

    prompt = (
        "You are a real-time AI assistant embedded in Grafux, a visual AI pipeline editor.\n\n"
        "You have FULL CONTROL over the canvas. You can:\n"
        "  • Run or regenerate any block\n"
        "  • Add, remove, or rename ports on any block\n"
        "  • Read and write port values\n"
        "  • Connect or disconnect ports between blocks\n"
        "  • Delete blocks from the canvas\n"
        "  • Load saved blocks from the project library onto the canvas\n"
        "  • Create brand-new blocks from scratch through conversation\n"
        "  • Answer questions about the diagram or Grafux in general\n\n"
        "Respond conversationally and concisely — the user may be speaking to you in real time.\n\n"
    )

    if active:
        prompt += f"ACTIVE BLOCKS (user has pressed agent button on {len(active)} block(s)):\n"
        for idx, blk in enumerate(active, 1):
            name    = blk.get("name", "?")
            btype   = blk.get("type", "?")
            bstatus = blk.get("status", "Idle")
            desc    = (blk.get("description") or "")[:120]
            prompt += f'{idx}. "{name}" (type: {btype}, status: {bstatus})\n'
            if desc:
                prompt += f"   Description: {desc}\n"
            ports = blk.get("ports", [])
            if ports:
                prompt += "   Ports:\n"
                for p in ports:
                    dir_ = "output" if p.get("is_output") else "input"
                    val  = (p.get("value") or "(empty)")[:100]
                    prompt += f'     [{dir_}] {p.get("name","?")}: {val}\n'
            conns = blk.get("connections", [])
            if conns:
                prompt += "   Connections:\n"
                for c in conns:
                    prompt += f'     {c.get("from_port")} --> "{c.get("to_block")}".{c.get("to_port")}\n'
            prompt += "\n"
    else:
        blocks = canvas.get("blocks", [])
        if not blocks:
            prompt += "CANVAS: (empty — no blocks on canvas)\n\n"
        else:
            prompt += f"CANVAS BLOCKS ({len(blocks)} block(s)):\n"
            for idx, blk in enumerate(blocks, 1):
                name    = blk.get("name", "?")
                btype   = blk.get("type", "?")
                bstatus = blk.get("status", "Idle")
                desc    = (blk.get("description") or "")[:120]
                prompt += f'{idx}. "{name}" (type: {btype}, status: {bstatus})\n'
                if desc:
                    prompt += f"   Description: {desc}\n"
                ports = blk.get("ports", [])
                if ports:
                    prompt += "   Ports:\n"
                    for p in ports:
                        dir_ = "output" if p.get("is_output") else "input"
                        val  = (p.get("value") or "(empty)")[:100]
                        prompt += f'     [{dir_}] {p.get("name","?")}: {val}\n'
                conns = blk.get("connections", [])
                if conns:
                    prompt += "   Connections:\n"
                    for c in conns:
                        prompt += f'     {c.get("from_port")} --> "{c.get("to_block")}".{c.get("to_port")}\n'
                prompt += "\n"

    prompt += session.catalogue + "\n\n"

    prompt += (
        "CANVAS STATE UPDATES:\n"
        "During the session you may receive text messages starting with '[Canvas state update]'. "
        "These are automatic notifications about port or value changes on the canvas. "
        "When you receive one: say exactly one short sentence acknowledging the change "
        "and do NOT emit any ##ACTIONS## tag.\n\n"
    )

    prompt += (
        "CRITICAL — HOW TO APPLY CHANGES:\n"
        "Whenever you perform an action you MUST append a machine-readable tag at the very end "
        "of your text response, on its own line, with NO XML tags or code fences around it:\n"
        '##ACTIONS##{ "actions":[...action objects...]}\n\n'
        "IMPORTANT: The JSON must use double quotes. Do NOT wrap it in <json> tags or markdown code fences.\n\n"
        'Most actions require "target_block": "<block_name>". '
        "connect_ports and disconnect_ports use from_block/to_block instead.\n\n"
        "Complete action vocabulary:\n"
        '  run_block:        {"type":"run_block","target_block":"Name"}\n'
        '  regenerate_block: {"type":"regenerate_block","target_block":"Name"}\n'
        '  add_port:         {"type":"add_port","target_block":"Name","direction":"input","port_name":"name"}\n'
        '  remove_port:      {"type":"remove_port","target_block":"Name","direction":"output","port_name":"name"}\n'
        '  rename_port:      {"type":"rename_port","target_block":"Name","direction":"input","old_port_name":"a","new_port_name":"b"}\n'
        '  set_port_value:   {"type":"set_port_value","target_block":"Name","direction":"input","port_name":"name","value":"val"}\n'
        '  set_description:  {"type":"set_description","target_block":"Name","description":"text"}\n'
        '  set_loop_time:    {"type":"set_loop_time","target_block":"Name","loop_count":3,"wait_time":5}\n'
        '  rename_block:     {"type":"rename_block","target_block":"OldName","new_name":"NewName"}\n'
        '  open_port:        {"type":"open_port","target_block":"Name","direction":"output","port_name":"name"}\n'
        '  connect_ports:    {"type":"connect_ports","from_block":"A","from_port":"out_port","to_block":"B","to_port":"in_port"}\n'
        '  disconnect_ports: {"type":"disconnect_ports","from_block":"A","from_port":"out_port","to_block":"B","to_port":"in_port"}\n'
        '  delete_block:     {"type":"delete_block","target_block":"Name"}\n'
        '  load_block:       {"type":"load_block","block_type":"VALUE_FROM_block_type_FIELD","block_name":"VALUE_FROM_block_name_FIELD"}\n'
        "    Each entry in the catalogue shows: block_type=\"...\" block_name=\"...\"\n"
        "    Copy those exact values into the load_block action.\n"
        '  create_block:     {"type":"create_block","block_type":"tools","block_name":"send_email",'
        '"description":"Sends an email","inputs":["recipient","subject","body"],"outputs":["result"],"category":"email","code":"import os, sys, json\\n@register_tool(...)\\ndef send_email_tool(args): ..."}\n'
        "    block_type must be one of: tools, topics, commands, procedures, components, devices, memory, selection, filter\n"
        "    For block_type=tools: ALWAYS include a 'code' field with the full Python implementation of the tool.\n"
        "    Python code contract — the generated code MUST:\n"
        "      1. import os, sys, json (and any needed stdlib/third-party modules).\n"
        "      2. Be decorated with @register_tool(name='NAME', description='DESC',\n"
        "           input_schema={'type':'object','properties':{PORT:{'type':'string'},...},'required':[INPUTS]}).\n"
        "      3. Define def NAME_tool(args): with this body:\n"
        "           script_dir = os.path.dirname(os.path.abspath(__file__))\n"
        "           tool_dir = os.path.dirname(script_dir)\n"
        "           output_dir = os.path.normpath(os.path.join(tool_dir, 'outputs'))\n"
        "           os.makedirs(output_dir, exist_ok=True)\n"
        "           Write 'running' to status.txt, '' to errors.txt and warnings.txt.\n"
        "           Copy own source to outputs/code.txt: open(os.path.abspath(__file__)) -> outputs/code.txt.\n"
        "           Read each input port: value_or_path = args.get(port_name, port_name);\n"
        "             if os.path.exists(value_or_path): read the file; else use value_or_path directly.\n"
        "           Perform the actual tool operation.\n"
        "           Write each output port result to outputs/PORTNAME.txt and the main result to outputs/results.txt.\n"
        "           Write 'success' to status.txt.\n"
        "           On exception: write error message to errors.txt and 'error' to status.txt.\n"
        "      4. In the JSON 'code' string value, encode every newline as \\n (standard JSON escape).\n\n"
        "Rules:\n"
        "  - direction must be exactly 'input' or 'output' (lowercase).\n"
        "  - port_name and block_name must use underscores (snake_case).\n"
        "  - For connect_ports: from_port must be an output port; to_port must be an input port.\n"
        "  - connect_ports and disconnect_ports do NOT use target_block; use from_block/to_block.\n"
        "  - If you are only answering a question, omit ##ACTIONS## entirely.\n\n"
        "CONFIRMATION REQUIRED — do NOT act until the user explicitly says yes:\n"
        "  load_block:   First describe the matching block and ask 'Would you like me to add [BlockName] to the canvas?'\n"
        "  create_block: Suggest type/name/ports, then ask 'Shall I create it?' Only emit after explicit confirmation.\n"
        "  delete_block: Always ask 'Are you sure you want to delete [BlockName]?' before emitting.\n\n"
        "EXECUTE IMMEDIATELY — no confirmation needed:\n"
        "  run_block, regenerate_block, set_port_value, add_port, remove_port, rename_port,\n"
        "  connect_ports, disconnect_ports, set_description, open_port, set_loop_time, rename_block.\n"
        "  MANDATORY: You MUST emit ##ACTIONS## whenever you perform any of these operations.\n"
        "  NEVER say 'I will run...' or 'I'm adding...' without ALSO emitting the ##ACTIONS## tag.\n"
        "  Emit ##ACTIONS## FIRST, then your verbal confirmation on the next line.\n"
        "  If you forget ##ACTIONS## the user's canvas will NOT update — always include it."
    )

    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────────

_ACTIONS_MARKER = "##ACTIONS##"


def parse_actions(text: str) -> tuple[str, list]:
    """Extract ##ACTIONS## block from text.  Returns (clean_text, actions_list)."""
    idx = text.find(_ACTIONS_MARKER)
    if idx == -1:
        return text, []

    clean_text = text[:idx].strip()
    json_str = text[idx + len(_ACTIONS_MARKER):].strip()

    # Strip optional <json>...</json> wrapper
    if json_str.lower().startswith("<json>"):
        json_str = json_str[6:]
        close = json_str.lower().find("</json>")
        if close != -1:
            json_str = json_str[:close]
        json_str = json_str.strip()

    # Strip markdown fences
    if json_str.startswith("```"):
        end = json_str.find("```", 3)
        json_str = (json_str[3:end] if end != -1 else json_str[3:]).strip()
        nl = json_str.find("\n")
        if nl != -1 and "{" not in json_str[:nl]:
            json_str = json_str[nl + 1:].strip()

    # Try parsing as-is first (valid JSON from LLM, possibly with single quotes
    # inside string values such as Python code in the 'code' field).
    # Only fall back to single-quote normalisation when the first attempt fails,
    # because blindly replacing every ' with " corrupts Python source code
    # embedded in the 'code' field.
    def _try_parse(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    data = _try_parse(json_str)
    if data is None and "'" in json_str:
        data = _try_parse(json_str.replace("'", '"'))

    if data is None:
        log.warning("Could not parse ##ACTIONS## JSON: %s", json_str[:200])
        return clean_text, []

    actions = data.get("actions", []) if isinstance(data, dict) else []
    return clean_text, actions

# ─────────────────────────────────────────────────────────────────────────────
# Voice action extraction (GPT fallback for Gemini Live)
# ─────────────────────────────────────────────────────────────────────────────

async def extract_actions_from_voice(session: Session, voice_transcript: str) -> list:
    """
    After a Gemini Live voice turn, the spoken transcript is natural language and
    never contains ##ACTIONS## JSON.  Call GPT-4o-mini with the transcript and a
    compact canvas/port context so it can emit the structured action JSON that the
    voice model omitted.
    """
    if not OPENAI_API_KEY or not voice_transcript.strip():
        return []

    # Build a compact context: block names + their ports
    context_blocks = session.active_blocks if session.active_blocks else session.canvas_state.get("blocks", [])
    ctx_lines = []
    for b in context_blocks[:8]:
        ports = b.get("ports", [])
        port_str = ", ".join(
            f'{"[out]" if p.get("is_output") else "[in]"} {p.get("name","?")}'
            for p in ports
        )
        ctx_lines.append(f'  "{b.get("name","?")}" (type:{b.get("type","?")}) ports: {port_str or "(none)"}')
    canvas_ctx = "\n".join(ctx_lines) if ctx_lines else "  (canvas empty)"

    prompt = (
        f"Canvas blocks:\n{canvas_ctx}\n\n"
        "The Grafux AI voice assistant just said:\n"
        f'"{voice_transcript.strip()}"\n\n'
        "Based on what the AI described doing, extract any canvas actions performed.\n"
        "Return ONLY a JSON object — no markdown, no explanation:\n"
        '{"actions": [...]}\n\n'
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
        "If the AI was only speaking (no canvas action performed), return: {\"actions\": []}"
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
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
                # Strip optional markdown fences
                if content.startswith("```"):
                    end = content.find("```", 3)
                    content = (content[3:end] if end != -1 else content[3:]).strip()
                    nl = content.find("\n")
                    if nl != -1 and "{" not in content[:nl]:
                        content = content[nl + 1:].strip()
                parsed = json.loads(content)
                actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
                log.info("Voice action extraction: %d action(s) from transcript", len(actions))
                return actions
    except Exception as exc:
        log.warning("Voice action extraction failed: %s", exc)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI text streaming
# ─────────────────────────────────────────────────────────────────────────────

async def stream_openai(session: Session, user_text: str) -> None:
    """Stream a GPT response for a text message.  Sends text_chunk and turn_complete."""
    ws = session.ws

    if not OPENAI_API_KEY:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": "OpenAI API key not configured on the Orchestrator server.",
        }))
        return

    system_prompt = build_system_prompt(session)
    messages = [{"role": "system", "content": system_prompt}]
    for turn in session.history[-HISTORY_MAX_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    full_text = ""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 2048,
                },
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"OpenAI error {resp.status_code}: {body[:200].decode()}",
                    }))
                    return

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
                            await ws.send_text(json.dumps({"type": "text_chunk", "text": delta}))
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    except httpx.RequestError as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        return

    clean_text, actions = parse_actions(full_text)

    # Fallback: GPT-4o intermittently omits ##ACTIONS## despite the prompt.
    # When the response text clearly describes an action (e.g. "I'll run the block"),
    # use the same GPT-4o-mini extraction pass used for voice transcripts.
    if not actions and clean_text.strip():
        actions = await extract_actions_from_voice(session, clean_text)

    session.history.append({"role": "user",      "content": user_text})
    session.history.append({"role": "assistant",  "content": clean_text})
    if len(session.history) > HISTORY_MAX_TURNS * 2:
        session.history = session.history[-(HISTORY_MAX_TURNS * 2):]

    await ws.send_text(json.dumps({
        "type":      "turn_complete",
        "full_text": clean_text,
        "actions":   actions,
    }))

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Live voice session
# ─────────────────────────────────────────────────────────────────────────────

async def run_gemini_voice_session(session: Session) -> None:
    """
    Manage a bidirectional Gemini Live audio session.

    Audio flow:
      C++ mic PCM → session.audio_to_gemini queue → Gemini
      Gemini audio → session.audio_from_gemini queue → C++ (sent as binary WS frames)
    """
    ws = session.ws

    if not GEMINI_API_KEY:
        await ws.send_text(json.dumps({
            "type": "error", "message": "GEMINI_API_KEY not configured on server.",
        }))
        return

    url = f"{GEMINI_LIVE_URL}?key={GEMINI_API_KEY}"

    system_prompt = build_system_prompt(session)
    setup_msg = {
        "setup": {
            # Must match the model used by the C++ GeminiLiveClient (gemini_live_client.cpp).
            # gemini-2.0-flash-live-001 is rejected with 1008 on v1beta bidiGenerateContent.
            "model": "models/gemini-2.5-flash-native-audio-preview-12-2025",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}},
                },
            },
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            # Enable transcription so we can read ##ACTIONS## from the AI's spoken text
            "inputAudioTranscription":  {},
            "outputAudioTranscription": {},
        }
    }

    try:
        async with websockets.connect(url, max_size=20 * 1024 * 1024) as gemini_ws:
            log.info("Gemini Live connected for user=%s", session.user_id)

            await gemini_ws.send(json.dumps(setup_msg))
            await ws.send_text(json.dumps({"type": "voice_started"}))

            async def _send_audio():
                while session.voice_active:
                    try:
                        pcm = await asyncio.wait_for(session.audio_to_gemini.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if pcm is None:
                        break
                    if isinstance(pcm, tuple) and pcm[0] == "__context__":
                        # Special sentinel: send raw JSON to Gemini directly
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

            async def _recv_gemini():
                transcript_buffer = ""
                async for raw in gemini_ws:
                    if not session.voice_active:
                        break
                    try:
                        data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if isinstance(raw, (bytes, bytearray)):
                            await ws.send_bytes(bytes(raw))
                        continue

                    sc = data.get("serverContent", {})

                    # Native-audio preview models surface AI transcript via outputTranscription
                    out_tr = sc.get("outputTranscription", {})
                    out_tr_text = out_tr.get("text", "")
                    if out_tr_text:
                        transcript_buffer += out_tr_text
                        await ws.send_text(json.dumps({
                            "type": "text_chunk", "text": out_tr_text,
                        }))

                    model_turn = sc.get("modelTurn", {})
                    parts = model_turn.get("parts", [])

                    for part in parts:
                        inline = part.get("inlineData", {})
                        if inline.get("mimeType", "").startswith("audio/"):
                            audio_b64 = inline.get("data", "")
                            if audio_b64:
                                pcm_bytes = base64.b64decode(audio_b64)
                                await ws.send_bytes(pcm_bytes)

                        text_part = part.get("text", "")
                        if text_part:
                            transcript_buffer += text_part
                            await ws.send_text(json.dumps({
                                "type": "text_chunk", "text": text_part,
                            }))

                    if sc.get("turnComplete"):
                        clean_text, actions = parse_actions(transcript_buffer)

                        # Fallback: voice model speaks natural language and never emits
                        # ##ACTIONS## JSON in its audio transcript.  Use GPT-4o-mini to
                        # extract the implied structured actions from what was spoken.
                        if not actions and transcript_buffer.strip():
                            actions = await extract_actions_from_voice(session, transcript_buffer)

                        if clean_text:
                            session.history.append({"role": "assistant", "content": clean_text})
                            if len(session.history) > HISTORY_MAX_TURNS * 2:
                                session.history = session.history[-(HISTORY_MAX_TURNS * 2):]
                        await ws.send_text(json.dumps({
                            "type":      "turn_complete",
                            "full_text": clean_text,
                            "actions":   actions,
                        }))
                        transcript_buffer = ""

                    if sc.get("interrupted"):
                        log.debug("Gemini interrupted current turn")

            send_task = asyncio.create_task(_send_audio())
            recv_task = asyncio.create_task(_recv_gemini())

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

    except websockets.exceptions.WebSocketException as exc:
        log.error("Gemini Live WS error: %s", exc)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"Gemini Live: {exc}"}))
        except Exception:
            pass
    except Exception as exc:
        log.error("Gemini Live unexpected error: %s", exc)
    finally:
        session.voice_active = False
        try:
            await ws.send_text(json.dumps({"type": "voice_stopped"}))
        except Exception:
            pass
        log.info("Gemini Live session ended for user=%s", session.user_id)

# ─────────────────────────────────────────────────────────────────────────────
# Canvas state helpers
# ─────────────────────────────────────────────────────────────────────────────

async def send_canvas_context_update(session: Session) -> None:
    """Inject a canvas-state summary into the active Gemini voice session."""
    if not session.voice_active:
        return

    blocks = session.canvas_state.get("blocks", [])
    if not blocks:
        return

    lines = ["[Canvas state update] Current state of all canvas blocks:"]
    for blk in blocks:
        name = blk.get("name", "?")
        lines.append(f'\nBlock "{name}" (type: {blk.get("type","?")}):')
        ports = blk.get("ports", [])
        if not ports:
            lines.append("  (no ports)")
        else:
            for p in ports:
                dir_ = "output" if p.get("is_output") else "input"
                val  = (p.get("value") or "(empty)")
                lines.append(f'  [{dir_}] {p.get("name","?")}: {val}')
    lines.append("\nPlease update your understanding of all canvas blocks accordingly.")

    client_content_msg = json.dumps({
        "clientContent": {
            "turns": [{"role": "user", "parts": [{"text": "\n".join(lines)}]}],
            "turnComplete": False,
        }
    })
    await session.audio_to_gemini.put(("__context__", client_content_msg))

# ─────────────────────────────────────────────────────────────────────────────
# Main WebSocket handler
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/session")
async def session_endpoint(
    websocket: WebSocket,
    token: str = Query(default=""),
    project_id: str = Query(default=""),
):
    """Main orchestrator WebSocket endpoint.  One session per connected client."""
    payload = decode_token(token)
    if payload is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()

    user_id  = payload.get("sub") or payload.get("user_id") or "anonymous"
    username = payload.get("username") or payload.get("email") or user_id

    session = Session(
        ws=websocket,
        user_id=user_id,
        username=username,
        project_id=project_id,
    )

    catalogue_task = asyncio.create_task(load_s3_catalogue(user_id, project_id, username))

    log.info("Session opened: user=%s project=%s", user_id, project_id)
    await websocket.send_text(json.dumps({"type": "session_ready"}))

    _canvas_update_pending = False
    _canvas_update_task: asyncio.Task | None = None

    async def _debounced_canvas_update():
        nonlocal _canvas_update_pending
        await asyncio.sleep(0.4)
        _canvas_update_pending = False
        await send_canvas_context_update(session)

    try:
        while True:
            msg = await websocket.receive()

            # Client disconnected — exit cleanly before calling receive() again
            if msg["type"] == "websocket.disconnect":
                log.info("Client disconnected: user=%s code=%s", user_id, msg.get("code"))
                break

            # ── Binary frame = PCM audio from mic ────────────────────────────
            if msg["type"] == "websocket.receive" and msg.get("bytes") is not None:
                if session.voice_active:
                    await session.audio_to_gemini.put(msg["bytes"])
                continue

            # ── Text frame = JSON command ────────────────────────────────────
            raw_text = msg.get("text", "")
            if not raw_text:
                continue

            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            elif msg_type == "canvas_update":
                session.canvas_state = data.get("canvas_state", {})
                if not _canvas_update_pending:
                    _canvas_update_pending = True
                    _canvas_update_task = asyncio.create_task(_debounced_canvas_update())

            elif msg_type == "set_active_blocks":
                session.active_blocks = data.get("blocks", [])

            elif msg_type == "text_message":
                if "canvas_state" in data:
                    session.canvas_state = data["canvas_state"]
                if "active_blocks" in data:
                    session.active_blocks = data["active_blocks"]
                if not catalogue_task.done():
                    try:
                        session.catalogue = await asyncio.wait_for(
                            asyncio.shield(catalogue_task), timeout=5.0
                        )
                    except (asyncio.TimeoutError, Exception):
                        session.catalogue = "(catalogue loading…)"
                else:
                    session.catalogue = catalogue_task.result()

                user_text = data.get("text", "").strip()
                if user_text:
                    asyncio.create_task(stream_openai(session, user_text))

            elif msg_type == "start_voice":
                if not session.voice_active:
                    if not catalogue_task.done():
                        try:
                            session.catalogue = await asyncio.wait_for(
                                asyncio.shield(catalogue_task), timeout=5.0
                            )
                        except (asyncio.TimeoutError, Exception):
                            session.catalogue = "(catalogue loading…)"
                    else:
                        session.catalogue = catalogue_task.result()

                    session.voice_active = True
                    while not session.audio_to_gemini.empty():
                        session.audio_to_gemini.get_nowait()
                    while not session.audio_from_gemini.empty():
                        session.audio_from_gemini.get_nowait()
                    session.gemini_task = asyncio.create_task(
                        run_gemini_voice_session(session)
                    )

            elif msg_type == "stop_voice":
                session.voice_active = False
                await session.audio_to_gemini.put(None)
                if session.gemini_task:
                    session.gemini_task.cancel()
                    session.gemini_task = None
                await websocket.send_text(json.dumps({"type": "voice_stopped"}))

            else:
                log.debug("Unknown message type: %s", msg_type)

    except WebSocketDisconnect:
        log.info("Session disconnected: user=%s", user_id)
    except Exception as exc:
        log.exception("Session error for user=%s: %s", user_id, exc)
    finally:
        session.voice_active = False
        await session.audio_to_gemini.put(None)
        if session.gemini_task:
            session.gemini_task.cancel()
        if _canvas_update_task:
            _canvas_update_task.cancel()
        catalogue_task.cancel()
        log.info("Session cleaned up: user=%s", user_id)

# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "grafux-orchestrator",
        "gemini":  bool(GEMINI_API_KEY),
        "openai":  bool(OPENAI_API_KEY),
        "s3":      bool(S3_BUCKET),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
