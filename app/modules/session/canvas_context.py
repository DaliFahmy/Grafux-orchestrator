"""Pure helpers for rendering canvas/diagram state into model-readable text.

These functions have no session state — they turn the block/port dictionaries the
client sends into the text blocks injected into the chat and voice prompts. Kept
separate from the session orchestration so they can be unit-tested in isolation
(see ``tests/test_session_port_read.py``).
"""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger

log = get_logger("session.canvas_context")


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


def format_library_entry(entry: dict[str, Any]) -> str:
    """Format one saved-library entry for DIAGRAM_CONTEXT."""
    block_type = entry.get("block_type", "")
    block_name = entry.get("block_name", "")
    category = entry.get("category", "")
    if category:
        return (
            f'  block_type="{block_type}" category="{category}" block_name="{block_name}"'
        )
    return f'  block_type="{block_type}" block_name="{block_name}"'


def render_port(port: dict[str, Any]) -> str:
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


def render_block(block: dict[str, Any], *, active: bool) -> str:
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
        lines.extend(render_port(p) for p in ports if isinstance(p, dict))
    conns = block.get("connections") or []
    if conns:
        for c in conns:
            if isinstance(c, dict):
                lines.append(
                    f'  connection: {c.get("from_port","")} -> '
                    f'{c.get("to_block","")}.{c.get("to_port","")}'
                )
    return "\n".join(lines)


def render_canvas_context(
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
    active_payload = [
        b for b in (active_blocks or []) if isinstance(b, dict) and b.get("name")
    ]
    active_names = {b["name"] for b in active_payload}

    # Active blocks: prefer the richer active_blocks payload, fall back to canvas.
    seen: set[str] = set()
    active_rendered: list[str] = []
    for b in active_payload:
        active_rendered.append(render_block(b, active=True))
        seen.add(b["name"])
    for b in blocks:
        if not isinstance(b, dict):
            continue
        name = b.get("name")
        if name in active_names and name not in seen:
            active_rendered.append(render_block(b, active=True))
            seen.add(name)

    other_rendered: list[str] = [
        render_block(b, active=False)
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


def parse_actions_tag(full_text: str, session_id: str) -> tuple[str, list[Any]]:
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
