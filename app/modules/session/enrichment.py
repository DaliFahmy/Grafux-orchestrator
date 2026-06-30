"""Fill newly created blocks with real content (code / grounded ports).

When a chat or voice turn emits ``create_block`` actions, the model only supplies
a shell. These helpers populate the heavy content — ``@register_tool`` Python for
tool blocks, grounded source-cited ports for topics, generated source for code —
by reusing the same generators the manual UnifiedWindow dialog calls, so the
client writes a populated block instead of an empty stub.

Every enricher is best-effort: any failure leaves the action unchanged so the
turn still completes and the client falls back to its own behavior. Shared by the
text turn and the voice relay so both produce identical blocks.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.logging import get_logger

log = get_logger("session.enrichment")


async def enrich_actions(actions: list[Any], session_id: str) -> None:
    """Enrich every ``create_block`` action in place, concurrently.

    Independent actions (a tool, a topic, a code block in one turn) each hit the
    network, so they run together rather than back-to-back.
    """
    if not actions:
        return
    jobs = []
    for action in actions:
        if not isinstance(action, dict) or action.get("type") != "create_block":
            continue
        block_type = str(action.get("block_type", "")).strip().lower()
        enricher = _ENRICHERS.get(block_type)
        if enricher is not None:
            jobs.append(enricher(action, session_id))
    if jobs:
        await asyncio.gather(*jobs, return_exceptions=True)


async def _enrich_tool_block(action: dict[str, Any], session_id: str) -> None:
    """Attach @register_tool-formatted Python to a newly created tool block."""
    from app.modules.blocks.router import (
        build_tool_codegen_message,
        generate_tool_code,
    )
    if str(action.get("code", "")).strip():
        return  # model already supplied code inline
    name = str(action.get("block_name", "")).strip()
    description = str(action.get("description", "")).strip()
    if not name or not description:
        return
    try:
        action["code"] = await generate_tool_code(
            build_tool_codegen_message(
                tool_name=name,
                description=description,
                inputs=action.get("inputs") or [],
                outputs=action.get("outputs") or [],
            )
        )
        log.info("create_block_codegen_ok", session_id=session_id, block=name)
    except Exception as exc:
        log.warning(
            "create_block_codegen_failed", session_id=session_id, block=name, error=str(exc)
        )


async def _enrich_search_block(action: dict[str, Any], session_id: str) -> None:
    """Fill a newly created search block's ports with AI-generated (grounded) content.

    Covers topics/components/procedures/commands — each uses its own ``[create_*]`` prompt
    section, and only the groundable types pull live web data (see ``generate_topic_payload``).
    """
    from app.modules.blocks.router import generate_topic_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return
    block_type = str(action.get("block_type", "")).strip().lower() or "topics"
    try:
        result = await generate_topic_payload(
            topic_name=name,
            category=str(action.get("category", "")).strip() or "general",
            description=str(action.get("description", "")).strip(),
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
            block_type=block_type,
        )
        params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
        output_ports = params.get("output_ports") or []
        if not output_ports:
            return  # nothing generated — leave action as-is (client makes a stub)
        action["output_ports"] = output_ports
        if params.get("input_ports"):
            action["input_ports"] = params["input_ports"]
        log.info(
            "create_search_grounding_ok",
            session_id=session_id, block=name, block_type=block_type, ports=len(output_ports),
        )
    except Exception as exc:
        log.warning(
            "create_search_grounding_failed",
            session_id=session_id, block=name, block_type=block_type, error=str(exc),
        )


async def _enrich_code_block(action: dict[str, Any], session_id: str) -> None:
    """Fill a newly created code block's ports with AI-generated source code."""
    from app.modules.blocks.router import generate_code_payload

    name = str(action.get("block_name", "")).strip()
    description = str(action.get("description", "")).strip()
    if not name or not description:
        return
    try:
        language = str(action.get("language", "")).strip() or "python"
        result = await generate_code_payload(
            block_name=name,
            category=str(action.get("category", "")).strip() or "general",
            description=description,
            language=language,
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
        )
        params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
        output_ports = params.get("output_ports") or []
        if not output_ports:
            return  # nothing generated — leave action as-is (client makes a stub)
        action["output_ports"] = output_ports
        if params.get("input_ports"):
            action["input_ports"] = params["input_ports"]
        log.info(
            "create_code_codegen_ok", session_id=session_id, block=name, language=language
        )
    except Exception as exc:
        log.warning(
            "create_code_codegen_failed", session_id=session_id, block=name, error=str(exc)
        )


# Seed keys the model may attach to a create_block action, forwarded to the scaffold
# builder so the primary input port is pre-filled from the command (e.g. an address or URL).
_SCAFFOLD_SEED_KEYS = ("address", "url", "gpu_model", "language")


async def _enrich_scaffold_block(action: dict[str, Any], session_id: str) -> None:
    """Lay out a newly created scaffold block's canonical ports (no AI call).

    Covers image/location/live/stream/gpu/claw/devices/memory/selection/filter — blocks whose
    real content is produced at *Run* (by a Grafux-interaction or devices service) or by wiring.
    Enrichment only scaffolds the ports — matching the manual UnifiedWindow path — and seeds the
    primary input from the command so the block is ready to Run.
    """
    from app.modules.blocks.router import generate_scaffold_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return
    block_type = str(action.get("block_type", "")).strip().lower()
    try:
        seeds = {k: action[k] for k in _SCAFFOLD_SEED_KEYS if str(action.get(k, "")).strip()}
        result = await generate_scaffold_payload(
            block_type=block_type,
            block_name=name,
            category=str(action.get("category", "")).strip() or "general",
            description=str(action.get("description", "")).strip(),
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
            seeds=seeds,
        )
        params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
        if params.get("output_ports"):
            action["output_ports"] = params["output_ports"]
        if params.get("input_ports"):
            action["input_ports"] = params["input_ports"]
        log.info("create_scaffold_ok", session_id=session_id, block=name, block_type=block_type)
    except Exception as exc:
        log.warning(
            "create_scaffold_failed",
            session_id=session_id, block=name, block_type=block_type, error=str(exc),
        )


# block_type → enricher. ``tools``/``code`` generate real content (Python / source); the
# search types are AI-generated (and grounded where applicable); everything else is a port
# scaffold whose content arrives at Run. Types absent here fall through unenriched (the client
# builds a generic stub).
_ENRICHERS = {
    "tools": _enrich_tool_block,
    "code": _enrich_code_block,
    # AI search blocks (own [create_*] prompt section; grounded where applicable).
    "topics": _enrich_search_block,
    "components": _enrich_search_block,
    "procedures": _enrich_search_block,
    "commands": _enrich_search_block,
    # Scaffold-only blocks (content produced at Run or by wiring).
    "image": _enrich_scaffold_block,
    "location": _enrich_scaffold_block,
    "live": _enrich_scaffold_block,
    "stream": _enrich_scaffold_block,
    "gpu": _enrich_scaffold_block,
    "claw": _enrich_scaffold_block,
    "devices": _enrich_scaffold_block,
    "memory": _enrich_scaffold_block,
    "selection": _enrich_scaffold_block,
    "filter": _enrich_scaffold_block,
}
