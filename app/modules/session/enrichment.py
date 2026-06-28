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


async def _enrich_topic_block(action: dict[str, Any], session_id: str) -> None:
    """Fill a newly created topic block's ports with grounded, current content."""
    from app.modules.blocks.router import generate_topic_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return
    try:
        result = await generate_topic_payload(
            topic_name=name,
            category=str(action.get("category", "")).strip() or "general",
            description=str(action.get("description", "")).strip(),
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
        )
        params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
        output_ports = params.get("output_ports") or []
        if not output_ports:
            return  # nothing grounded — leave action as-is (client makes a stub)
        action["output_ports"] = output_ports
        if params.get("input_ports"):
            action["input_ports"] = params["input_ports"]
        log.info(
            "create_topic_grounding_ok", session_id=session_id, block=name, ports=len(output_ports)
        )
    except Exception as exc:
        log.warning(
            "create_topic_grounding_failed", session_id=session_id, block=name, error=str(exc)
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


async def _enrich_image_block(action: dict[str, Any], session_id: str) -> None:
    """Lay out a newly created image block's standard ports.

    The image itself is produced at Run by the image service, so enrichment only
    scaffolds the input ports (prompt/modification/search_for) and output ports
    (image/image_name/image_description/improvements) — matching the manual
    UnifiedWindow path so voice/text-created image blocks have the right shape.
    """
    from app.modules.blocks.router import generate_image_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return
    try:
        result = await generate_image_payload(
            block_name=name,
            category=str(action.get("category", "")).strip() or "general",
            description=str(action.get("description", "")).strip(),
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
        )
        params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
        if params.get("output_ports"):
            action["output_ports"] = params["output_ports"]
        if params.get("input_ports"):
            action["input_ports"] = params["input_ports"]
        log.info("create_image_scaffold_ok", session_id=session_id, block=name)
    except Exception as exc:
        log.warning(
            "create_image_scaffold_failed", session_id=session_id, block=name, error=str(exc)
        )


# block_type → enricher. Replaces the if/elif dispatch chain.
_ENRICHERS = {
    "tools": _enrich_tool_block,
    "topics": _enrich_topic_block,
    "code": _enrich_code_block,
    "image": _enrich_image_block,
}
