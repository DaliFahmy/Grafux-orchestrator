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
import difflib
import json
import time
from typing import Any

from app.core.logging import get_logger

log = get_logger("session.enrichment")

# An enricher returns ``(status, detail)``: status is "ok" (content produced or nothing
# to do) or "failed" (a soft failure the client should surface, with ``detail`` the reason).
# An enricher that returns ``None`` (e.g. a test double) is treated as ("ok", "").
EnrichStatus = tuple[str, str]


def is_enrichable(action: Any) -> bool:
    """True when a ``create_block`` action has an enricher (so the turn marks it pending)."""
    if not isinstance(action, dict) or action.get("type") != "create_block":
        return False
    return str(action.get("block_type", "")).strip().lower() in _ENRICHERS


async def _dispatch(action: dict[str, Any], session_id: str) -> tuple[dict[str, Any], str, str]:
    """Run the matching enricher for one action; return ``(action, status, detail)``.

    Catches every exception so one enricher can never sink the others — a raise maps to
    ("failed", message). The action is mutated in place; the caller reads the patch keys off it.
    """
    block_type = str(action.get("block_type", "")).strip().lower()
    enricher = _ENRICHERS.get(block_type)
    if enricher is None:
        return action, "ok", ""
    try:
        result = await enricher(action, session_id)
    except Exception as exc:
        log.warning(
            "enrich_dispatch_failed", session_id=session_id,
            block=action.get("block_name"), block_type=block_type, error=str(exc),
        )
        return action, "failed", str(exc)
    if isinstance(result, tuple) and len(result) == 2:
        return action, str(result[0]), str(result[1])
    return action, "ok", ""


async def enrich_actions(actions: list[Any], session_id: str) -> None:
    """Enrich every ``create_block`` action in place, concurrently.

    Independent actions (a tool, a topic, a code block in one turn) each hit the
    network, so they run together rather than back-to-back. Non-streaming variant used
    by REST callers and tests; failures are swallowed (best-effort).
    """
    if not actions:
        return
    jobs = [_dispatch(a, session_id) for a in actions if is_enrichable(a)]
    if jobs:
        await asyncio.gather(*jobs, return_exceptions=True)


async def enrich_actions_streaming(actions: list[Any], session_id: str):
    """Enrich concurrently, yielding ``(action, status, detail)`` as each enricher finishes.

    Same dispatch/mutate-in-place contract as :func:`enrich_actions`, but lets the caller
    push each enriched action to the client the moment it's ready instead of waiting for the
    slowest one (web grounding can take 5-40s). Only enrichable ``create_block`` actions are
    dispatched — the rest are already final from ``turn_complete``.
    """
    tasks = [asyncio.create_task(_dispatch(a, session_id)) for a in actions if is_enrichable(a)]
    for fut in asyncio.as_completed(tasks):
        yield await fut


async def _enrich_tool_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Attach @register_tool-formatted Python to a newly created tool block."""
    from app.modules.blocks.router import (
        build_tool_codegen_message,
        generate_tool_code,
    )
    if str(action.get("code", "")).strip():
        return ("ok", "")  # model already supplied code inline
    name = str(action.get("block_name", "")).strip()
    description = str(action.get("description", "")).strip()
    if not name or not description:
        return ("ok", "")  # nothing to generate from — leave as-is
    code = await generate_tool_code(
        build_tool_codegen_message(
            tool_name=name,
            description=description,
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
        )
    )
    if not code.strip():
        return ("failed", "no tool code generated")
    action["code"] = code
    log.info("create_block_codegen_ok", session_id=session_id, block=name)
    return ("ok", "")


async def _enrich_search_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Fill a newly created search block's ports with AI-generated (grounded) content.

    Covers topics/components/procedures/commands — each uses its own ``[create_*]`` prompt
    section, and only the groundable types pull live web data (see ``generate_topic_payload``).
    """
    from app.modules.blocks.router import generate_topic_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")
    block_type = str(action.get("block_type", "")).strip().lower() or "topics"
    result = await generate_topic_payload(
        topic_name=name,
        category=str(action.get("category", "")).strip() or "general",
        description=str(action.get("description", "")).strip(),
        inputs=action.get("inputs") or [],
        outputs=action.get("outputs") or [],
        block_type=block_type,
    )
    if result is None:
        return ("failed", "AI not configured")
    params = result.get("tool_calls", [{}])[0].get("params", {})
    output_ports = params.get("output_ports") or []
    if not output_ports:
        return ("failed", "no entities generated (grounding returned nothing)")
    action["output_ports"] = output_ports
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]
    log.info(
        "create_search_grounding_ok",
        session_id=session_id, block=name, block_type=block_type, ports=len(output_ports),
    )
    return ("ok", "")


async def _enrich_code_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Fill a newly created code block's ports with AI-generated source code."""
    from app.modules.blocks.router import generate_code_payload

    name = str(action.get("block_name", "")).strip()
    description = str(action.get("description", "")).strip()
    if not name or not description:
        return ("ok", "")
    language = str(action.get("language", "")).strip() or "python"
    result = await generate_code_payload(
        block_name=name,
        category=str(action.get("category", "")).strip() or "general",
        description=description,
        language=language,
        inputs=action.get("inputs") or [],
        outputs=action.get("outputs") or [],
    )
    if result is None:
        return ("failed", "AI not configured")
    params = result.get("tool_calls", [{}])[0].get("params", {})
    output_ports = params.get("output_ports") or []
    if not output_ports:
        return ("failed", "no code generated")
    action["output_ports"] = output_ports
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]
    log.info("create_code_codegen_ok", session_id=session_id, block=name, language=language)
    return ("ok", "")


# Seed keys the model may attach to a create_block action, forwarded to the scaffold
# builder so the primary input port is pre-filled from the command (e.g. an address or URL).
_SCAFFOLD_SEED_KEYS = ("address", "url", "gpu_model", "language", "top", "pdk",
                       "clock_period", "spec", "explanation")


# The claw design ports the devices scaffolder drafts from a description. Secrets
# (credentials/api_keys) are intentionally excluded — they stay empty for the user.
_CLAW_DESIGN_KEYS = ("soul", "skills", "agent", "task", "memory", "tools_config", "connections")

# Cached Composio toolkit slug list (fallback normalization when the devices scaffolder is
# unreachable but the model named apps explicitly). Refetched after the TTL lapses.
_TOOLKITS_TTL = 3600.0
_toolkits_cache: dict[str, Any] = {"at": 0.0, "slugs": []}


async def _cached_toolkits(client: Any) -> list[str]:
    """The claw toolkit slugs, cached with a 1h TTL (best-effort; ``[]`` when unavailable)."""
    now = time.monotonic()
    cached = _toolkits_cache["slugs"]
    if cached and (now - _toolkits_cache["at"]) < _TOOLKITS_TTL:
        return cached
    slugs = await client.list_claw_toolkits()
    if slugs:
        _toolkits_cache["slugs"] = slugs
        _toolkits_cache["at"] = now
    return slugs


def _normalize_apps(apps: list[str], slugs: list[str]) -> list[str]:
    """Resolve app names to valid toolkit slugs (fuzzy, cutoff 0.8), de-duped (order preserved)."""
    slugset = {s.lower() for s in slugs}
    out: list[str] = []
    seen: set[str] = set()
    for app in apps:
        a = str(app).strip().lower()
        if not a:
            continue
        if slugset and a not in slugset:
            match = difflib.get_close_matches(a, sorted(slugset), n=1, cutoff=0.8)
            if match:
                a = match[0]
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _set_claw_port(action: dict[str, Any], port_name: str, content: str) -> None:
    """Write ``content`` into the named input port, creating it if the scaffold omitted it."""
    ports = action.get("input_ports") or []
    for p in ports:
        if str(p.get("port_name", "")).strip().lower() == port_name:
            p["port_content"] = content
            return
    base_path = ""
    for p in ports:
        pp = str(p.get("port_path", ""))
        if "/inputs/" in pp:
            base_path = pp.rsplit("/", 1)[0]
            break
    new_port: dict[str, Any] = {"port_name": port_name, "port_content": content}
    if base_path:
        new_port["port_path"] = f"{base_path}/{port_name}.txt"
    ports.append(new_port)
    action["input_ports"] = ports


async def _enrich_claw_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Scaffold a claw block AND draft its design ports (connections/soul/task/…).

    Lays out the canonical claw ports first (no AI — the always-safe fallback), then best-effort
    calls the devices ``/claw/scaffold`` endpoint, which AI-drafts the design ports from the
    description and normalizes ``connections`` to valid Composio toolkit slugs (e.g. a request to
    "send info to Google Sheets" fills the ``connections`` port with ``["googlesheets"]``). Any
    explicit ``connections`` the model supplied are passed as a hint and honored on failure.
    """
    from app.modules.blocks.router import generate_scaffold_payload
    from app.modules.devices.client import DevicesClient

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")

    # 1. Canonical port layout — never fails, so the block is always well-formed.
    scaffold = await generate_scaffold_payload(
        block_type="claw",
        block_name=name,
        category=str(action.get("category", "")).strip() or "general",
        description=str(action.get("description", "")).strip(),
        inputs=action.get("inputs") or [],
        outputs=action.get("outputs") or [],
    )
    params = (scaffold or {}).get("tool_calls", [{}])[0].get("params", {})
    if params.get("output_ports"):
        action["output_ports"] = params["output_ports"]
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]

    # 2. Draft the design ports (soul/task/connections/…) via the devices scaffolder.
    description = str(action.get("description", "")).strip()
    explicit = [str(a).strip() for a in (action.get("connections") or []) if str(a).strip()]
    client = DevicesClient()
    drafted = await client.scaffold_claw(description, name, connections=explicit)

    if drafted:
        for key in _CLAW_DESIGN_KEYS:
            val = str(drafted.get(key, "") or "").strip()
            if val:
                _set_claw_port(action, key, val)
    elif explicit:
        # Devices scaffolder unreachable — still honor the user's explicit apps.
        normalized = _normalize_apps(explicit, await _cached_toolkits(client))
        if normalized:
            _set_claw_port(action, "connections", json.dumps(normalized))

    log.info(
        "create_claw_ok", session_id=session_id, block=name,
        drafted=bool(drafted), connections=bool(explicit or (drafted or {}).get("connections")),
    )
    return ("ok", "")


async def _enrich_scaffold_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Lay out a newly created scaffold block's canonical ports (no AI call).

    Covers image/location/live/stream/white_board/gpu/claw/devices/selection/filter — blocks whose
    real content is produced at *Run* (by a Grafux-interaction or devices service) or by wiring.
    Enrichment only scaffolds the ports — matching the manual UnifiedWindow path — and seeds the
    primary input from the command so the block is ready to Run.
    """
    from app.modules.blocks.router import generate_scaffold_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")
    block_type = str(action.get("block_type", "")).strip().lower()
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
    return ("ok", "")


async def _enrich_memory_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Scaffold a memory block's ports for the mode it was asked for.

    Memory is the one scaffold type whose ports depend on a PARAM, not just the
    type: snapshot, sequential, accumulate and merge are four different blocks.
    Without this the generic scaffolder would always lay out the snapshot shape, so
    "create an accumulate memory block" produced a snapshot one with no way to tell.

    The normalised mode is written back onto the action because the client needs it
    too — ``memory_mode`` is persisted on the block and decides what Run does.
    """
    from app.modules.blocks.router import (
        MEMORY_MODES,
        generate_scaffold_payload,
        memory_scaffold_spec,
    )

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")

    mode = str(action.get("memory_mode", "")).strip().lower()
    if mode not in MEMORY_MODES:
        mode = "snapshot"
    action["memory_mode"] = mode

    result = await generate_scaffold_payload(
        block_type="memory",
        block_name=name,
        category=str(action.get("category", "")).strip() or "general",
        description=str(action.get("description", "")).strip(),
        inputs=action.get("inputs") or [],
        outputs=action.get("outputs") or [],
        spec_override=memory_scaffold_spec(mode),
    )
    params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
    if params.get("output_ports"):
        action["output_ports"] = params["output_ports"]
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]
    log.info("create_memory_ok", session_id=session_id, block=name, memory_mode=mode)
    return ("ok", "")


async def _enrich_testbench_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Scaffold a testbench block's ports and, when there is a spec, generate the tests.

    The spec comes from the action's ``spec`` seed or, failing that, the description
    ("create a testbench for the FIFO that ignores writes when full" IS the spec in
    practice). RTL is rarely available at voice-create time — the code block it will
    be wired to may not exist yet — so the interface is left for the model to infer
    from the spec and Run re-generates once ``rtl`` is wired. Without any spec the
    block is scaffolded empty, ready to fill and Run.
    """
    from app.modules.blocks.router import generate_scaffold_payload, generate_testbench_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")
    category = str(action.get("category", "")).strip() or "general"
    description = str(action.get("description", "")).strip()
    spec = str(action.get("spec", "")).strip() or description
    top = str(action.get("top", "")).strip()

    result = None
    if spec:
        result = await generate_testbench_payload(
            block_name=name,
            category=category,
            description=description,
            spec=spec,
            rtl=str(action.get("rtl", "")).strip(),
            top=top,
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
        )
    status: EnrichStatus = ("ok", "")
    if result is None:
        result = await generate_scaffold_payload(
            block_type="testbench",
            block_name=name,
            category=category,
            description=description,
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
            seeds={k: action[k] for k in ("top", "spec") if str(action.get(k, "")).strip()},
        )
        if spec:
            status = ("failed", "AI not configured")
    params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
    if params.get("output_ports"):
        action["output_ports"] = params["output_ports"]
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]
    log.info(
        "create_testbench_ok", session_id=session_id, block=name,
        generated=bool(spec) and status[0] == "ok",
    )
    return status


async def _enrich_code_hdl_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Scaffold a code_hdl block's ports and, when there is a spec, generate the design.

    Mirrors :func:`_enrich_testbench_block`, and for the same reason: the spec comes
    from the action's ``spec`` seed or, failing that, the description ("an 8-entry
    FIFO that ignores writes when full" IS the spec in practice). The top module
    defaults to the block name so the testbench and simulator have something to
    address before the user has typed anything. Without any spec the block is
    scaffolded empty, ready to fill and Run.
    """
    from app.modules.blocks.router import generate_code_hdl_payload, generate_scaffold_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")
    category = str(action.get("category", "")).strip() or "general"
    description = str(action.get("description", "")).strip()
    spec = str(action.get("spec", "")).strip() or description
    top = str(action.get("top", "")).strip() or name.replace(" ", "_")
    language = str(action.get("language", "")).strip() or "systemverilog"

    result = None
    status: EnrichStatus = ("ok", "")
    if spec:
        try:
            result = await generate_code_hdl_payload(
                block_name=name,
                category=category,
                description=description,
                spec=spec,
                language=language,
                top=top,
                inputs=action.get("inputs") or [],
                outputs=action.get("outputs") or [],
            )
        except ValueError as exc:
            # A non-HDL language reached the block (the chat guessed "python").
            # Scaffold it rather than failing the whole create: the user can pick a
            # language on the block and Run.
            log.warning("create_code_hdl_rejected", session_id=session_id, block=name,
                        error=str(exc))
            status = ("failed", str(exc))
            language = "systemverilog"
    if result is None:
        result = await generate_scaffold_payload(
            block_type="code_hdl",
            block_name=name,
            category=category,
            description=description,
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
            seeds={"top": top, "spec": spec, "language": language},
        )
        if spec and status[0] == "ok":
            status = ("failed", "AI not configured")
    params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
    if params.get("output_ports"):
        action["output_ports"] = params["output_ports"]
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]
    log.info(
        "create_code_hdl_ok", session_id=session_id, block=name, language=language,
        generated=bool(spec) and status[0] == "ok",
    )
    return status


async def _enrich_spec_hdl_block(action: dict[str, Any], session_id: str) -> EnrichStatus:
    """Scaffold a spec_hdl block's ports and write the specification from the description.

    The third of the spec-driven trio, and the one where the description is most
    obviously the real input: "create a spec for an 8-entry FIFO that ignores
    writes when full" IS the explanation, so there is nothing for the user to
    re-type. The `spec` seed key is accepted as an alias because the chat model
    has been told to pass the shared specification under that name for the whole
    HDL family, and a user who says "with this spec: ..." should not get an empty
    block for using the word the tool schema uses.

    A non-HDL language is scaffolded rather than failed, exactly as on code_hdl:
    the chat guesses "python" often enough that losing the whole block over it
    would be worse than a block whose language port needs one correction.
    """
    from app.modules.blocks.router import generate_scaffold_payload, generate_spec_hdl_payload

    name = str(action.get("block_name", "")).strip()
    if not name:
        return ("ok", "")
    category = str(action.get("category", "")).strip() or "general"
    description = str(action.get("description", "")).strip()
    explanation = (
        str(action.get("explanation", "")).strip()
        or str(action.get("spec", "")).strip()
        or description
    )
    top = str(action.get("top", "")).strip()
    language = str(action.get("language", "")).strip() or "systemverilog"

    result = None
    status: EnrichStatus = ("ok", "")
    if explanation:
        try:
            result = await generate_spec_hdl_payload(
                block_name=name,
                category=category,
                description=description,
                explanation=explanation,
                language=language,
                top=top,
                inputs=action.get("inputs") or [],
                outputs=action.get("outputs") or [],
            )
        except ValueError as exc:
            log.warning("create_spec_hdl_rejected", session_id=session_id, block=name,
                        error=str(exc))
            status = ("failed", str(exc))
            language = "systemverilog"
    if result is None:
        result = await generate_scaffold_payload(
            block_type="spec_hdl",
            block_name=name,
            category=category,
            description=description,
            inputs=action.get("inputs") or [],
            outputs=action.get("outputs") or [],
            seeds={"top": top, "spec": explanation, "language": language},
        )
        if explanation and status[0] == "ok":
            status = ("failed", "AI not configured")
    params = (result or {}).get("tool_calls", [{}])[0].get("params", {})
    if params.get("output_ports"):
        action["output_ports"] = params["output_ports"]
    if params.get("input_ports"):
        action["input_ports"] = params["input_ports"]
    log.info(
        "create_spec_hdl_ok", session_id=session_id, block=name, language=language,
        generated=bool(explanation) and status[0] == "ok",
    )
    return status


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
    # claw also drafts its design ports (connections/soul/task/…) via the devices scaffolder.
    "claw": _enrich_claw_block,
    "devices": _enrich_scaffold_block,
    # memory picks its ports from memory_mode, so it needs more than the generic
    # scaffolder (snapshot / sequential / accumulate are three different blocks).
    "memory": _enrich_memory_block,
    "selection": _enrich_scaffold_block,
    "filter": _enrich_scaffold_block,
    "white_board": _enrich_scaffold_block,
    # Chip design (EDA). Scaffold-only: the RTL, netlist and GDS all arrive at
    # Run from the tools themselves, so there is nothing for an LLM to draft here
    # beyond the ports and their defaults (pdk, clock period, stage range).
    "verilator": _enrich_scaffold_block,
    "yosys": _enrich_scaffold_block,
    "openroad": _enrich_scaffold_block,
    # The testbench IS AI content (tests derived from the spec), so it generates at
    # create time like code, falling back to a port scaffold without a spec/key.
    "testbench": _enrich_testbench_block,
    # So is the design it verifies: spec in, synthesizable RTL out.
    "code_hdl": _enrich_code_hdl_block,
    # ...and so is the contract BOTH of those are written from.
    "spec_hdl": _enrich_spec_hdl_block,
}
