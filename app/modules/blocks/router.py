from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import time
import uuid
from typing import NamedTuple

from fastapi import APIRouter

from app.config import get_settings
from app.core.constants import (
    BLOCK_TYPE_SECTION as _BLOCK_TYPE_SECTION,
)
from app.core.constants import (
    CODE_FIX_RTL_SECTION as _CODE_FIX_RTL_SECTION,
)
from app.core.constants import (
    GROUNDABLE_BLOCK_TYPES as _GROUNDABLE_BLOCK_TYPES,
)
from app.core.llm import _strip_code_fences, call_llm_json, call_llm_text
from app.core.logging import get_logger
from app.dependencies import CurrentUser
from app.modules.blocks.hdl import (
    design_interface,
    hdl_family,
    infer_top_module,
    module_ports,
    validate_hdl_design,
    validate_rtl_fix,
    validate_spec,
    validate_testbench,
)
from app.modules.blocks.schemas import (
    CodeGenerateRequest,
    CodeHdlGenerateRequest,
    ImageGenerateRequest,
    ImprovementsRequest,
    RegenerateFilterRequest,
    RegenerateToolRequest,
    RunAccumulateRequest,
    RunFilterRequest,
    RunSearchRequest,
    RunSelectionRequest,
    SpecHdlGenerateRequest,
    TestbenchGenerateRequest,
    TopicGenerateRequest,
)
from app.prompts import get_json_schema, get_system_prompt

log = get_logger("blocks.router")

router = APIRouter(prefix="/blocks", tags=["blocks"])


def _make_port_path(category: str, name: str, port_type: str, port_name: str) -> str:
    return f"data/topics/{category}/{name}/{port_type}/{port_name}.txt"


def _simple_topic_response(body: TopicGenerateRequest) -> dict:
    """Fallback: build a minimal block without AI when OpenAI is not configured."""
    name = body.topic_name.replace(" ", "_")
    cat = body.category

    ip = [
        {
            "port_name": "block_description",
            "port_content": body.description,
            "port_path": _make_port_path(cat, name, "inputs", "block_description"),
        }
    ]
    for inp in body.inputs:
        if inp and inp not in ("description", "block_description"):
            ip.append({
                "port_name": inp,
                "port_content": "",
                "port_path": _make_port_path(cat, name, "inputs", inp),
            })

    op = []
    for out in body.outputs:
        if out:
            op.append({
                "port_name": out,
                "port_content": "",
                "port_path": _make_port_path(cat, name, "outputs", out),
            })

    return {
        "tool_calls": [
            {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "block_id": uuid.uuid4().hex[:8],
                    "block_type": "topics",
                    "x": 0,
                    "y": 0,
                    "input_ports": ip,
                    "output_ports": op,
                },
            }
        ],
        "connections": [],
    }


# A short/empty description is a user *query* ("best 5 ai tools 2026"), so we ground it in
# live web results. A long description is treated as the user's own source material and
# trusted as-is — no live search.
_GROUND_DESCRIPTION_MAX_CHARS = 400


async def _gather_topic_grounding(
    query: str,
    max_results: int = 6,
    crawl_top: int = 1,
) -> tuple[str, list[str]]:
    """Fetch live web sources so topic entities are real and current.

    Reuses the research module (Tavily search + optional FireCrawl deep-crawl). Returns
    ``(context_text, citations)`` for injection into the topic prompt, or ``("", [])`` when
    Tavily is not configured or anything fails — grounding must never break generation.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        return "", []

    try:
        from app.modules.research.citations import (
            extract_citations,
            format_sources_for_context,
        )
        from app.modules.research.firecrawl_client import FirecrawlClient
        from app.modules.research.tavily_client import TavilyClient

        sources = await TavilyClient().search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
            search_depth="advanced",
        )
        if not sources:
            log.info("topic_grounding_empty", query=query)
            return "", []

        # Deep-crawl the top result(s) for richer, current content (concurrently).
        if crawl_top > 0 and settings.firecrawl_api_key:
            firecrawl = FirecrawlClient()
            targets = sources[:crawl_top]
            scraped_results = await asyncio.gather(
                *(firecrawl.scrape(s.url) for s in targets),
                return_exceptions=True,
            )
            for source, scraped in zip(targets, scraped_results, strict=False):
                if isinstance(scraped, BaseException):
                    log.warning("topic_grounding_scrape_failed", url=source.url, error=str(scraped))
                    continue
                md = (scraped or {}).get("markdown", "")
                if md:
                    source.content = md[:3000]

        log.info("topic_grounding_ok", query=query, sources=len(sources))
        return format_sources_for_context(sources), extract_citations(sources)
    except Exception as exc:
        log.warning("topic_grounding_failed", query=query, error=str(exc))
        return "", []


async def _augment_with_grounding(user_message: str, query: str) -> str:
    """Append live web results + citations to a user message, or return it unchanged.

    Shared by the topic generator and the run/search path so both inject grounding the
    same way. Returns ``user_message`` untouched when grounding yields nothing (Tavily
    unconfigured, empty results, or any failure) — grounding must never break generation.
    """
    context_text, citations = await _gather_topic_grounding(query)
    if not context_text:
        return user_message
    today = datetime.date.today().isoformat()
    return user_message + (
        f"\n\nLIVE WEB SEARCH RESULTS (fetched {today}; treat these as the provided "
        "content — extract facts ONLY from them and prefer the most recent / current "
        f"information):\n{context_text}\n\n"
        "SOURCES (append the matching one to each port's content as "
        '"Source: <title> (date) — <url>"):\n'
        + "\n".join(citations)
    )


async def generate_topic_payload(
    *,
    topic_name: str,
    category: str = "general",
    description: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    ground: bool | None = None,
    model: str | None = None,
    block_type: str = "topics",
) -> dict | None:
    """Generate a grounded search-block envelope (the `tool_calls` dict) via AI.

    Shared core of the topic/component/procedure/command generators: builds the prompt, runs
    live web grounding when the block type is groundable and the description is a short query,
    calls the LLM with that type's ``[create_*]`` prompt section, and returns the parsed
    ``{tool_calls, ...}`` result with output ports filled with real, current content.

    ``block_type`` selects the prompt section (topics → ``create_topic``, components →
    ``create_component``, procedures → ``create_procedure``, commands → ``create_cmd``). Only
    types in ``GROUNDABLE_BLOCK_TYPES`` are web-grounded; commands never are.

    Returns ``None`` when no LLM is configured or generation fails, so callers can fall back
    (the REST endpoint to a simple template, the stream path to leaving the action unchanged).
    """
    settings = get_settings()
    if not settings.openai_api_key and not settings.anthropic_api_key:
        return None

    inputs = inputs or []
    outputs = outputs or []
    name = topic_name.replace(" ", "_")
    cat = category or "general"

    section = _BLOCK_TYPE_SECTION.get(block_type, "create_topic")
    system_prompt = (
        get_system_prompt(section)
        + "\n\nJSON FORMAT REFERENCE:\n"
        + get_json_schema()
    )

    user_message = f"Block name: {name}\nBlock type: {block_type}\nCategory: {cat}\n"
    if inputs:
        user_message += f"Requested input ports (besides 'block_description'): {', '.join(inputs)}\n"
    if outputs:
        user_message += f"Requested output ports: {', '.join(outputs)}\n"
    if description:
        user_message += f"\nDescription / content to extract data from:\n{description[:3000]}"

    # Ground in live web data when no rich content was provided — only for groundable types
    # (topics/components/procedures). Commands and any non-groundable type are never grounded,
    # even if the caller passes ``ground=True``.
    groundable = block_type in _GROUNDABLE_BLOCK_TYPES
    do_ground = groundable and (
        ground
        if ground is not None
        else len(description.strip()) < _GROUND_DESCRIPTION_MAX_CHARS
    )
    if do_ground:
        search_query = (description.strip() or name.replace("_", " ")).strip()
        if outputs:
            search_query += " " + " ".join(outputs)
        user_message = await _augment_with_grounding(user_message, search_query)

    result: dict = await call_llm_json(
        system_prompt, user_message, model=model, temperature=0.2,
    )

    # Ensure block_id is filled
    if result.get("tool_calls"):
        params = result["tool_calls"][0].get("params", {})
        if not params.get("block_id"):
            params["block_id"] = uuid.uuid4().hex[:8]
            result["tool_calls"][0]["params"] = params

    return result


@router.post("/generate/topic")
async def generate_topic_block(
    body: TopicGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Generate a structured topic block using AI (OpenAI) or a simple template fallback.

    When the request carries only a short query (no pre-fetched page content), live web
    results are fetched and injected so the extracted entities are real, current, and
    source-cited instead of recalled from stale training data.
    """
    settings = get_settings()

    if not settings.openai_api_key and not settings.anthropic_api_key:
        log.info("blocks_generate_topic_fallback", reason="no_llm_key", topic=body.topic_name)
        return _simple_topic_response(body)

    try:
        result = await generate_topic_payload(
            topic_name=body.topic_name,
            category=body.category,
            description=body.description,
            inputs=body.inputs,
            outputs=body.outputs,
            ground=body.ground,
            model=body.run_llm_model,
        )
        if result is None:
            return _simple_topic_response(body)
        log.info("blocks_generate_topic_ok", topic=body.topic_name)
        return result

    except Exception as exc:
        log.error("blocks_generate_topic_error", topic=body.topic_name, error=str(exc))
        # Graceful fallback to simple template
        return _simple_topic_response(body)


async def _call_openai_json(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Shared helper: provider-routed JSON completion, returns a parsed dict.

    Routes by the selected ``model`` id (claude-* → Anthropic, gpt-*/o3 → OpenAI);
    falls back to the OpenAI default when ``model`` is empty/unknown.
    """
    return await call_llm_json(
        system_prompt, user_message, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


async def _call_openai_text(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Shared helper: provider-routed text completion, returns raw text."""
    return await call_llm_text(
        system_prompt, user_message, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )


def build_tool_codegen_message(
    *,
    tool_name: str,
    description: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    existing_code: str = "",
) -> str:
    """Assemble the user message the [create_tool] prompt expects.

    Supplies the tool name, input/output port names, and the requirement text so the
    generated file's @register_tool decorator and I/O scaffold reference the real ports.
    """
    inputs = [p for p in (inputs or []) if p]
    outputs = [p for p in (outputs or []) if p]
    parts = [
        f"Tool name: {tool_name}",
        f"Input ports: {', '.join(inputs) if inputs else '(none)'}",
        f"Output ports: {', '.join(outputs) if outputs else '(none)'}",
        f"Tool Requirement:\n{description.strip()}",
    ]
    if existing_code.strip():
        parts.append(
            "Existing code (improve it; keep behaviour unless the requirement changed):\n"
            f"{existing_code.strip()}"
        )
    return "\n\n".join(parts)


# Small TTL cache for pure code generation. Identical (model, requirement) requests —
# e.g. recreating a just-deleted tool, or regenerating — skip the LLM round-trip. Only
# generate_tool_code uses it: its output is a deterministic-ish source string with no
# embedded ids/timestamps. Topic generation is excluded (time-sensitive, web-grounded)
# and code-payload generation is excluded (its envelope carries a fresh block_id).
_GEN_CACHE: dict[str, tuple[float, str]] = {}
_GEN_CACHE_TTL = 3600.0  # seconds
_GEN_CACHE_MAX = 256


def _gen_cache_get(key: str) -> str | None:
    entry = _GEN_CACHE.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _GEN_CACHE_TTL:
        _GEN_CACHE.pop(key, None)
        return None
    return val


def _gen_cache_put(key: str, val: str) -> None:
    if len(_GEN_CACHE) >= _GEN_CACHE_MAX and key not in _GEN_CACHE:
        oldest = min(_GEN_CACHE, key=lambda k: _GEN_CACHE[k][0])
        _GEN_CACHE.pop(oldest, None)
    _GEN_CACHE[key] = (time.monotonic(), val)


async def generate_tool_code(user_message: str, *, temperature: float = 0.2, model: str | None = None) -> str:
    """Generate a complete @register_tool-formatted Python file from a tool requirement.

    Uses the shared [create_tool] Msg_config section — the same structural contract the
    manual UnifiedWindow path produces — so voice/text tools fit the MCP server. Returns
    raw Python source (markdown fences stripped). Identical requests are served from a
    short-lived in-process cache to avoid re-paying the (slow) codegen round-trip.
    """
    system_prompt = get_system_prompt(_BLOCK_TYPE_SECTION["tools"])
    if not system_prompt:
        raise ValueError("create_tool prompt section missing from Msg_config")
    cache_key = hashlib.sha256(
        f"{model or ''}\x00{user_message}".encode()
    ).hexdigest()
    cached = _gen_cache_get(cache_key)
    if cached is not None:
        log.info("blocks_tool_codegen_cache_hit")
        return cached
    raw = await _call_openai_text(system_prompt, user_message, temperature=temperature, model=model, max_tokens=8192)
    code = _strip_code_fences(raw)
    if code.strip():
        _gen_cache_put(cache_key, code)
    return code


def _code_port_path(category: str, name: str, port_type: str, port_name: str) -> str:
    return f"data/code/{category}/{name}/{port_type}/{port_name}.txt"


# Common language spellings that mean the same thing, normalized to one canonical token
# so a returned/requested-language comparison doesn't flag js vs javascript as a mismatch.
_LANG_ALIASES = {
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "py": "python",
    "python": "python",
    "c++": "cpp",
    "cpp": "cpp",
    "c#": "csharp",
    "csharp": "csharp",
    "cs": "csharp",
    "golang": "go",
    "go": "go",
    "rs": "rust",
    "rust": "rust",
    "sh": "bash",
    "shell": "bash",
    "bash": "bash",
    # Hardware description languages — the code block is how a user gets AI-written
    # RTL onto the canvas to feed the verilator / yosys / openroad blocks, and
    # "sv" / "system verilog" are what people actually type.
    "verilog": "verilog",
    "v": "verilog",
    "sv": "systemverilog",
    "systemverilog": "systemverilog",
    "system verilog": "systemverilog",
    "vhdl": "vhdl",
}


def _normalize_lang(lang: str) -> str:
    """Canonicalize a language name for alias-tolerant comparison (js == javascript)."""
    key = (lang or "").strip().lower()
    return _LANG_ALIASES.get(key, key)


# The languages whose module interface can be checked and frozen across a repair.
_HDL_LANGS = frozenset({"verilog", "systemverilog", "vhdl"})

# One repair round when the fixer breaks the interface it was told to preserve.
# One, not many: a second violation means the model has lost the thread, and the
# user is better served by seeing the design plus the complaint than by another
# minute of retries.
_CODE_FIX_REPAIR_ROUNDS = 1


def code_prompt_section(*, feedback: str, previous_code: str, language: str) -> str:
    """
    Which Msg_config section a code request should use: repair, or write fresh.

    [fix_rtl] needs BOTH a design to repair and a report saying what is wrong with
    it — feedback alone has nothing to edit, and a previous design alone has
    nothing to fix. Either on its own would quietly degrade into a regeneration
    that is free to change the port list, which is exactly what the verification
    loop cannot survive.

    HDL only. The frozen-interface rule that makes [fix_rtl] safe is meaningless
    for Python, so a code block with feedback in any other language stays on
    [create_code], which already treats feedback as extra requirement text.
    """
    if not (feedback or "").strip() or not (previous_code or "").strip():
        return _BLOCK_TYPE_SECTION["code"]
    if _normalize_lang(language) not in _HDL_LANGS:
        return _BLOCK_TYPE_SECTION["code"]
    return _CODE_FIX_RTL_SECTION


def build_code_fix_message(
    *,
    block_name: str,
    language: str,
    top: str,
    ports: list[str],
    previous_code: str,
    feedback: str,
    description: str = "",
    problems: list[str] | None = None,
) -> str:
    """Assemble the user message the [fix_rtl] prompt expects.

    The INTERFACE line is the highest-value line in the prompt: it turns "keep the
    module header identical" from an aspiration into something the model can copy
    and the server can check afterwards with :func:`hdl.validate_rtl_fix`. It is
    derived here from the previous design, never asked of the caller.
    """
    iface = ", ".join(ports) if ports else "(could not be parsed — copy the existing header verbatim)"
    parts = [
        f"Block name: {block_name}",
        f"Programming language: {language or 'verilog'}",
        f"Module interface that MUST NOT change (top module: {top or '(unknown)'}): {iface}",
    ]
    if (description or "").strip():
        parts.append(f"What this design is meant to do:\n{description.strip()}")
    parts.append(
        "Current code (this is what failed; return a corrected version of THIS "
        f"design):\n{previous_code.strip()}"
    )
    parts.append(
        "Failing tests reported by the simulator (fix the DESIGN; the tests are "
        f"correct and must not change):\n{feedback.strip()}"
    )
    if problems:
        parts.append(
            "Your previous attempt was REJECTED for these reasons; return a "
            "corrected design:\n- " + "\n- ".join(problems)
        )
    return "\n\n".join(parts)


def build_code_gen_message(
    *,
    block_name: str,
    description: str,
    language: str,
    outputs: list[str] | None = None,
) -> str:
    """Assemble the user message the [create_code] prompt expects.

    Supplies the block name, the target programming LANGUAGE, the requested output ports,
    and the requirement text so the model generates a complete program in that language.
    """
    outputs = [p for p in (outputs or []) if p]
    parts = [
        f"Block name: {block_name}",
        f"Programming language: {language or 'python'}",
        f"Output ports: {', '.join(outputs) if outputs else '(none)'}",
        f"Requirement (what the code should do):\n{description.strip()}",
    ]
    return "\n\n".join(parts)


async def generate_code_payload(
    *,
    block_name: str,
    category: str = "general",
    description: str = "",
    language: str = "python",
    feedback: str = "",
    previous_code: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict | None:
    """Generate a code block envelope (the `tool_calls` dict) via AI.

    Mirrors :func:`generate_topic_payload` but produces source code in the requested
    ``language`` instead of researched entities: it calls OpenAI with the [create_code]
    prompt in JSON mode and returns a ``{tool_calls, ...}`` envelope whose output ports
    (``code``/``explanation``/``improvements``/``dependencies``/``language``) hold the
    generated source plus its explanation, improvement ideas, and dependency list. No web
    grounding — code generation must not be polluted with live search results.

    With ``feedback`` and ``previous_code`` on an HDL language this becomes a REPAIR
    instead: the [fix_rtl] prompt is given the failing-test report and the design that
    produced it, and is required to return the same module interface. That is what
    closes the verification loop — a verilator block's ``failures`` output feeds this
    port, and the repaired design goes back round. The returned interface is checked
    (:func:`hdl.validate_rtl_fix`) with one repair round, because a fix that renames a
    port silently breaks the very tests it was meant to satisfy.

    Returns ``None`` when OpenAI is not configured or generation fails, so callers can fall
    back (the REST endpoint to an empty-port stub, the stream path to leaving the action as-is).
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    name = block_name.replace(" ", "_")
    cat = category or "general"
    lang = (language or "python").strip()

    section = code_prompt_section(
        feedback=feedback, previous_code=previous_code, language=lang)
    system_prompt = get_system_prompt(section)
    if not system_prompt:
        raise ValueError(f"{section} prompt section missing from Msg_config")

    fixing = section == _CODE_FIX_RTL_SECTION
    problems: list[str] = []
    result: dict = {}
    code = ""
    if fixing:
        top_name = infer_top_module(previous_code)
        ports = module_ports(previous_code, top_name)
        draft = previous_code
        for attempt in range(_CODE_FIX_REPAIR_ROUNDS + 1):
            user_message = build_code_fix_message(
                block_name=name, language=lang, top=top_name, ports=ports,
                previous_code=draft, feedback=feedback, description=description,
                problems=problems or None,
            )
            result = await _call_openai_json(
                system_prompt, user_message, temperature=0.2, max_tokens=8192)
            code = _strip_code_fences(str(result.get("code", "")))
            problems = validate_rtl_fix(code, previous_code, top_name)
            if not problems:
                break
            log.warning("rtl_fix_rejected", block=name, attempt=attempt, problems=problems)
    else:
        user_message = build_code_gen_message(
            block_name=name,
            description=description,
            language=lang,
            outputs=outputs,
        )
        result = await _call_openai_json(system_prompt, user_message, temperature=0.2)
        code = _strip_code_fences(str(result.get("code", "")))

    # Validate the returned language against what was requested. The model occasionally
    # echoes a different language than asked; trust the requested one (the code is what
    # the user wired up around). Alias-aware so js/javascript etc. aren't false positives.
    returned_lang = str(result.get("language", "")).strip()
    if returned_lang and _normalize_lang(returned_lang) != _normalize_lang(lang):
        log.warning(
            "code_language_mismatch", block=name, requested=lang, returned=returned_lang,
        )
    out_lang = lang  # always surface the requested language on the port

    improvements = str(result.get("improvements", ""))
    if problems:
        # The design is still handed back: a rejected repair the user can read and
        # correct beats an empty port, and the loop's own interface check stops it
        # from being propagated as though it were clean.
        improvements = ("Validation: " + "; ".join(problems)
                        + ("\n" + improvements if improvements else ""))

    op = [
        {
            "port_name": "code",
            "port_content": code,
            "port_path": _code_port_path(cat, name, "outputs", "code"),
        },
        {
            "port_name": "explanation",
            "port_content": str(result.get("explanation", "")),
            "port_path": _code_port_path(cat, name, "outputs", "explanation"),
        },
        {
            "port_name": "improvements",
            "port_content": improvements,
            "port_path": _code_port_path(cat, name, "outputs", "improvements"),
        },
        {
            "port_name": "dependencies",
            "port_content": str(result.get("dependencies", "")),
            "port_path": _code_port_path(cat, name, "outputs", "dependencies"),
        },
        {
            "port_name": "language",
            "port_content": out_lang,
            "port_path": _code_port_path(cat, name, "outputs", "language"),
        },
    ]
    ip = [
        {
            "port_name": "block_description",
            "port_content": description,
            "port_path": _code_port_path(cat, name, "inputs", "block_description"),
        },
        {
            "port_name": "language",
            "port_content": lang,
            "port_path": _code_port_path(cat, name, "inputs", "language"),
        },
        # The failing-test report a verilator block writes here to ask for a
        # repair. Present on every code block so the wire can be drawn before
        # there is anything to say.
        {
            "port_name": "feedback",
            "port_content": feedback,
            "port_path": _code_port_path(cat, name, "inputs", "feedback"),
        },
    ]
    for inp in inputs or []:
        if inp and inp not in ("description", "block_description", "language",
                               "feedback"):
            ip.append({
                "port_name": inp,
                "port_content": "",
                "port_path": _code_port_path(cat, name, "inputs", inp),
            })

    return {
        "tool_calls": [
            {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "block_id": uuid.uuid4().hex[:8],
                    "block_type": "code",
                    "x": 0,
                    "y": 0,
                    "input_ports": ip,
                    "output_ports": op,
                },
            }
        ],
        "connections": [],
    }


def _simple_code_response(body: CodeGenerateRequest) -> dict:
    """Fallback: build a minimal code block without AI when OpenAI is not configured.

    On a REPAIR request the existing design is echoed back on the ``code`` port
    rather than blanked. This is not cosmetic: the app writes these ports over the
    block's own, so returning an empty ``code`` here would wipe a working design
    every time a key was missing or a generation failed — the loop would destroy
    the thing it exists to improve.
    """
    name = body.block_name.replace(" ", "_")
    cat = body.category or "general"
    lang = (body.language or "python").strip()
    kept_code = (body.previous_code or "").strip()
    note = ("The design was returned unchanged: code generation is not "
            "configured, so the reported failures could not be acted on."
            if kept_code else "")
    return {
        "tool_calls": [
            {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "block_id": uuid.uuid4().hex[:8],
                    "block_type": "code",
                    "x": 0,
                    "y": 0,
                    "input_ports": [
                        {"port_name": "block_description", "port_content": body.description,
                         "port_path": _code_port_path(cat, name, "inputs", "block_description")},
                        {"port_name": "language", "port_content": lang,
                         "port_path": _code_port_path(cat, name, "inputs", "language")},
                        {"port_name": "feedback", "port_content": body.feedback,
                         "port_path": _code_port_path(cat, name, "inputs", "feedback")},
                    ],
                    "output_ports": [
                        {"port_name": pn,
                         "port_content": (
                             lang if pn == "language"
                             else kept_code if pn == "code"
                             else note if pn == "improvements"
                             else ""),
                         "port_path": _code_port_path(cat, name, "outputs", pn)}
                        for pn in ("code", "explanation", "improvements", "dependencies", "language")
                    ],
                },
            }
        ],
        "connections": [],
    }


@router.post("/generate/code")
async def generate_code_block(
    body: CodeGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Generate a code block (source code in the chosen language) using AI, with fallback.

    Serves both the manual UnifiedWindow creation path and the app's Run/Regenerate buttons.
    Falls back to an empty-port stub when OpenAI is unconfigured or generation fails.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        log.info("blocks_generate_code_fallback", reason="no_openai_key", block=body.block_name)
        return _simple_code_response(body)
    try:
        result = await generate_code_payload(
            block_name=body.block_name,
            category=body.category,
            description=body.description,
            language=body.language,
            feedback=body.feedback,
            previous_code=body.previous_code,
            inputs=body.inputs,
            outputs=body.outputs,
        )
        if result is None:
            return _simple_code_response(body)
        log.info("blocks_generate_code_ok", block=body.block_name, language=body.language,
                 fixing=bool(body.feedback.strip() and body.previous_code.strip()))
        return result
    except Exception as exc:
        log.error("blocks_generate_code_error", block=body.block_name, error=str(exc))
        return _simple_code_response(body)


# ── Testbench block (AI-generated cocotb tests derived from the spec) ────────────

# Frameworks the [create_testbench] prompt knows how to write. "sv" is reserved for
# a SystemVerilog/UVM-lite harness; only cocotb is wired through verilator today.
_TESTBENCH_FRAMEWORKS = ("cocotb",)

# One repair round when the first draft fails validation (invented signals, no
# @cocotb.test, syntax error). One, not many: a second failure means the spec or
# interface is the problem and the user should see it on the improvements port.
_TESTBENCH_REPAIR_ROUNDS = 1


def build_testbench_gen_message(
    *,
    block_name: str,
    top: str,
    ports: list[str],
    spec: str,
    framework: str = "cocotb",
    style: str = "directed+random",
    coverage_goals: str = "",
    extra_tests: str = "",
    feedback: str = "",
    previous_testbench: str = "",
    problems: list[str] | None = None,
) -> str:
    """Assemble the user message the [create_testbench] prompt expects.

    The INTERFACE is given as a bare port list — deliberately not the RTL body — so
    the model cannot derive expectations from the implementation. An unknown
    interface (no RTL yet) is stated as such so the model infers ports from the spec
    and the user re-runs once the code block is wired in.
    """
    iface = (
        ", ".join(ports) if ports
        else "(unknown — no RTL wired yet; infer conventional port names from the spec "
             "and list them in the explanation)"
    )
    parts = [
        f"Block name: {block_name}",
        f"Top module: {top or '(infer from spec)'}",
        f"Interface (the ONLY signals you may reference as dut.<name>): {iface}",
        f"Framework: {framework or 'cocotb'}",
        f"Test style: {style or 'directed+random'}",
        f"Spec (the contract every expected value must come from):\n{spec.strip()}",
    ]
    if coverage_goals.strip():
        parts.append(f"Coverage goals:\n{coverage_goals.strip()}")
    if extra_tests.strip():
        parts.append(f"Extra tests requested:\n{extra_tests.strip()}")
    if feedback.strip():
        parts.append(
            "Reviewer feedback on the previous testbench (fix the TESTS accordingly):\n"
            f"{feedback.strip()}"
        )
    if previous_testbench.strip():
        parts.append(f"Previous testbench:\n{previous_testbench.strip()}")
    if problems:
        parts.append(
            "The previous draft was REJECTED for these reasons; return a corrected testbench:\n- "
            + "\n- ".join(problems)
        )
    return "\n\n".join(parts)


def _generated_envelope(
    *,
    block_type: str,
    name: str,
    category: str,
    description: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    extra_inputs: list[str] | None = None,
    extra_outputs: list[str] | None = None,
) -> dict:
    """The ``{tool_calls, connections}`` envelope for an AI-generated, scaffolded block.

    Port ORDER and SET come from ``_SCAFFOLD_SPECS[block_type]`` so the generated
    block is port-identical to the scaffold-only fallback and to the Qt dialog.
    Shared by every block type whose content is written by the model but whose
    port list is fixed (testbench, code_hdl) — as opposed to the code block, which
    predates the scaffold specs and spells its ports out inline.
    """
    spec = _SCAFFOLD_SPECS[block_type]

    def port_path(port_type: str, port_name: str) -> str:
        return _scaffold_port_path(
            block_type, spec.category_based, category, name, port_type, port_name
        )

    ip = [{
        "port_name": "block_description",
        "port_content": description,
        "port_path": port_path("inputs", "block_description"),
    }]
    seen = {"block_description", "description"}
    for pn in list(spec.inputs) + list(extra_inputs or []):
        if pn and pn not in seen:
            seen.add(pn)
            ip.append({
                "port_name": pn,
                "port_content": inputs.get(pn) or spec.defaults.get(pn, ""),
                "port_path": port_path("inputs", pn),
            })
    op = []
    seen_out: set[str] = set()
    for pn in list(spec.outputs) + list(extra_outputs or []):
        if pn and pn not in seen_out:
            seen_out.add(pn)
            op.append({
                "port_name": pn,
                "port_content": outputs.get(pn, ""),
                "port_path": port_path("outputs", pn),
            })
    return {
        "tool_calls": [{
            "id": 1,
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "block_id": uuid.uuid4().hex[:8],
                "block_type": block_type,
                "x": 0,
                "y": 0,
                "input_ports": ip,
                "output_ports": op,
            },
        }],
        "connections": [],
    }


def _test_plan_text(raw: object) -> str:
    """Normalise the model's test_plan (list or JSON string) to a JSON string."""
    if isinstance(raw, (list, dict)):
        return json.dumps(raw, ensure_ascii=False)
    return str(raw or "")


async def generate_testbench_payload(
    *,
    block_name: str,
    category: str = "general",
    description: str = "",
    spec: str = "",
    rtl: str = "",
    top: str = "",
    framework: str = "cocotb",
    style: str = "directed+random",
    coverage_goals: str = "",
    extra_tests: str = "",
    feedback: str = "",
    previous_testbench: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    model: str | None = None,
) -> dict | None:
    """Generate a testbench block envelope via AI, validated against the RTL interface.

    Flow: extract the module's port list from ``rtl`` (never the body) → ask the
    [create_testbench] prompt for spec-derived cocotb tests → validate (Python parses,
    has a @cocotb.test, references only real ports) → one repair round on failure →
    surface any residual problems on ``improvements`` and mark ``status``.

    Returns ``None`` when no LLM key is configured so callers fall back to a scaffold.
    """
    settings = get_settings()
    if not settings.openai_api_key and not getattr(settings, "anthropic_api_key", ""):
        return None

    name = block_name.replace(" ", "_")
    cat = category or "general"
    spec_text = (spec or "").strip() or (description or "").strip()
    if not spec_text:
        raise ValueError("a spec (or description) is required to generate a testbench")
    fw = (framework or "cocotb").strip().lower()
    if fw not in _TESTBENCH_FRAMEWORKS:
        log.warning("testbench_framework_unsupported", block=name, framework=fw)
        fw = "cocotb"

    top_name = (top or "").strip() or (infer_top_module(rtl) if rtl.strip() else "")
    ports = module_ports(rtl, top_name) if rtl.strip() and top_name else []

    system_prompt = get_system_prompt(_BLOCK_TYPE_SECTION["testbench"])
    if not system_prompt:
        raise ValueError("create_testbench prompt section missing from Msg_config")

    problems: list[str] = []
    result: dict = {}
    testbench = ""
    draft = previous_testbench
    for attempt in range(_TESTBENCH_REPAIR_ROUNDS + 1):
        user_message = build_testbench_gen_message(
            block_name=name, top=top_name, ports=ports, spec=spec_text, framework=fw,
            style=style, coverage_goals=coverage_goals, extra_tests=extra_tests,
            feedback=feedback, previous_testbench=draft, problems=problems or None,
        )
        result = await _call_openai_json(
            system_prompt, user_message, temperature=0.2, model=model, max_tokens=8192,
        )
        testbench = _strip_code_fences(str(result.get("testbench", "")))
        problems = validate_testbench(testbench, rtl, top_name)
        if not problems:
            break
        log.warning(
            "testbench_validation_failed", block=name, attempt=attempt, problems=problems,
        )
        draft = testbench

    improvements = str(result.get("improvements", "")).strip()
    if problems:
        improvements = (
            "Validation: " + "; ".join(problems)
            + ("\n" + improvements if improvements else "")
        )
    status = "ok" if not problems else "needs_review"
    out_top = str(result.get("top", "")).strip() or top_name

    return _generated_envelope(
        block_type="testbench",
        name=name,
        category=cat,
        description=description,
        inputs={
            "spec": spec_text,
            "rtl": rtl,
            "top": top_name,
            "framework": fw,
            "style": style or "directed+random",
            "coverage_goals": coverage_goals,
            "extra_tests": extra_tests,
            "feedback": feedback,
        },
        outputs={
            "testbench": testbench,
            "sva": _strip_code_fences(str(result.get("sva", ""))),
            "test_plan": _test_plan_text(result.get("test_plan", "")),
            "top": out_top,
            "explanation": str(result.get("explanation", "")),
            "improvements": improvements,
            "status": status,
            "errors": "",
        },
        extra_inputs=inputs,
        extra_outputs=outputs,
    )


def _simple_testbench_response(body: TestbenchGenerateRequest, error: str = "") -> dict:
    """Fallback: a port-complete testbench block with no generated tests."""
    name = body.block_name.replace(" ", "_")
    spec_text = (body.spec or "").strip()
    return _generated_envelope(
        block_type="testbench",
        name=name,
        category=body.category or "general",
        description=spec_text,
        inputs={
            "spec": spec_text, "rtl": body.rtl, "top": body.top,
            "framework": body.framework or "cocotb", "style": body.style or "directed+random",
            "coverage_goals": body.coverage_goals, "extra_tests": body.extra_tests,
            "feedback": body.feedback,
        },
        outputs={"top": body.top, "status": "error" if error else "", "errors": error},
        extra_inputs=body.inputs,
        extra_outputs=body.outputs,
    )


@router.post("/generate/testbench")
async def generate_testbench_block(
    body: TestbenchGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Generate a testbench block (spec-derived cocotb tests) using AI, with fallback.

    Serves the manual UnifiedWindow creation path and the app's Run/Regenerate
    buttons on a testbench block. Without a key, or on failure, returns a
    port-complete stub whose ``errors``/``status`` ports say why.
    """
    settings = get_settings()
    if not settings.openai_api_key and not getattr(settings, "anthropic_api_key", ""):
        log.info("blocks_generate_testbench_fallback", reason="no_llm_key", block=body.block_name)
        return _simple_testbench_response(body, error="AI not configured")
    try:
        result = await generate_testbench_payload(
            block_name=body.block_name,
            category=body.category,
            description=body.spec,
            spec=body.spec,
            rtl=body.rtl,
            top=body.top,
            framework=body.framework,
            style=body.style,
            coverage_goals=body.coverage_goals,
            extra_tests=body.extra_tests,
            feedback=body.feedback,
            inputs=body.inputs,
            outputs=body.outputs,
        )
        if result is None:
            return _simple_testbench_response(body, error="AI not configured")
        log.info("blocks_generate_testbench_ok", block=body.block_name, top=body.top)
        return result
    except Exception as exc:
        log.error("blocks_generate_testbench_error", block=body.block_name, error=str(exc))
        return _simple_testbench_response(body, error=str(exc))


# ── spec_hdl block (AI-written specification: the contract the loop turns on) ──

# One repair round, for the same reason code_hdl takes one: a second failure
# means the EXPLANATION is the problem, and the user is better served by seeing
# that on the improvements port than by paying for another round of it.
_SPEC_HDL_REPAIR_ROUNDS = 1

# The design parameters a hardware spec always needs and prose always leaves
# implicit. Each is a port on the block; this table is what turns a filled port
# into a line the prompt can honour, and keeps the wording identical between the
# fresh and the revision flow.
_SPEC_PARAM_LABELS: tuple[tuple[str, str], ...] = (
    ("data_width", "Data width"),
    ("addr_width", "Address width"),
    ("parameters", "Further parameters"),
    ("logic_style", "Logic style (combinational / sequential)"),
    ("reset_style", "Reset style"),
    ("clocking", "Clocking scheme"),
    ("protocol", "Interface protocol"),
    ("throughput", "Throughput / latency requirement"),
)

# Values that mean "you decide". A port carrying its own default is not an
# instruction, and forwarding it as one would have the model dutifully specify a
# synchronous active-high reset for a purely combinational block.
_SPEC_PARAM_UNSET = frozenset({"auto", "any", "n/a", "na", "none", "-", "unspecified"})


def spec_hdl_parameters(values: dict[str, str]) -> list[str]:
    """The filled design parameters as ``"Label: value"`` lines, in port order."""
    lines = []
    for port, label in _SPEC_PARAM_LABELS:
        val = str(values.get(port, "") or "").strip()
        if val and val.lower() not in _SPEC_PARAM_UNSET:
            lines.append(f"{label}: {val}")
    return lines


def build_spec_hdl_gen_message(
    *,
    block_name: str,
    explanation: str,
    language: str,
    top: str,
    parameters: list[str] | None = None,
    constraints: str = "",
    design: str = "",
    previous_spec: str = "",
    feedback: str = "",
    problems: list[str] | None = None,
) -> str:
    """Assemble the user message the [create_spec_hdl] prompt expects.

    One builder for both flows, unlike code_hdl's pair: a revision is the same
    request with the current spec and the feedback appended, because the output
    contract does not change - the model still returns a whole specification,
    never a diff. There is no separate [fix_spec] section for the same reason.
    """
    parts = [
        f"Block name: {block_name}",
        f"Target hardware description language: {language or 'systemverilog'}",
        f"Top module: {top or block_name}",
        f"Explanation (the rough description to specify):\n{explanation.strip()}",
    ]
    if parameters:
        parts.append("Design parameters (the user pinned these; honour them exactly):\n- "
                     + "\n- ".join(parameters))
    if constraints.strip():
        parts.append(
            "Constraints (target technology, conventions, anything the explanation "
            f"does not cover):\n{constraints.strip()}"
        )
    if design.strip():
        parts.append(
            "Existing design (specify what the module SHOULD do; where this code "
            "disagrees, say so in improvements rather than ratifying it):\n"
            f"{design.strip()}"
        )
    if previous_spec.strip():
        parts.append(
            "Current specification (revise it; keep the requirement numbering "
            f"stable):\n{previous_spec.strip()}"
        )
    if feedback.strip():
        parts.append(
            "Feedback saying the current specification is wrong, incomplete or "
            f"ambiguous (revise the clauses it implicates, keep the rest):\n{feedback.strip()}"
        )
    if problems:
        parts.append(
            "Your previous draft was REJECTED for these reasons; return a corrected "
            "specification:\n- " + "\n- ".join(problems)
        )
    return "\n\n".join(parts)


async def generate_spec_hdl_payload(
    *,
    block_name: str,
    category: str = "general",
    description: str = "",
    explanation: str = "",
    previous_code: str = "",
    previous_spec: str = "",
    language: str = "systemverilog",
    top: str = "",
    data_width: str = "",
    addr_width: str = "",
    parameters: str = "",
    logic_style: str = "",
    reset_style: str = "",
    clocking: str = "",
    protocol: str = "",
    throughput: str = "",
    constraints: str = "",
    feedback: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    model: str | None = None,
) -> dict | None:
    """Generate (or revise) a spec_hdl block envelope via AI.

    [create_spec_hdl] writes the contract from the explanation and the pinned
    design parameters -> :func:`validate_spec` checks it is traceable (enumerated
    requirements), machine-readable (the interface parses) and self-consistent
    (the signals discussed exist) -> one repair round on failure -> residual
    problems and every warning land on ``improvements`` with
    ``status=needs_review``.

    ``feedback`` does not switch prompts the way it does on a code block. The
    output contract is identical either way - a whole specification, never a
    diff - so a revision is the same section with the current spec and the
    failure appended, and there is no [fix_spec] to keep in step with it.

    ``top`` is taken from the model's answer only when the caller did not pin
    one, so a canvas wired to a module name cannot have it renamed underneath it.

    Returns ``None`` when no LLM key is configured so callers fall back to a scaffold.
    """
    settings = get_settings()
    if not settings.openai_api_key and not getattr(settings, "anthropic_api_key", ""):
        return None

    name = block_name.replace(" ", "_")
    cat = category or "general"
    lang = _normalize_lang(language) or "systemverilog"
    if lang not in _HDL_LANGS:
        raise ValueError(
            f"'{language}' is not a hardware description language; "
            f"spec_hdl accepts {', '.join(sorted(_HDL_LANGS))}"
        )

    text = (explanation or "").strip() or (description or "").strip()
    design = (previous_code or "").strip()
    if not text and not design:
        raise ValueError(
            "an explanation (or description, or an existing design) is required "
            "to write a specification"
        )
    # A spec written from RTL alone is a legitimate request - recovering the
    # contract of a design someone inherited - so the explanation is allowed to
    # be empty as long as there is something to specify.
    if not text:
        text = (
            "Recover the specification of the existing design below: state the "
            "behaviour its interface implies, as a contract a new implementation "
            "could be written against."
        )

    system_prompt = get_system_prompt(_BLOCK_TYPE_SECTION["spec_hdl"])
    if not system_prompt:
        raise ValueError("create_spec_hdl prompt section missing from Msg_config")

    pinned_top = (top or "").strip()
    param_lines = spec_hdl_parameters({
        "data_width": data_width, "addr_width": addr_width, "parameters": parameters,
        "logic_style": logic_style, "reset_style": reset_style, "clocking": clocking,
        "protocol": protocol, "throughput": throughput,
    })

    problems: list[str] = []
    warnings: list[str] = []
    result: dict = {}
    draft_spec = (previous_spec or "").strip()
    for attempt in range(_SPEC_HDL_REPAIR_ROUNDS + 1):
        user_message = build_spec_hdl_gen_message(
            block_name=name, explanation=text, language=lang,
            top=pinned_top or name, parameters=param_lines, constraints=constraints,
            design=design, previous_spec=draft_spec, feedback=feedback,
            problems=problems or None,
        )
        result = await _call_openai_json(
            system_prompt, user_message, temperature=0.2, model=model, max_tokens=8192,
        )
        spec_text = _strip_code_fences(str(result.get("spec", "")))
        problems, warnings = validate_spec(
            spec_text,
            str(result.get("requirements", "")),
            str(result.get("interface", "")),
            signals_analysis=str(result.get("signals_analysis", "")),
            design=design,
            top=pinned_top,
        )
        if not problems:
            break
        log.warning(
            "spec_hdl_validation_failed", block=name, attempt=attempt, problems=problems,
        )
        draft_spec = spec_text or draft_spec

    spec_text = _strip_code_fences(str(result.get("spec", "")))
    # A revision that produced nothing must not blank the contract it was meant
    # to improve: the app overwrites every port it is handed, and losing the spec
    # loses the design and the tests with it.
    if not spec_text.strip() and (previous_spec or "").strip():
        spec_text = previous_spec.strip()

    notes = list(problems) + list(warnings)
    improvements = str(result.get("improvements", "")).strip()
    if notes:
        improvements = (
            "Validation: " + "; ".join(notes) + ("\n" + improvements if improvements else "")
        )

    out_top = pinned_top or str(result.get("top", "")).strip() or name

    return _generated_envelope(
        block_type="spec_hdl",
        name=name,
        category=cat,
        description=description,
        inputs={
            "explanation": text,
            "previous_code": design,
            "top": pinned_top,
            "language": lang,
            "data_width": data_width,
            "addr_width": addr_width,
            "parameters": parameters,
            "logic_style": logic_style,
            "reset_style": reset_style,
            "clocking": clocking,
            "protocol": protocol,
            "throughput": throughput,
            "constraints": constraints,
            "feedback": feedback,
        },
        outputs={
            "spec": spec_text,
            "top": out_top,
            "interface": str(result.get("interface", "")),
            "signals_analysis": str(result.get("signals_analysis", "")),
            "parameters": str(result.get("parameters", "")),
            "timing": str(result.get("timing", "")),
            "requirements": str(result.get("requirements", "")),
            "assumptions": str(result.get("assumptions", "")),
            "explanation": str(result.get("explanation", "")),
            "improvements": improvements,
            "status": "needs_review" if notes else "ok",
            "errors": "",
        },
        extra_inputs=inputs,
        extra_outputs=outputs,
    )


def _simple_spec_hdl_response(body: SpecHdlGenerateRequest, error: str = "") -> dict:
    """Fallback: a port-complete spec_hdl block with no generated specification.

    A revision keeps the spec it was given, for the reason ``_simple_code_hdl_response``
    keeps the design: the app writes every port it is handed, so an empty ``spec``
    here would destroy the contract the design and the tests were both written
    from - a far worse outcome than a run that did nothing.
    """
    name = body.block_name.replace(" ", "_")
    lang = _normalize_lang(body.language) or "systemverilog"
    kept = (body.previous_spec or "").strip()
    note = (
        "The specification was returned unchanged: AI generation is not configured, "
        "so the feedback could not be acted on."
        if kept else ""
    )
    return _generated_envelope(
        block_type="spec_hdl",
        name=name,
        category=body.category or "general",
        description=(body.description or body.explanation or "").strip(),
        inputs={
            "explanation": (body.explanation or "").strip(),
            "previous_code": (body.previous_code or "").strip(),
            "top": body.top,
            "language": lang,
            "data_width": body.data_width,
            "addr_width": body.addr_width,
            "parameters": body.parameters,
            "logic_style": body.logic_style,
            "reset_style": body.reset_style,
            "clocking": body.clocking,
            "protocol": body.protocol,
            "throughput": body.throughput,
            "constraints": body.constraints,
            "feedback": body.feedback,
        },
        outputs={
            "spec": kept,
            "top": (body.top or "").strip() or name,
            "improvements": note,
            "status": "error" if error else "",
            "errors": error,
        },
        extra_inputs=body.inputs,
        extra_outputs=body.outputs,
    )


@router.post("/generate/spec_hdl")
async def generate_spec_hdl_block(
    body: SpecHdlGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Generate or revise a spec_hdl block (the loop's specification) using AI.

    Serves the manual UnifiedWindow creation path and the app's Run/Regenerate
    buttons. Without a key, or on failure, returns a port-complete stub whose
    ``errors``/``status`` ports say why - and which never blanks an existing
    specification.
    """
    settings = get_settings()
    if not settings.openai_api_key and not getattr(settings, "anthropic_api_key", ""):
        log.info("blocks_generate_spec_hdl_fallback", reason="no_llm_key", block=body.block_name)
        return _simple_spec_hdl_response(body, error="AI not configured")
    try:
        result = await generate_spec_hdl_payload(
            block_name=body.block_name,
            category=body.category,
            description=body.description,
            explanation=body.explanation,
            previous_code=body.previous_code,
            previous_spec=body.previous_spec,
            language=body.language,
            top=body.top,
            data_width=body.data_width,
            addr_width=body.addr_width,
            parameters=body.parameters,
            logic_style=body.logic_style,
            reset_style=body.reset_style,
            clocking=body.clocking,
            protocol=body.protocol,
            throughput=body.throughput,
            constraints=body.constraints,
            feedback=body.feedback,
            inputs=body.inputs,
            outputs=body.outputs,
            model=body.run_llm_model or None,
        )
        if result is None:
            return _simple_spec_hdl_response(body, error="AI not configured")
        log.info(
            "blocks_generate_spec_hdl_ok",
            block=body.block_name, language=body.language,
            revising=bool(body.feedback and body.previous_spec),
        )
        return result
    except Exception as exc:
        log.error("blocks_generate_spec_hdl_error", block=body.block_name, error=str(exc))
        return _simple_spec_hdl_response(body, error=str(exc))


# ── code_hdl block (AI-generated RTL derived from the spec) ──────────────────

# One repair round when the first draft fails validation (truncated, wrong top,
# unparsable interface). One, not many: a second failure means the spec is the
# problem and the user should see it on the improvements port rather than pay for
# another round of the same mistake.
_CODE_HDL_REPAIR_ROUNDS = 1

# Said on the improvements port when the design is VHDL. Not a silent empty
# interface: Verilator is Verilog/SystemVerilog only, so a VHDL design cannot
# reach a simulator, which means `feedback` never arrives and the fix loop never
# runs. An empty `interface` with no explanation reads as a parser bug and sends
# the user hunting for something that is not broken.
_VHDL_LOOP_NOTE = (
    "VHDL designs cannot be simulated by the verilator block (it is Verilog and "
    "SystemVerilog only), so the testbench, simulation and fix loop are "
    "unavailable for this language and the interface port is left empty."
)


def code_hdl_prompt_section(*, feedback: str, previous_code: str) -> str:
    """[fix_rtl] when there is a design AND a failure to repair it against, else [create_code_hdl].

    Simpler than :func:`code_prompt_section` because a code_hdl block is always
    HDL — the language guard that stops a Python `code` block from reaching the
    RTL fixer has nothing to guard here.
    """
    if not (feedback or "").strip() or not (previous_code or "").strip():
        return _BLOCK_TYPE_SECTION["code_hdl"]
    return _CODE_FIX_RTL_SECTION


def build_code_hdl_gen_message(
    *,
    block_name: str,
    language: str,
    top: str,
    spec: str,
    constraints: str = "",
    problems: list[str] | None = None,
) -> str:
    """Assemble the user message the [create_code_hdl] prompt expects."""
    parts = [
        f"Block name: {block_name}",
        f"Hardware description language: {language or 'systemverilog'}",
        f"Top module: {top or block_name}",
        f"Spec (the contract the design must satisfy):\n{spec.strip()}",
    ]
    if constraints.strip():
        parts.append(
            "Constraints (reset style, interface conventions, target technology):\n"
            f"{constraints.strip()}"
        )
    if problems:
        parts.append(
            "Your previous draft was REJECTED for these reasons; return a corrected design:\n- "
            + "\n- ".join(problems)
        )
    return "\n\n".join(parts)


async def generate_code_hdl_payload(
    *,
    block_name: str,
    category: str = "general",
    description: str = "",
    spec: str = "",
    language: str = "systemverilog",
    top: str = "",
    constraints: str = "",
    feedback: str = "",
    previous_code: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    model: str | None = None,
) -> dict | None:
    """Generate (or repair) a code_hdl block envelope via AI.

    Fresh flow: [create_code_hdl] writes a synthesizable design from the spec →
    :func:`validate_hdl_design` checks it parses, is not truncated, is named what
    was asked and is not a testbench → one repair round on failure → residual
    problems and any synthesizability warnings land on ``improvements``.

    Repair flow (``feedback`` + ``previous_code``): the shared [fix_rtl] prompt,
    validated by :func:`validate_rtl_fix`, which enforces what the prompt only
    asks for — the module name and port list stay byte-identical.

    ``top`` and ``interface`` are always parsed from the design that was actually
    emitted, never read from the model's JSON: [fix_rtl]'s output contract has no
    such keys, and a claim that disagreed with the source would be worse than no
    claim at all.

    Returns ``None`` when no LLM key is configured so callers fall back to a scaffold.
    """
    settings = get_settings()
    if not settings.openai_api_key and not getattr(settings, "anthropic_api_key", ""):
        return None

    name = block_name.replace(" ", "_")
    cat = category or "general"
    lang = _normalize_lang(language) or "systemverilog"
    if lang not in _HDL_LANGS:
        raise ValueError(
            f"'{language}' is not a hardware description language; "
            f"code_hdl accepts {', '.join(sorted(_HDL_LANGS))}"
        )
    family = hdl_family(lang)

    fixing = bool((feedback or "").strip() and (previous_code or "").strip())
    spec_text = (spec or "").strip() or (description or "").strip()
    if not fixing and not spec_text:
        raise ValueError("a spec (or description) is required to generate a design")

    section = code_hdl_prompt_section(feedback=feedback, previous_code=previous_code)
    system_prompt = get_system_prompt(section)
    if not system_prompt:
        raise ValueError(f"{section} prompt section missing from Msg_config")

    # On a repair the interface is frozen to whatever the failing design declared,
    # so it is read from that design rather than from the (possibly stale) port.
    top_name = (top or "").strip()
    if fixing:
        top_name = top_name or infer_top_module(previous_code)
        frozen_ports = module_ports(previous_code, top_name)
    else:
        top_name = top_name or name

    problems: list[str] = []
    warnings: list[str] = []
    result: dict = {}
    code = ""
    draft = previous_code
    for attempt in range(_CODE_HDL_REPAIR_ROUNDS + 1):
        if fixing:
            user_message = build_code_fix_message(
                block_name=name, language=lang, top=top_name, ports=frozen_ports,
                previous_code=draft, feedback=feedback, description=spec_text,
                problems=problems or None,
            )
        else:
            user_message = build_code_hdl_gen_message(
                block_name=name, language=lang, top=top_name, spec=spec_text,
                constraints=constraints, problems=problems or None,
            )
        result = await _call_openai_json(
            system_prompt, user_message, temperature=0.2, model=model, max_tokens=8192,
        )
        code = _strip_code_fences(str(result.get("code", "")))
        if fixing:
            problems, warnings = (validate_rtl_fix(code, previous_code, top_name), [])
        else:
            problems, warnings = validate_hdl_design(code, lang, top_name)
        if not problems:
            break
        log.warning(
            "code_hdl_validation_failed",
            block=name, attempt=attempt, fixing=fixing, problems=problems,
        )
        draft = code or draft

    # A repair that produced nothing must not blank the design it was meant to
    # improve: the app overwrites every port it is handed, so an empty `code`
    # here is data loss rather than a failed run.
    if fixing and not code.strip():
        code = previous_code

    out_top, ports = design_interface(code, lang, top_name)
    notes = list(problems)
    if family == "vhdl":
        notes.append(_VHDL_LOOP_NOTE)
    notes.extend(warnings)

    improvements = str(result.get("improvements", "")).strip()
    if notes:
        improvements = (
            "Validation: " + "; ".join(notes) + ("\n" + improvements if improvements else "")
        )

    return _generated_envelope(
        block_type="code_hdl",
        name=name,
        category=cat,
        description=description,
        inputs={
            "spec": spec_text,
            "language": lang,
            "top": top_name,
            "constraints": constraints,
            "feedback": feedback,
        },
        outputs={
            "code": code,
            "top": out_top,
            "interface": json.dumps(ports, ensure_ascii=False) if ports else "",
            "language": lang,
            "explanation": str(result.get("explanation", "")),
            "improvements": improvements,
            "status": "needs_review" if notes else "ok",
            "errors": "",
        },
        extra_inputs=inputs,
        extra_outputs=outputs,
    )


def _simple_code_hdl_response(body: CodeHdlGenerateRequest, error: str = "") -> dict:
    """Fallback: a port-complete code_hdl block with no generated design.

    A repair request keeps the design it was given. Without that the app would
    overwrite a working design with an empty string every time the key is missing
    — the run that was meant to improve it would destroy it instead.
    """
    name = body.block_name.replace(" ", "_")
    lang = _normalize_lang(body.language) or "systemverilog"
    kept = (body.previous_code or "").strip()
    note = (
        "The design was returned unchanged: HDL generation is not configured, so the "
        "reported failures could not be acted on."
        if kept else ""
    )
    out_top, ports = design_interface(kept, lang, body.top) if kept else (body.top, [])
    return _generated_envelope(
        block_type="code_hdl",
        name=name,
        category=body.category or "general",
        description=(body.description or body.spec or "").strip(),
        inputs={
            "spec": (body.spec or "").strip(),
            "language": lang,
            "top": body.top,
            "constraints": body.constraints,
            "feedback": body.feedback,
        },
        outputs={
            "code": kept,
            "top": out_top,
            "interface": json.dumps(ports, ensure_ascii=False) if ports else "",
            "language": lang,
            "improvements": note,
            "status": "error" if error else "",
            "errors": error,
        },
        extra_inputs=body.inputs,
        extra_outputs=body.outputs,
    )


@router.post("/generate/code_hdl")
async def generate_code_hdl_block(
    body: CodeHdlGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Generate or repair a code_hdl block (spec-derived RTL) using AI, with fallback.

    Serves the manual UnifiedWindow creation path and the app's Run/Regenerate
    buttons, including the verify loop's repair leg. Without a key, or on failure,
    returns a port-complete stub whose ``errors``/``status`` ports say why — and
    which never blanks an existing design.
    """
    settings = get_settings()
    if not settings.openai_api_key and not getattr(settings, "anthropic_api_key", ""):
        log.info("blocks_generate_code_hdl_fallback", reason="no_llm_key", block=body.block_name)
        return _simple_code_hdl_response(body, error="AI not configured")
    try:
        result = await generate_code_hdl_payload(
            block_name=body.block_name,
            category=body.category,
            description=body.description,
            spec=body.spec,
            language=body.language,
            top=body.top,
            constraints=body.constraints,
            feedback=body.feedback,
            previous_code=body.previous_code,
            inputs=body.inputs,
            outputs=body.outputs,
            model=body.run_llm_model or None,
        )
        if result is None:
            return _simple_code_hdl_response(body, error="AI not configured")
        log.info(
            "blocks_generate_code_hdl_ok",
            block=body.block_name, language=body.language,
            fixing=bool(body.feedback and body.previous_code),
        )
        return result
    except Exception as exc:
        log.error("blocks_generate_code_hdl_error", block=body.block_name, error=str(exc))
        return _simple_code_hdl_response(body, error=str(exc))


# ── Scaffold-only blocks (no AI call, never need a key) ─────────────────────────
#
# Several block types produce their real content at *Run* time, not at creation:
# image/location/live/stream call a Grafux-interaction service; gpu/claw/devices
# call the devices server; memory/selection/filter are wired to other blocks.
# For these, creation just lays out the canonical input/output ports so the block
# has the right shape — identical to what the manual UnifiedWindow dialog builds
# (see Grafux-app/src/ui/dialogs/unifiedblockcreationdialog.cpp). Each spec mirrors
# that dialog's generate*/finalize* port lists.
#
# Spec fields:
#   category_based     — folder layout data/<type>/<category>/<name> vs data/<type>/<name>
#   inputs / outputs   — canonical port names (block_description is always prepended)
#   seed_map           — {seed key from the create_block action → input port to fill}
#   seed_from_desc     — input port seeded from the block description when no seed given
#   defaults           — input port → default content (matches the dialog's seeds)
class _ScaffoldSpec(NamedTuple):
    category_based: bool
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    seed_map: dict[str, str] = {}
    seed_from_desc: str | None = None
    defaults: dict[str, str] = {}


_SCAFFOLD_SPECS: dict[str, _ScaffoldSpec] = {
    "image": _ScaffoldSpec(
        category_based=True,
        inputs=("prompt", "modification", "search_for", "image"),
        outputs=("image", "image_name", "image_description", "improvements", "status"),
        seed_from_desc="prompt",
    ),
    "location": _ScaffoldSpec(
        category_based=False,
        inputs=("name", "full_address", "street", "postal_code", "city", "country"),
        outputs=("name", "full_address", "street", "postal_code", "city", "country", "status"),
        seed_map={"address": "full_address"},
        seed_from_desc="full_address",
    ),
    "live": _ScaffoldSpec(
        category_based=False,
        inputs=("live_stream_link", "question"),
        outputs=("answer", "transcript", "status"),
        seed_map={"url": "live_stream_link"},
    ),
    "stream": _ScaffoldSpec(
        category_based=False,
        inputs=("question",),
        outputs=("answer", "transcript", "status"),
    ),
    "gpu": _ScaffoldSpec(
        category_based=True,
        inputs=("gpu_model", "image", "cloud_type", "compile_flags", "code", "language",
                "args", "timeout", "keep_warm_minutes", "files", "output_globs",
                "api_keys", "credentials"),
        outputs=("response", "status", "gpu_id", "errors", "warnings", "benchmark",
                 "artifacts", "warm_until", "cost"),
        seed_map={"gpu_model": "gpu_model", "language": "language"},
    ),
    "claw": _ScaffoldSpec(
        category_based=True,
        inputs=("soul", "skills", "agent", "credentials", "api_keys", "task",
                "text_message", "memory", "tools_config", "connections"),
        outputs=("response", "status", "claw_id", "errors"),
    ),
    "devices": _ScaffoldSpec(
        category_based=True,
        inputs=("device_id", "command", "code", "language", "args", "timeout", "text", "file"),
        outputs=("results", "response", "status", "errors", "terminal", "file",
                 "warnings", "improvements"),
        defaults={"language": "cpp", "timeout": "130"},
    ),
    "memory": _ScaffoldSpec(
        category_based=False,
        inputs=("input",),
        outputs=(),
    ),
    "selection": _ScaffoldSpec(
        category_based=False,
        inputs=("criteria",),
        outputs=("selected", "analysis"),
    ),
    "filter": _ScaffoldSpec(
        category_based=False,
        inputs=("code", "criteria", "input"),
        outputs=("filtered", "analysis", "errors", "warnings", "improvements", "code"),
    ),
    # The whiteboard block draws on a Miro board at Run: a filled "prompt" has the AI
    # design the board, otherwise "notes" plus any FURTHER input port the user wires in
    # becomes a sticky note. "board_id" is filled by the first Run so a later
    # Run/Regenerate extends the same board instead of creating another one.
    "white_board": _ScaffoldSpec(
        category_based=False,
        inputs=("board_name", "prompt", "notes", "board_id"),
        outputs=("board_url", "embed_url", "board_id", "summary", "status"),
        seed_from_desc="prompt",
    ),
    # ------------------------------------------------------------------
    # Chip design (EDA).  These three form a pipeline on the canvas:
    #   code (language=verilog) -> verilator -> yosys -> openroad
    # so the ports are named to make the obvious wiring the natural one:
    # verilator echoes "rtl" through, yosys consumes "rtl" and emits "netlist",
    # openroad consumes "netlist".
    #
    # Each list must stay IDENTICAL to the Qt dialog's finalize<Type>() port
    # arrays and to the executor's build<Type>Outputs(); tests below assert the
    # exact sets, because a silent mismatch means AI-created and hand-created
    # blocks get different ports.
    # ------------------------------------------------------------------
    # Simulate or lint a design. The block a user reaches for when the AI just
    # wrote Verilog and they want to know whether it actually works.
    # The verification inputs sit next to the port they modify rather than at the
    # end, because the port order is what the user reads off the block face.
    # "mode" stays "sim": a Python testbench is recognised server-side and run as
    # cocotb, so wiring a testbench block into an existing verilator block works
    # without anyone remembering to change a dropdown.
    # "max_iterations" is the ONLY port here the server never sees — the fix loop
    # runs in the app, and the server runs exactly one simulation per request.
    "verilator": _ScaffoldSpec(
        category_based=True,
        inputs=("rtl", "testbench", "sva", "top", "mode", "simulator", "tests",
                "seed", "collect_coverage", "max_iterations", "defines",
                "include_dirs", "files", "trace", "sim_args", "verilator_flags",
                "timeout", "instance_type", "image", "api_keys"),
        outputs=("status", "passed", "results", "failures", "coverage",
                 "coverage_report", "iterations", "sim_output", "lint", "errors",
                 "warnings", "waveform", "rtl", "top", "log", "artifacts",
                 "eda_id", "cost", "improvements_rtl", "improvements_test"),
        seed_map={"top": "top"},
        defaults={"mode": "sim", "trace": "1", "timeout": "900",
                  "simulator": "verilator", "collect_coverage": "1",
                  "max_iterations": "1"},
    ),
    # Synthesize RTL into a gate-level netlist mapped onto the PDK's cells.
    "yosys": _ScaffoldSpec(
        category_based=True,
        inputs=("rtl", "top", "pdk", "liberty", "synth_flags", "defines",
                "include_dirs", "files", "timeout", "instance_type", "image",
                "api_keys", "credentials"),
        outputs=("netlist", "status", "top", "pdk", "stats", "report", "errors",
                 "warnings", "log", "artifacts", "eda_id", "cost"),
        seed_map={"top": "top", "pdk": "pdk"},
        defaults={"pdk": "sky130hd", "timeout": "900"},
    ),
    # Floorplan -> place -> CTS -> route -> GDS. "rtl" is a fallback: with the
    # netlist port empty the flow runs its own synthesis, so the block still works
    # standalone without a yosys block upstream.
    "openroad": _ScaffoldSpec(
        category_based=True,
        inputs=("netlist", "rtl", "top", "pdk", "sdc", "clock_port", "clock_period",
                "core_utilization", "aspect_ratio", "die_area", "core_area",
                "place_density", "from_stage", "to_stage", "extra_config", "files",
                "timeout", "instance_type", "image", "api_keys"),
        outputs=("status", "stage", "gds", "def", "netlist_final", "spef",
                 "layout_png", "metrics", "reports", "errors", "warnings", "log",
                 "artifacts", "eda_id", "cost", "improvements"),
        seed_map={"top": "top", "pdk": "pdk", "clock_period": "clock_period"},
        defaults={"pdk": "sky130hd", "clock_port": "clk", "clock_period": "10",
                  "core_utilization": "45", "aspect_ratio": "1",
                  "from_stage": "synth", "to_stage": "final", "timeout": "7200"},
    ),
    # Verification. The testbench block is AI-generated (like code) but is listed
    # here too so the create path always lays out the same ports whether or not
    # the model produced content:
    #   code.code -> testbench.rtl ; testbench.testbench -> verilator.testbench
    # "spec" is the behaviour under test (seeded from the description when the
    # user did not separate the two); "feedback" is a reviewer's note for a
    # testbench repair round; "top" is echoed so verilator can be wired off it.
    "testbench": _ScaffoldSpec(
        category_based=True,
        inputs=("spec", "rtl", "top", "framework", "style", "coverage_goals",
                "extra_tests", "feedback"),
        outputs=("testbench", "sva", "test_plan", "top", "explanation",
                 "improvements", "status", "errors"),
        seed_map={"top": "top", "spec": "spec"},
        seed_from_desc="spec",
        defaults={"framework": "cocotb", "style": "directed+random"},
    ),
    # The RTL source of that same loop. Kept apart from the code block because a
    # design is not a program: it needs a spec rather than a description, a module
    # name every downstream block addresses it by, and synthesizability rules a
    # general-purpose programming prompt has no reason to know about.
    #   code_hdl.code -> testbench.rtl and verilator.rtl ; code_hdl.top -> both tops
    #   verilator.failures -> code_hdl.feedback   (written by the client verify loop)
    # "interface" is OUT only: it is the port list parsed from the design that was
    # actually emitted, so it is evidence rather than a request. A pinned interface
    # would be a second source of truth that nothing enforces; say it in
    # "constraints" instead. "previous_code" is likewise absent — the repair path
    # reads the block's own `code` output.
    "code_hdl": _ScaffoldSpec(
        category_based=True,
        inputs=("spec", "language", "top", "constraints", "feedback"),
        outputs=("code", "top", "interface", "language", "explanation",
                 "improvements", "status", "errors"),
        seed_map={"top": "top", "spec": "spec", "language": "language"},
        seed_from_desc="spec",
        defaults={"language": "systemverilog"},
    ),
    # The CONTRACT the two blocks above are both derived from. It exists because
    # a design and its tests only agree about "correct" when they are handed the
    # SAME spec, and because prose leaves the parameters a design always needs
    # implicit — widths, clocking, reset polarity, combinational vs sequential.
    #   spec_hdl.spec -> code_hdl.spec AND testbench.spec   (the same text, twice)
    #   spec_hdl.top  -> code_hdl.top  AND testbench.top
    #   code_hdl.code -> spec_hdl.previous_code
    #   verilator.improvements_rtl (or failures) -> spec_hdl.feedback
    # That last wire closes the OUTER loop: [fix_rtl] repairs a design against
    # failing tests, but when a design and its tests disagree the fault is often
    # the contract they were both written from, and nothing could repair that.
    #
    # "explanation" is in BOTH lists and is NOT echoed through — the rare
    # exception to the rule the verilator comment states. The input is the rough
    # human description this block starts from; the output is the plain-language
    # summary of the finished spec. They are the same word for the same idea at
    # opposite ends of the block, and renaming either to dodge the collision
    # ("summary", "brief") would make the user hunt for the port that holds what
    # they typed. Same for "top" and "parameters": pinned on the way in,
    # resolved on the way out.
    #
    # "interface" is OUT only, as on code_hdl and for the same reason: a pinned
    # interface would be a second source of truth nothing enforces. Here it is a
    # PROPOSAL — what the spec implies — while code_hdl's is evidence parsed
    # from the design it emitted.
    "spec_hdl": _ScaffoldSpec(
        category_based=True,
        inputs=("explanation", "previous_code", "top", "language", "data_width",
                "addr_width", "parameters", "logic_style", "reset_style",
                "clocking", "protocol", "throughput", "constraints", "feedback"),
        outputs=("spec", "top", "interface", "signals_analysis", "parameters",
                 "timing", "requirements", "assumptions", "explanation",
                 "improvements", "status", "errors"),
        seed_map={"top": "top", "language": "language", "spec": "explanation"},
        seed_from_desc="explanation",
        defaults={"language": "systemverilog", "logic_style": "auto",
                  "reset_style": "sync_active_high", "clocking": "single_clock"},
    ),
}


# The memory block is the one type whose port shape depends on a PARAM rather than
# only on the type: memory_mode picks between three quite different blocks. The
# entry in _SCAFFOLD_SPECS above stays the snapshot shape so a mode-unaware caller
# still gets a working block; _enrich_memory_block passes the right one of these as
# a spec override. Mirrors UnifiedBlockCreationDialog::generateMemory().
#   snapshot   — one "input"; the timestamped outputs are created at Run.
#   sequential — the data inputs are user-named, so only the output is canonical.
#   accumulate — "data" in; the record and its AI review out.
MEMORY_MODES = ("snapshot", "sequential", "accumulate")

_MEMORY_MODE_SPECS: dict[str, _ScaffoldSpec] = {
    "snapshot": _ScaffoldSpec(category_based=False, inputs=("input",), outputs=()),
    "sequential": _ScaffoldSpec(category_based=False, inputs=("input",), outputs=("output",)),
    "accumulate": _ScaffoldSpec(
        category_based=False,
        inputs=("data",),
        outputs=("accumulated_data", "analysis"),
    ),
}


def memory_scaffold_spec(memory_mode: str) -> _ScaffoldSpec:
    """The scaffold spec for a memory mode, falling back to snapshot."""
    return _MEMORY_MODE_SPECS.get(
        (memory_mode or "").strip().lower(), _MEMORY_MODE_SPECS["snapshot"]
    )



def _scaffold_port_path(
    block_type: str, category_based: bool, category: str, name: str,
    port_type: str, port_name: str,
) -> str:
    if category_based:
        return f"data/{block_type}/{category}/{name}/{port_type}/{port_name}.txt"
    return f"data/{block_type}/{name}/{port_type}/{port_name}.txt"


async def generate_scaffold_payload(
    *,
    block_type: str,
    block_name: str,
    category: str = "general",
    description: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    seeds: dict[str, str] | None = None,
    spec_override: _ScaffoldSpec | None = None,
) -> dict | None:
    """Build a scaffold-only block envelope (the ``tool_calls`` dict) — no AI call.

    Lays out the canonical ports for ``block_type`` per ``_SCAFFOLD_SPECS`` (matching the
    manual dialog), prepending ``block_description`` and appending any extra requested
    ``inputs``/``outputs``. ``seeds`` carries the primary-input values parsed from the user's
    command (e.g. ``{"address": "Eiffel Tower"}`` → the ``full_address`` port) so the block is
    ready to Run. Always succeeds and never needs an API key; returns ``None`` for an unknown
    type so the caller falls back to its own stub behavior.

    ``spec_override`` replaces the registry entry for this one call. It exists for the
    memory block, whose port shape is chosen by its ``memory_mode`` param rather than
    by its type alone (see :func:`memory_scaffold_spec`).
    """
    spec = spec_override or _SCAFFOLD_SPECS.get(block_type)
    if spec is None:
        return None

    name = block_name.replace(" ", "_")
    cat = category or "general"
    seeds = seeds or {}

    # Resolve seed content per input port: explicit seed keys win, then description, then defaults.
    port_seed: dict[str, str] = {}
    for seed_key, port in spec.seed_map.items():
        val = str(seeds.get(seed_key, "")).strip()
        if val:
            port_seed[port] = val
    if spec.seed_from_desc and spec.seed_from_desc not in port_seed:
        desc = (description or "").strip()
        if desc:
            port_seed[spec.seed_from_desc] = desc
    for port, default in spec.defaults.items():
        port_seed.setdefault(port, default)

    def _path(port_type: str, port_name: str) -> str:
        return _scaffold_port_path(block_type, spec.category_based, cat, name, port_type, port_name)

    ip = [
        {
            "port_name": "block_description",
            "port_content": description,
            "port_path": _path("inputs", "block_description"),
        }
    ]
    seen_in = {"block_description", "description"}
    for inp in list(spec.inputs) + list(inputs or []):
        if inp and inp not in seen_in:
            seen_in.add(inp)
            ip.append({
                "port_name": inp,
                "port_content": port_seed.get(inp, ""),
                "port_path": _path("inputs", inp),
            })

    op = []
    seen_out: set[str] = set()
    for out in list(spec.outputs) + list(outputs or []):
        if out and out not in seen_out:
            seen_out.add(out)
            op.append({
                "port_name": out,
                "port_content": "",
                "port_path": _path("outputs", out),
            })

    return {
        "tool_calls": [
            {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "block_id": uuid.uuid4().hex[:8],
                    "block_type": block_type,
                    "x": 0,
                    "y": 0,
                    "input_ports": ip,
                    "output_ports": op,
                },
            }
        ],
        "connections": [],
    }


async def generate_image_payload(
    *,
    block_name: str,
    category: str = "general",
    description: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict | None:
    """Build an image block envelope — thin wrapper over :func:`generate_scaffold_payload`.

    The image bytes are produced at *Run* by the image service (``Grafux-interaction/image``);
    creation only scaffolds the ports (``block_description`` + prompt/modification/search_for/
    image → image/image_name/image_description/improvements/status).
    """
    return await generate_scaffold_payload(
        block_type="image",
        block_name=block_name,
        category=category,
        description=description,
        inputs=inputs,
        outputs=outputs,
    )


@router.post("/generate/image")
async def generate_image_block(
    body: ImageGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Create an image block's port scaffold (image/image_name/description/improvements).

    Serves both the manual UnifiedWindow creation path and voice/text creation. The
    actual image is produced at Run by the image service, so this only lays out the
    ports — it always succeeds and needs no API key.
    """
    result = await generate_image_payload(
        block_name=body.block_name,
        category=body.category,
        description=body.description,
        inputs=body.inputs,
        outputs=body.outputs,
    )
    log.info("blocks_generate_image_ok", block=body.block_name)
    return result


@router.post("/run/search")
async def run_search_block(
    body: RunSearchRequest,
    user: CurrentUser,
) -> dict:
    """Fill (or, when recreate_ports is set, redesign) a search block's output ports."""
    # Regenerate uses a prompt that lets the model redesign the port set; a regular
    # Run uses the fill prompt that keeps the existing ports and only refreshes content.
    system_prompt = (
        get_system_prompt("regenerate_search") if body.recreate_ports else None
    ) or get_system_prompt("run_search")
    if not system_prompt:
        system_prompt = (
            "You are an AI assistant. Given a block name, type, and context, "
            "return a JSON object where each key is an output port name and the value "
            "is detailed content for that port. Return ONLY valid JSON."
        )

    user_message = body.context_message
    if body.recreate_ports:
        # Regenerate: existing ports are context the model MAY change, not ports to fill.
        if body.existing_output_ports:
            user_message += "\n\nCURRENT OUTPUT PORTS (you may keep, rename, remove, or add new ones):\n"
            for name in body.existing_output_ports:
                user_message += f"- {name}\n"
        user_message += (
            "\nDesign the most useful set of output ports for this block from the "
            "context above. Return a flat JSON object whose keys are the output port "
            "names you choose and whose values are the detailed content for each port.\n"
        )
    elif body.existing_output_ports:
        user_message += "\n\nOUTPUT PORTS TO FILL:\n"
        for name in body.existing_output_ports:
            user_message += f"- {name}\n"

    # Ground the output in live web data so the filled ports are real, current, and
    # source-cited instead of recalled from stale training knowledge. Auto-decide by
    # default: ground the search block types. (This used to also skip grounding when the
    # app injected already-fetched YouTube/website text under known markers; the block
    # reference fields that produced it are gone, so the markers can no longer appear.)
    # An explicit `ground` flag overrides the heuristic.
    do_ground = (
        body.ground
        if body.ground is not None
        else body.block_type in _GROUNDABLE_BLOCK_TYPES
    )
    if do_ground:
        query = (
            body.block_name.replace("_", " ") + " " + " ".join(body.existing_output_ports)
        ).strip()
        user_message = await _augment_with_grounding(user_message, query)

    try:
        result = await _call_openai_json(system_prompt, user_message, temperature=0.3, model=body.run_llm_model)
        log.info("blocks_run_search_ok", block=body.block_name, block_type=body.block_type)
        return result
    except Exception as exc:
        log.error("blocks_run_search_error", block=body.block_name, error=str(exc))
        raise


@router.post("/run/selection")
async def run_selection_block(
    body: RunSelectionRequest,
    user: CurrentUser,
) -> dict:
    """Use AI to select the best candidate matching the given criteria."""
    candidate_list = "\n".join(
        f"- {c.get('name', '')}: {c.get('value', '')}" for c in body.candidates
    )

    system_prompt = (
        "You are a selection assistant. Given a list of candidates and a selection criteria, "
        "choose the single best matching candidate. "
        "Return ONLY valid JSON with two keys: "
        '"selected" (the exact name of the chosen candidate) and '
        '"analysis" (a brief explanation of why it was selected).'
    )
    user_message = (
        f"Block: {body.block_name}\n\n"
        f"Selection criteria:\n{body.criteria}\n\n"
        f"Candidates:\n{candidate_list}"
    )

    try:
        result = await _call_openai_json(system_prompt, user_message, temperature=0.2, model=body.run_llm_model)
        log.info("blocks_run_selection_ok", block=body.block_name)
        return result
    except Exception as exc:
        log.error("blocks_run_selection_error", block=body.block_name, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Accumulate memory -> the analysis port
# ---------------------------------------------------------------------------
#
# The one thing a memory block cannot do for itself.  Appending a value to a
# growing record is file I/O, and the app does it locally before calling here --
# what needs a model is the JUDGEMENT about that value: is it new, has it been
# said already, does it disagree with something stored earlier.
#
# Server-side caps.  The app caps both halves before sending (see
# kAccumulateMax* in Grafux-app/src/ui/diagram/blocks/aiblockexecutor.cpp); these
# bound the message for a client that does not, so one Run of a long-lived block
# cannot become a 400k-token call.  The record is kept from its TAIL because the
# newest entries are what a repeat or a contradiction is most likely to be about.
_ACCUMULATE_MAX_PREVIOUS_CHARS = 60_000
_ACCUMULATE_MAX_NEW_CHARS = 20_000

_ACCUMULATE_TRUNCATION_MARKER = "[earlier entries truncated]"


def _accumulate_tail(text: str, limit: int) -> str:
    """Keep the last ``limit`` characters, marking the cut when one was made."""
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{_ACCUMULATE_TRUNCATION_MARKER}\n{text[-limit:]}"


@router.post("/run/accumulate")
async def run_accumulate_block(
    body: RunAccumulateRequest,
    user: CurrentUser,
) -> dict:
    """Review one new value against a memory block's accumulated record."""
    previous = _accumulate_tail(body.previous_accumulated, _ACCUMULATE_MAX_PREVIOUS_CHARS)
    new_data = (body.new_data or "")[:_ACCUMULATE_MAX_NEW_CHARS]

    system_prompt = (
        "You review a growing record of information for a memory block on a visual "
        "canvas. You are given the record AS IT STOOD BEFORE this run, and the ONE "
        "new piece of data that has just been appended to it. Judge ONLY the new "
        "data against the record.\n"
        "Decide three things: what the new data adds or changes relative to the "
        "record; whether it repeats something already there (the same fact, even if "
        "worded differently -- not merely the same topic); and whether it "
        "CONTRADICTS anything already there (the same subject given two "
        "incompatible values, states or claims).\n"
        f'If the record begins with "{_ACCUMULATE_TRUNCATION_MARKER}" you are seeing '
        "only its most recent part: say what you can about what is visible and do "
        "not claim that something is absent from the record.\n"
        "Return ONLY valid JSON with three keys: "
        '"analysis" (a few sentences on how the new data relates to the record), '
        '"repeated" (boolean -- true only if the record already states this), and '
        '"contradictions" (an array of strings, one per conflict, each naming the '
        "earlier claim and the new one; an empty array when there is no conflict)."
    )

    user_message = (
        f"Block: {body.block_name}\n"
        f"What this memory block is for: {body.description or '(not stated)'}\n\n"
        f"Accumulated record BEFORE this run:\n{previous or '(empty -- this is the first entry)'}\n\n"
        f"New data just appended:\n{new_data}"
    )

    try:
        result = await _call_openai_json(
            system_prompt, user_message, temperature=0.2, model=body.run_llm_model
        )
        log.info("blocks_run_accumulate_ok", block=body.block_name)
        return result
    except Exception as exc:
        log.error("blocks_run_accumulate_error", block=body.block_name, error=str(exc))
        raise


@router.post("/run/filter")
async def run_filter_block(
    body: RunFilterRequest,
    user: CurrentUser,
) -> dict:
    """Apply filter criteria to input data using AI (WASM path — no local Python)."""
    system_prompt = (
        "You are a data filtering assistant. Given input data and filter criteria, "
        "apply the criteria and return a JSON object with these keys: "
        '"filtered" (the filtered/transformed result), '
        '"analysis" (brief explanation of what was done), '
        '"errors" (any issues encountered, or empty string), '
        '"warnings" (any warnings, or empty string), '
        '"improvements" (suggestions for the filter code or criteria). '
        "Return ONLY valid JSON."
    )
    code_section = f"\nFilter code (for reference):\n{body.code}" if body.code.strip() else ""
    user_message = (
        f"Block: {body.block_name}\n"
        f"Filter type: {body.filter_type}\n"
        f"Description: {body.description}\n"
        f"Criteria: {body.criteria}\n"
        f"Input data:\n{body.input_value[:4000]}"
        f"{code_section}"
    )

    try:
        result = await _call_openai_json(system_prompt, user_message, temperature=0.2, model=body.run_llm_model)
        log.info("blocks_run_filter_ok", block=body.block_name)
        return result
    except Exception as exc:
        log.error("blocks_run_filter_error", block=body.block_name, error=str(exc))
        raise



# ---------------------------------------------------------------------------
# Run review -> the improvements ports
# ---------------------------------------------------------------------------
#
# The improvements ports are the one output an EDA/device run CANNOT produce:
# they are a REVIEW of the run, not a product of it.  The devices server emits
# results, coverage and logs; turning those into "here is what to change" needs
# a model, so it lands here rather than in Grafux-devices.
#
# One prompt section serves all three kinds.  The output contract is identical
# (prose buckets), so three sections would triple the OUTPUT FORMAT/PROHIBITIONS
# boilerplate AND need a kind->section map -- a second BLOCK_TYPE_SECTION, which
# app/core/constants.py explicitly warns against.  The evidence is already
# labelled by port name in the user message, so the model needs a lens, not a
# different persona.

# Server-side belt-and-braces cap.  The app caps every field before sending
# (EdaImprovements::buildRequestBody); this bounds the whole message so a client
# that does not cap cannot turn one finished run into a 400k-token call.
_IMPROVEMENTS_MAX_CHARS = 80_000

# Evidence dropped first when the message is over budget: the least specific
# sections, in order.  Everything not listed is kept, so `failures`, `results`
# and `rtl` -- the sections the review is actually built from -- are the last
# things to go.  Mirrors the client-side drop order deliberately.
_IMPROVEMENTS_DROP_ORDER = ("log", "sim_output", "reports", "artifacts", "coverage")

# The buckets the model returns, mapped onto ports by the app.  Named for what
# they MEAN rather than for the verilator port names, because "improvements_rtl"
# is a nonsense label for a device block running Python.
_IMPROVEMENTS_KEYS = ("design", "tests", "summary")


def build_improvements_message(
    *,
    kind: str,
    block_description: str = "",
    verdict: str = "",
    run: dict[str, str] | None = None,
) -> str:
    """Assemble the user message ``[improve_run]`` expects, under the size cap.

    Sections are emitted in a stable order so the model's attention lands the
    same way on every call, and each carries its PORT NAME as the label -- that
    label is what the prompt tells the model to cite as evidence.
    """
    run = run or {}
    head = [f"KIND: {kind}"]
    if verdict:
        head.append(f"VERDICT: {verdict}")
    if block_description.strip():
        head.append(f"DESCRIPTION: {block_description.strip()}")

    # Drop the least specific evidence until the body fits, rather than
    # truncating blindly: a review that lost `failures` to make room for a log
    # tail is worse than one that never saw the log.
    sections = dict(run)
    def render() -> str:
        parts = ["\n".join(head)]
        for name, text in sections.items():
            if not (text or "").strip():
                continue
            parts.append(f"--- {name} ---\n{text.strip()}")
        return "\n\n".join(parts)

    body = render()
    for droppable in _IMPROVEMENTS_DROP_ORDER:
        if len(body) <= _IMPROVEMENTS_MAX_CHARS:
            break
        if sections.pop(droppable, None) is not None:
            body = render()

    if len(body) > _IMPROVEMENTS_MAX_CHARS:
        body = body[:_IMPROVEMENTS_MAX_CHARS] + "\n...[truncated]"
    return body


async def generate_improvements_payload(
    *,
    block_name: str,
    kind: str = "device",
    block_description: str = "",
    verdict: str = "",
    run: dict[str, str] | None = None,
    model: str | None = None,
) -> dict | None:
    """Review a finished run.  Returns {"design", "tests", "summary"} or None.

    ``None`` means "no LLM configured".  The caller answers with a graceful
    empty result rather than raising: the run itself already SUCCEEDED, and a
    review outage must never turn a green run red on the canvas.
    """
    settings = get_settings()
    if not (getattr(settings, "openai_api_key", "") or getattr(settings, "anthropic_api_key", "")):
        return None

    system_prompt = get_system_prompt("improve_run")
    if not system_prompt:
        raise ValueError("improve_run prompt section is missing from Msg_config")

    user_message = build_improvements_message(
        kind=kind, block_description=block_description, verdict=verdict, run=run
    )
    raw = await _call_openai_json(
        system_prompt, user_message, temperature=0.3, model=model, max_tokens=2048
    )
    return {key: str(raw.get(key, "") or "").strip() for key in _IMPROVEMENTS_KEYS}


@router.post("/run/improvements")
async def run_improvements(body: ImprovementsRequest, user: CurrentUser) -> dict:
    """Review a finished verilator / openroad / device run for its improvements ports.

    NEVER raises.  Every failure path returns HTTP 200 with ``status="error"``
    and empty buckets, because this endpoint is called AFTER the run has already
    been reported to the user -- see generate_improvements_payload.
    """
    try:
        payload = await generate_improvements_payload(
            block_name=body.block_name,
            kind=body.kind,
            block_description=body.block_description,
            verdict=body.verdict,
            run=body.run,
            model=body.run_llm_model,
        )
        if payload is None:
            log.info("blocks_improvements_no_key", block=body.block_name, kind=body.kind)
            return {
                "design": "",
                "tests": "",
                "summary": "",
                "status": "error",
                "errors": "no AI key configured, so the run could not be reviewed",
            }
        log.info("blocks_improvements_ok", block=body.block_name, kind=body.kind)
        return {**payload, "status": "ok", "errors": ""}
    except Exception as exc:  # noqa: BLE001 - deliberately total; see docstring
        log.error("blocks_improvements_error", block=body.block_name, error=str(exc))
        return {"design": "", "tests": "", "summary": "", "status": "error", "errors": str(exc)}

@router.post("/regenerate/tool")
async def regenerate_tool_block(
    body: RegenerateToolRequest,
    user: CurrentUser,
) -> dict:
    """Regenerate a tool block's code in the MCP @register_tool format.

    Uses the shared [create_tool] prompt so the regenerated file matches what the manual
    UnifiedWindow path produces (decorator + file-based input/output scaffold). The app
    assembles `body.prompt` with the tool name, input port names, description, and current
    code, which is exactly the requirement text the prompt expects.
    """
    try:
        code = await generate_tool_code(body.prompt, temperature=0.2, model=body.regen_llm_model)
        log.info("blocks_regenerate_tool_ok", block=body.block_name)
        # Leave description empty so the app keeps the user's existing description port;
        # only the code + change summary are authoritative here.
        return {
            "code": code,
            "description": "",
            "change": "Regenerated tool code in MCP @register_tool format.",
        }
    except Exception as exc:
        log.error("blocks_regenerate_tool_error", block=body.block_name, error=str(exc))
        raise


@router.post("/regenerate/filter")
async def regenerate_filter_block(
    body: RegenerateFilterRequest,
    user: CurrentUser,
) -> dict:
    """Regenerate a filter block's Python code using AI."""
    system_prompt = (
        "You are an expert Python developer. Generate a Python filter script based on the given description and criteria. "
        "The script should read from input_path and criteria_path, filter/transform the data, and write results to output files. "
        "Return ONLY valid JSON with one key: "
        '"code" (the complete Python filter script). '
        "The script must handle errors gracefully and write to the output files specified in config."
    )

    try:
        result = await _call_openai_json(system_prompt, body.prompt, temperature=0.3, model=body.regen_llm_model)
        log.info("blocks_regenerate_filter_ok", block=body.block_name)
        return result
    except Exception as exc:
        log.error("blocks_regenerate_filter_error", block=body.block_name, error=str(exc))
        raise
