from __future__ import annotations

import asyncio
import datetime
import uuid

from fastapi import APIRouter

from app.config import get_settings
from app.core.constants import (
    BLOCK_TYPE_SECTION as _BLOCK_TYPE_SECTION,
)
from app.core.constants import (
    GROUNDABLE_BLOCK_TYPES as _GROUNDABLE_BLOCK_TYPES,
)
from app.core.llm import _strip_code_fences, call_llm_json, call_llm_text
from app.core.logging import get_logger
from app.dependencies import CurrentUser
from app.modules.blocks.schemas import (
    CodeGenerateRequest,
    ImageGenerateRequest,
    RegenerateFilterRequest,
    RegenerateToolRequest,
    RunFilterRequest,
    RunSearchRequest,
    RunSelectionRequest,
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
# live web results. A long description is already-fetched page content (the app puts
# YouTube/website text here) and is trusted as-is — no live search.
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
) -> dict | None:
    """Generate a grounded topic block envelope (the `tool_calls` dict) via AI.

    Shared core of the topic generator: builds the prompt, runs live web grounding when the
    description is a short query, calls OpenAI with the [create_topic] prompt, and returns the
    parsed ``{tool_calls, ...}`` result with output ports filled with real, current content.

    Returns ``None`` when OpenAI is not configured or generation fails, so callers can fall back
    (the REST endpoint to a simple template, the stream path to leaving the action unchanged).
    """
    settings = get_settings()
    if not settings.openai_api_key and not settings.anthropic_api_key:
        return None

    inputs = inputs or []
    outputs = outputs or []
    name = topic_name.replace(" ", "_")
    cat = category or "general"

    section = _BLOCK_TYPE_SECTION.get("topics", "create_topic")
    system_prompt = (
        get_system_prompt(section)
        + "\n\nJSON FORMAT REFERENCE:\n"
        + get_json_schema()
    )

    user_message = f"Topic name: {name}\nCategory: {cat}\n"
    if inputs:
        user_message += f"Requested input ports (besides 'block_description'): {', '.join(inputs)}\n"
    if outputs:
        user_message += f"Requested output ports: {', '.join(outputs)}\n"
    if description:
        user_message += f"\nDescription / content to extract data from:\n{description[:3000]}"

    # Ground the topic in live web data when no rich content was provided.
    do_ground = (
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


async def generate_tool_code(user_message: str, *, temperature: float = 0.2, model: str | None = None) -> str:
    """Generate a complete @register_tool-formatted Python file from a tool requirement.

    Uses the shared [create_tool] Msg_config section — the same structural contract the
    manual UnifiedWindow path produces — so voice/text tools fit the MCP server. Returns
    raw Python source (markdown fences stripped).
    """
    system_prompt = get_system_prompt(_BLOCK_TYPE_SECTION["tools"])
    if not system_prompt:
        raise ValueError("create_tool prompt section missing from Msg_config")
    raw = await _call_openai_text(system_prompt, user_message, temperature=temperature, model=model, max_tokens=8192)
    return _strip_code_fences(raw)


def _code_port_path(category: str, name: str, port_type: str, port_name: str) -> str:
    return f"data/code/{category}/{name}/{port_type}/{port_name}.txt"


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

    Returns ``None`` when OpenAI is not configured or generation fails, so callers can fall
    back (the REST endpoint to an empty-port stub, the stream path to leaving the action as-is).
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    name = block_name.replace(" ", "_")
    cat = category or "general"
    lang = (language or "python").strip()

    system_prompt = get_system_prompt(_BLOCK_TYPE_SECTION["code"])
    if not system_prompt:
        raise ValueError("create_code prompt section missing from Msg_config")

    user_message = build_code_gen_message(
        block_name=name,
        description=description,
        language=lang,
        outputs=outputs,
    )

    result = await _call_openai_json(system_prompt, user_message, temperature=0.2)
    code = _strip_code_fences(str(result.get("code", "")))

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
            "port_content": str(result.get("improvements", "")),
            "port_path": _code_port_path(cat, name, "outputs", "improvements"),
        },
        {
            "port_name": "dependencies",
            "port_content": str(result.get("dependencies", "")),
            "port_path": _code_port_path(cat, name, "outputs", "dependencies"),
        },
        {
            "port_name": "language",
            "port_content": str(result.get("language", "")).strip() or lang,
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
    ]
    for inp in inputs or []:
        if inp and inp not in ("description", "block_description", "language"):
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
    """Fallback: build a minimal code block without AI when OpenAI is not configured."""
    name = body.block_name.replace(" ", "_")
    cat = body.category or "general"
    lang = (body.language or "python").strip()
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
                    ],
                    "output_ports": [
                        {"port_name": pn,
                         "port_content": lang if pn == "language" else "",
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
            inputs=body.inputs,
            outputs=body.outputs,
        )
        if result is None:
            return _simple_code_response(body)
        log.info("blocks_generate_code_ok", block=body.block_name, language=body.language)
        return result
    except Exception as exc:
        log.error("blocks_generate_code_error", block=body.block_name, error=str(exc))
        return _simple_code_response(body)


def _image_port_path(category: str, name: str, port_type: str, port_name: str) -> str:
    return f"data/image/{category}/{name}/{port_type}/{port_name}.txt"


# Standard image-block ports. The image bytes are produced later by the image
# service on Run (see Grafux-interaction/image); creation only scaffolds the
# ports so the block has the right input/output shape to wire connections to.
_IMAGE_INPUT_PORTS = ("prompt", "modification", "search_for")
_IMAGE_OUTPUT_PORTS = ("image", "image_name", "image_description", "improvements", "status")


async def generate_image_payload(
    *,
    block_name: str,
    category: str = "general",
    description: str = "",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict | None:
    """Build an image block envelope (the ``tool_calls`` dict) — port scaffold only.

    Unlike the topic/code generators, this makes no AI call and never needs an API
    key: the image block produces its picture at *Run* time by calling the image
    service (``Grafux-interaction/image``), so creation just lays out the standard
    input ports (``block_description`` + prompt/modification/search_for) and output
    ports (``image``/``image_name``/``image_description``/``improvements``). Any
    extra requested ports are appended. Returns the ``{tool_calls, ...}`` envelope.
    """
    name = block_name.replace(" ", "_")
    cat = category or "general"

    ip = [
        {
            "port_name": "block_description",
            "port_content": description,
            "port_path": _image_port_path(cat, name, "inputs", "block_description"),
        }
    ]
    seen_in = {"block_description"}
    for inp in list(_IMAGE_INPUT_PORTS) + list(inputs or []):
        if inp and inp not in seen_in and inp not in ("description",):
            seen_in.add(inp)
            ip.append({
                "port_name": inp,
                "port_content": "",
                "port_path": _image_port_path(cat, name, "inputs", inp),
            })

    op = []
    seen_out: set[str] = set()
    for out in list(_IMAGE_OUTPUT_PORTS) + list(outputs or []):
        if out and out not in seen_out:
            seen_out.add(out)
            op.append({
                "port_name": out,
                "port_content": "",
                "port_path": _image_port_path(cat, name, "outputs", out),
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
                    "block_type": "image",
                    "x": 0,
                    "y": 0,
                    "input_ports": ip,
                    "output_ports": op,
                },
            }
        ],
        "connections": [],
    }


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
    # default: ground the search block types, but skip when the user already attached
    # reference material (the app injects it under these markers) — that content is the
    # trusted source. An explicit `ground` flag overrides the heuristic.
    do_ground = (
        body.ground
        if body.ground is not None
        else (
            body.block_type in _GROUNDABLE_BLOCK_TYPES
            and "YOUTUBE VIDEO CONTENT" not in body.context_message
            and "WEBSITE CONTENT" not in body.context_message
        )
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
