from __future__ import annotations

import datetime
import json
import uuid

from fastapi import APIRouter

from app.config import get_settings
from app.core.logging import get_logger
from app.dependencies import CurrentUser
from app.modules.blocks.schemas import (
    TopicGenerateRequest,
    RunSearchRequest,
    RunSelectionRequest,
    RunFilterRequest,
    RegenerateToolRequest,
    RegenerateFilterRequest,
)
from app.prompts import get_system_prompt, get_json_schema

log = get_logger("blocks.router")

# Map block_type → Msg_config section name
_BLOCK_TYPE_SECTION: dict[str, str] = {
    "topics": "create_topic",
    "components": "create_component",
    "commands": "create_cmd",
    "tools": "create_tool",
    "procedures": "create_procedure",
}

router = APIRouter(prefix="/blocks", tags=["blocks"])


def _make_port_path(category: str, name: str, port_type: str, port_name: str) -> str:
    return f"data/topics/{category}/{name}/{port_type}/{port_name}.txt"


def _simple_topic_response(body: TopicGenerateRequest) -> dict:
    """Fallback: build a minimal block without AI when OpenAI is not configured."""
    name = body.topic_name.replace(" ", "_")
    cat = body.category

    ip = [
        {
            "port_name": "description",
            "port_content": body.description,
            "port_path": _make_port_path(cat, name, "inputs", "description"),
        }
    ]
    for inp in body.inputs:
        if inp and inp != "description":
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

# Search-block types whose Run/Regenerate output should be grounded in live web data.
_GROUNDABLE_BLOCK_TYPES = {"topics", "components", "procedures"}


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

        # Deep-crawl the top result(s) for richer, current content.
        if crawl_top > 0 and settings.firecrawl_api_key:
            firecrawl = FirecrawlClient()
            for source in sources[:crawl_top]:
                scraped = await firecrawl.scrape(source.url)
                md = scraped.get("markdown", "")
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
) -> dict | None:
    """Generate a grounded topic block envelope (the `tool_calls` dict) via AI.

    Shared core of the topic generator: builds the prompt, runs live web grounding when the
    description is a short query, calls OpenAI with the [create_topic] prompt, and returns the
    parsed ``{tool_calls, ...}`` result with output ports filled with real, current content.

    Returns ``None`` when OpenAI is not configured or generation fails, so callers can fall back
    (the REST endpoint to a simple template, the stream path to leaving the action unchanged).
    """
    settings = get_settings()
    if not settings.openai_api_key:
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
        user_message += f"Requested input ports (besides 'description'): {', '.join(inputs)}\n"
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

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    result: dict = json.loads(content)

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

    if not settings.openai_api_key:
        log.info("blocks_generate_topic_fallback", reason="no_openai_key", topic=body.topic_name)
        return _simple_topic_response(body)

    try:
        result = await generate_topic_payload(
            topic_name=body.topic_name,
            category=body.category,
            description=body.description,
            inputs=body.inputs,
            outputs=body.outputs,
            ground=body.ground,
        )
        if result is None:
            return _simple_topic_response(body)
        log.info("blocks_generate_topic_ok", topic=body.topic_name)
        return result

    except Exception as exc:
        log.error("blocks_generate_topic_error", topic=body.topic_name, error=str(exc))
        # Graceful fallback to simple template
        return _simple_topic_response(body)


async def _call_openai_json(system_prompt: str, user_message: str, temperature: float = 0.3) -> dict:
    """Shared helper: call OpenAI with response_format=json_object and return parsed dict."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(response.choices[0].message.content or "{}")


async def _call_openai_text(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    """Shared helper: call OpenAI for raw text (no JSON mode) and return the message text."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _strip_code_fences(text: str) -> str:
    """Drop a leading/trailing markdown ``` fence if the model added one despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]          # drop the opening ``` / ```python line
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]     # drop the closing fence
    return t.strip()


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


async def generate_tool_code(user_message: str, *, temperature: float = 0.2) -> str:
    """Generate a complete @register_tool-formatted Python file from a tool requirement.

    Uses the shared [create_tool] Msg_config section — the same structural contract the
    manual UnifiedWindow path produces — so voice/text tools fit the MCP server. Returns
    raw Python source (markdown fences stripped).
    """
    system_prompt = get_system_prompt(_BLOCK_TYPE_SECTION["tools"])
    if not system_prompt:
        raise ValueError("create_tool prompt section missing from Msg_config")
    raw = await _call_openai_text(system_prompt, user_message, temperature=temperature)
    return _strip_code_fences(raw)


@router.post("/run/search")
async def run_search_block(
    body: RunSearchRequest,
    user: CurrentUser,
) -> dict:
    """Fill (or, when recreate_ports is set, redesign) a search block's output ports."""
    settings = get_settings()

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
        result = await _call_openai_json(system_prompt, user_message, temperature=0.3)
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
    candidate_names = [c.get("name", "") for c in body.candidates]
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
        result = await _call_openai_json(system_prompt, user_message, temperature=0.2)
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
        result = await _call_openai_json(system_prompt, user_message, temperature=0.2)
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
        code = await generate_tool_code(body.prompt, temperature=0.2)
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
        result = await _call_openai_json(system_prompt, body.prompt, temperature=0.3)
        log.info("blocks_regenerate_filter_ok", block=body.block_name)
        return result
    except Exception as exc:
        log.error("blocks_regenerate_filter_error", block=body.block_name, error=str(exc))
        raise
