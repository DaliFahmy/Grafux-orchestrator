"""Provider-routing layer for per-block model selection.

The frontend sends the chosen model id (e.g. ``gpt-5``, ``gemini-2.5-flash``,
``claude-opus-4-8``) with each block request. This module dispatches a (system
prompt, user message) pair to the right provider based on the model-id prefix and
returns a normalized result — a parsed ``dict`` (JSON mode) or raw ``str`` (text mode),
matching the shapes the block router already consumes.

Design notes:
* Empty / unknown model id → OpenAI default (``settings.openai_model``). This is the
  non-regression guarantee: callers that send nothing behave exactly as before.
* Anthropic's Messages API differs from OpenAI — ``system`` is a top-level param (not a
  message), ``max_tokens`` is required, there is no ``response_format=json_object``, and
  ``temperature`` / ``top_p`` are rejected on Opus 4.8. The Anthropic branch handles all
  of these (JSON is coaxed via prompt + fence-stripping).
* Missing ``ANTHROPIC_API_KEY`` with a ``claude-*`` selection → graceful fallback to the
  OpenAI default, mirroring the existing no-OpenAI-key fallbacks elsewhere.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("core.llm")

_DEFAULT_MAX_TOKENS = 4096

# Shared SDK clients keyed by the running event loop. Like the HTTP client pool, an
# SDK client binds to the loop it was created on, so the web process gets one reused
# client while Celery's per-task loops each get their own. One pool per provider.
_openai_clients: dict[int, tuple[asyncio.AbstractEventLoop, Any]] = {}
_anthropic_clients: dict[int, tuple[asyncio.AbstractEventLoop, Any]] = {}
_gemini_clients: dict[int, tuple[asyncio.AbstractEventLoop, Any]] = {}


def _pooled(
    pool: dict[int, tuple[asyncio.AbstractEventLoop, Any]],
    factory,
) -> Any:
    """Return a loop-bound SDK client from ``pool``, creating one via ``factory`` if needed.

    Shared core of the per-provider getters: reuse the client bound to the running
    loop, evicting entries whose loop has closed (Celery per-task loops).
    """
    loop = asyncio.get_running_loop()
    entry = pool.get(id(loop))
    if entry is not None and entry[0] is loop:
        return entry[1]
    for key, (cached_loop, _client) in list(pool.items()):
        if cached_loop.is_closed():
            pool.pop(key, None)
    client = factory()
    pool[id(loop)] = (loop, client)
    return client


def get_async_openai() -> Any:
    """Return a pooled ``AsyncOpenAI`` client bound to the running loop.

    Reuses the underlying httpx connection pool across calls instead of paying a
    fresh TLS handshake per request. Raises ``ValueError`` if no API key is set,
    matching the existing ``_openai_chat`` guard.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    from openai import AsyncOpenAI

    return _pooled(_openai_clients, lambda: AsyncOpenAI(api_key=settings.openai_api_key))


def get_async_anthropic() -> Any:
    """Return a pooled ``AsyncAnthropic`` client bound to the running loop.

    Mirrors ``get_async_openai`` so the Anthropic branch reuses its connection pool
    instead of a fresh TLS handshake per call. The SDK import stays lazy so tests can
    monkeypatch ``anthropic.AsyncAnthropic``. Raises ``ValueError`` if no key is set.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    from anthropic import AsyncAnthropic

    return _pooled(_anthropic_clients, lambda: AsyncAnthropic(api_key=settings.anthropic_api_key))


def get_gemini_client(api_key: str) -> Any:
    """Return a pooled google-genai ``Client`` bound to the running loop.

    Only the parent ``Client`` is reused — callers still open a fresh
    ``client.aio.live.connect(...)`` session per voice relay. Reusing the parent
    avoids re-paying client construction/TLS setup on every voice session start.
    """
    from google import genai as google_genai

    return _pooled(_gemini_clients, lambda: google_genai.Client(api_key=api_key))


def resolve_provider(model_id: str | None) -> tuple[str, str]:
    """Map a model id to ``(provider, model)``.

    Empty or unrecognized ids fall back to the OpenAI default so existing traffic
    that sends no model is unaffected.
    """
    settings = get_settings()
    m = (model_id or "").strip()
    if not m:
        return "openai", settings.openai_model
    low = m.lower()
    if low.startswith("claude"):
        return "anthropic", m
    if low.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai", m
    if low.startswith("gemini"):
        return "gemini", m
    log.info("llm_unknown_model_fallback", requested=m)
    return "openai", settings.openai_model


def _strip_code_fences(text: str) -> str:
    """Drop a leading/trailing markdown ``` fence if the model added one."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _anthropic_available(provider: str) -> bool:
    """True only when the Anthropic branch can actually run (provider + key)."""
    if provider != "anthropic":
        return False
    if not get_settings().anthropic_api_key:
        log.info("llm_anthropic_key_missing_fallback")
        return False
    return True


async def _openai_chat(
    system_prompt: str,
    user_message: str,
    *,
    model: str,
    temperature: float,
    want_json: bool,
) -> str:
    """Call OpenAI and return the raw assistant text (unchanged from prior behavior)."""
    client = get_async_openai()
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
    }
    if want_json:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def _anthropic_chat(
    system_prompt: str,
    user_message: str,
    *,
    model: str,
    max_tokens: int,
    want_json: bool,
) -> str:
    """Call Anthropic's Messages API and return the first text block.

    No ``temperature`` is passed (rejected on Opus 4.8). For JSON mode we append a
    strict JSON-only instruction; the caller strips fences and parses.
    """
    system = system_prompt
    if want_json:
        system = (
            system_prompt
            + "\n\nReturn ONLY a single valid JSON object. "
            "No markdown, no code fences, no prose."
        )

    client = get_async_anthropic()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")


async def call_llm_text(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """Provider-routed text completion. Returns raw assistant text."""
    provider, resolved = resolve_provider(model)
    if _anthropic_available(provider):
        return await _anthropic_chat(
            system_prompt, user_message,
            model=resolved, max_tokens=max_tokens, want_json=False,
        )
    # openai (default) and gemini (phase-1 fallback) both run the OpenAI branch.
    if provider == "gemini":
        log.info("llm_gemini_text_unimplemented_fallback", requested=resolved)
    settings = get_settings()
    openai_model = resolved if provider == "openai" else settings.openai_model
    return await _openai_chat(
        system_prompt, user_message,
        model=openai_model, temperature=temperature, want_json=False,
    )


async def call_llm_json(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    """Provider-routed JSON completion. Returns a parsed ``dict``.

    Raises ``json.JSONDecodeError`` / ``ValueError`` on failure — the same surface the
    existing callers already catch, so their graceful fallbacks still fire.
    """
    provider, resolved = resolve_provider(model)
    if _anthropic_available(provider):
        raw = await _anthropic_chat(
            system_prompt, user_message,
            model=resolved, max_tokens=max_tokens, want_json=True,
        )
        return json.loads(_strip_code_fences(raw) or "{}")
    if provider == "gemini":
        log.info("llm_gemini_json_unimplemented_fallback", requested=resolved)
    settings = get_settings()
    openai_model = resolved if provider == "openai" else settings.openai_model
    raw = await _openai_chat(
        system_prompt, user_message,
        model=openai_model, temperature=temperature, want_json=True,
    )
    return json.loads(raw or "{}")


def make_chat_model(model_id: str | None, *, streaming: bool = False):
    """LangChain chat-model factory routed by model id.

    Returns ``ChatAnthropic`` for ``claude-*`` (when a key is configured), else
    ``ChatOpenAI``. Both expose the same ``.bind_tools`` / ``.ainvoke`` interface used by
    the agent runtime and workflow engine.
    """
    provider, resolved = resolve_provider(model_id)
    settings = get_settings()
    if provider == "anthropic" and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=resolved,
            api_key=settings.anthropic_api_key,
            max_tokens=_DEFAULT_MAX_TOKENS,
            streaming=streaming,
        )
    if provider == "anthropic":
        log.info("llm_anthropic_key_missing_fallback", context="make_chat_model")
    from langchain_openai import ChatOpenAI

    openai_model = resolved if provider == "openai" else settings.openai_model
    return ChatOpenAI(
        model=openai_model,
        api_key=settings.openai_api_key,
        streaming=streaming,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool calling
#
# ``call_llm_text`` / ``call_llm_json`` above are single-shot: one (system, user)
# pair in, one string out. An agent needs the other shape — a growing transcript,
# a tool list, and a turn that may come back asking to CALL something. That is
# what this section adds, and it is the primitive the block agents are built on.
#
# The transcript passed in is PROVIDER-NEUTRAL; each branch translates it on the
# way out. That is deliberate: a block's Agent model dropdown can switch provider
# between steps, and a transcript stored in one vendor's wire format could not
# survive that. Neutral shapes:
#
#     {"role": "user",      "content": "..."}
#     {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...]}
#     {"role": "tool",      "tool_call_id": "...", "name": "...",
#                           "content": "...", "is_error": False}
#     {"role": "system",    "content": "..."}   # mid-conversation, see below
#
# Tool declarations are neutral too — the {"name", "description", "parameters"}
# shape ``canvas_tools._func`` already emits, so CANVAS_FUNCTION_DECLARATIONS is
# usable verbatim.
#
# On caching: an agent re-sends its whole context every step, so the top-level
# ``system`` is kept FROZEN and cached, and volatile context (the canvas render,
# the journal) is passed as a mid-conversation {"role": "system"} entry inside
# ``messages``. Caching is a prefix match — putting the canvas in the top-level
# system block would invalidate the cache on every single step, which looks like
# caching without being it.
# ─────────────────────────────────────────────────────────────────────────────

# Non-streaming agent steps: big enough that a long tool call is never truncated
# mid-argument, small enough to stay under the SDK's HTTP timeout. Separate from
# _DEFAULT_MAX_TOKENS so the single-shot callers above keep their old ceiling.
_AGENT_MAX_TOKENS = 16000

# Models that take `thinking: {"type": "adaptive"}` and `output_config.effort`.
# Everything else (Haiku 4.5 and older) still wants the removed budget_tokens
# form, so it gets neither parameter rather than a 400.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


def _supports_adaptive_thinking(model: str) -> bool:
    return model.lower().startswith(_ADAPTIVE_THINKING_PREFIXES)


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked to call."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolTurn:
    """One assistant turn: prose, plus whatever it wants called next.

    ``stop_reason`` is normalized to ``"tool_use"`` (call the tools and come
    back), ``"refusal"`` (the model declined — do not read ``text`` as an answer)
    or ``"end"``, so callers never branch on vendor strings. ``provider``/``model``
    report what ACTUALLY ran, which is how a silent fallback becomes visible
    instead of mysterious.
    """

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict[str, int]
    provider: str
    model: str
    # The provider's own content blocks, kept so the turn can be replayed
    # EXACTLY as it was received. On Anthropic that includes thinking blocks,
    # which must go back unchanged -- dropping or rebuilding them breaks the
    # turn and can trigger ordering/signature 400s. Opaque: never inspected,
    # never edited, and ignored by providers that do not need it.
    raw_content: Any = None


def to_openai_tools(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral declarations to OpenAI ``tools=`` entries."""
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for d in declarations
    ]


def to_anthropic_tools(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral declarations to Anthropic ``tools=`` entries.

    Deliberately does NOT set ``strict: true``. Strict schemas require every
    property to be listed in ``required`` alongside ``additionalProperties:
    false``, and the canvas declarations are mostly optional parameters
    (``create_block`` alone has a dozen type-specific seeds). Turning it on would
    mean rewriting every schema; until then the loop parses arguments
    defensively via ``_loads_args`` instead.
    """
    return [
        {
            "name": d["name"],
            "description": d.get("description", ""),
            "input_schema": d.get("parameters") or {"type": "object", "properties": {}},
        }
        for d in declarations
    ]


def _loads_args(raw: Any) -> dict[str, Any]:
    """Parse a tool-call argument blob, degrading to ``{}`` rather than raising.

    Always parsed, never string-matched: current models vary their JSON string
    escaping inside tool inputs. And a model that emits malformed arguments
    should get a tool error it can react to on the next step, not kill the loop.
    """
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral transcript to OpenAI ``messages`` (system is prepended by the caller)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": str(msg.get("content", "")),
            })
            continue
        if role == "assistant" and msg.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": msg.get("content") or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in msg["tool_calls"]
                ],
            })
            continue
        out.append({"role": role, "content": str(msg.get("content", ""))})
    return out


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral transcript to Anthropic ``messages``.

    Anthropic wants every tool_result for one assistant turn in a SINGLE user
    message, so consecutive neutral ``tool`` entries are coalesced. Emitting them
    one-per-message is rejected by the API, and splitting parallel results across
    messages teaches the model to stop making parallel calls — which is the whole
    reason this is not a straight map.

    A neutral ``{"role": "system"}`` entry passes straight through as a
    mid-conversation system message (supported on Opus 5 / Opus 4.8 with no beta
    header). It must follow a user message and be either last or followed by an
    assistant turn — the caller's job, since only the caller knows the order.
    """
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def _flush() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": str(msg.get("content", "")) or "(no output)",
                "is_error": bool(msg.get("is_error")),
            })
            continue
        _flush()
        if role == "assistant" and msg.get("raw_content"):
            # Verbatim replay. This is the ONLY correct way to send an Anthropic
            # assistant turn back: with adaptive thinking on (the default for the
            # Opus 5 family, which is what a block agent runs on) the response
            # carries thinking blocks that must return unchanged. Under
            # display:"omitted" their text is empty but they still carry
            # signatures, so they look safe to drop and are not -- rebuilding the
            # turn from text + tool_calls silently loses them.
            out.append({"role": "assistant", "content": msg["raw_content"]})
            continue
        if role == "assistant" and msg.get("tool_calls"):
            # Reconstructed: only for a turn that came from another provider, or
            # from a caller that kept no raw content.
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": str(msg["content"])})
            blocks.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in msg["tool_calls"]
            )
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": role, "content": str(msg.get("content", ""))})

    _flush()
    return out


async def _openai_tools_chat(
    system: str,
    messages: list[dict[str, Any]],
    declarations: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
) -> ToolTurn:
    client = get_async_openai()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *_to_openai_messages(messages)],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if declarations:
        kwargs["tools"] = to_openai_tools(declarations)
    resp = await client.chat.completions.create(**kwargs)

    message = resp.choices[0].message
    calls = [
        ToolCall(
            id=tc.id or f"call_{i}",
            name=tc.function.name,
            arguments=_loads_args(tc.function.arguments),
        )
        for i, tc in enumerate(getattr(message, "tool_calls", None) or [])
    ]
    usage = getattr(resp, "usage", None)
    return ToolTurn(
        text=message.content or "",
        tool_calls=calls,
        stop_reason="tool_use" if calls else "end",
        usage={
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        },
        provider="openai",
        model=model,
    )


async def _anthropic_tools_chat(
    system: str,
    messages: list[dict[str, Any]],
    declarations: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    effort: str,
) -> ToolTurn:
    """Anthropic tool-use turn.

    Four deliberate differences from the OpenAI branch:

    * **no sampling parameters.** ``temperature``/``top_p``/``top_k`` are not
      merely unused here — they are removed on Opus 5, Opus 4.8/4.7 and Sonnet 5
      and return a 400. Do not "restore" them for consistency with the OpenAI
      branch.
    * **adaptive thinking + effort.** Reasoning about a spec/RTL mismatch is
      exactly the workload that repays thinking, and ``budget_tokens`` is a 400
      on these models. Skipped entirely for models that predate adaptive
      thinking, which want the old form.
    * **the system block is cached** (``cache_control: ephemeral``). It is the
      frozen half of the prompt by construction — see the module note above.
    * **refusal is a normal outcome, not an exception.** A declined request comes
      back HTTP 200 with ``stop_reason == "refusal"``, and ``stop_details`` is
      populated only then, so it is checked before ``content`` is read.
    """
    client = get_async_anthropic()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": _to_anthropic_messages(messages),
    }
    if _supports_adaptive_thinking(model):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}
    if declarations:
        kwargs["tools"] = to_anthropic_tools(declarations)
    resp = await client.messages.create(**kwargs)

    usage = getattr(resp, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }

    if getattr(resp, "stop_reason", None) == "refusal":
        details = getattr(resp, "stop_details", None)
        reason = getattr(details, "explanation", "") or "the request was declined"
        log.warning(
            "llm_refusal",
            model=model,
            category=getattr(details, "category", None),
        )
        return ToolTurn(
            text=reason, tool_calls=[], stop_reason="refusal",
            usage=usage_dict, provider="anthropic", model=model,
        )

    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=_loads_args(block.input),
            ))

    return ToolTurn(
        text="".join(text_parts),
        tool_calls=calls,
        stop_reason="tool_use" if calls else "end",
        usage=usage_dict,
        provider="anthropic",
        model=model,
        raw_content=resp.content,
    )


async def call_llm_tools(
    system: str,
    messages: list[dict[str, Any]],
    declarations: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = _AGENT_MAX_TOKENS,
    temperature: float = 0.2,
    effort: str = "high",
) -> ToolTurn:
    """Provider-routed tool-calling turn — one step of an agent loop.

    Routing matches ``call_llm_text``: ``claude-*`` to Anthropic when a key is
    configured, everything else (including ``gemini-*``, whose native path is not
    implemented) to OpenAI. ``temperature`` applies to the OpenAI branch only and
    ``effort`` to the Anthropic one; each is ignored by the other rather than
    being an error, so a caller can switch provider between steps without
    rebuilding its arguments.

    The returned ``ToolTurn`` names the provider and model that actually ran, so
    a fallback is reported rather than hidden.
    """
    provider, resolved = resolve_provider(model)
    if _anthropic_available(provider):
        return await _anthropic_tools_chat(
            system, messages, declarations,
            model=resolved, max_tokens=max_tokens, effort=effort,
        )
    if provider == "gemini":
        log.info("llm_gemini_tools_unimplemented_fallback", requested=resolved)
    settings = get_settings()
    openai_model = resolved if provider == "openai" else settings.openai_model
    return await _openai_tools_chat(
        system, messages, declarations,
        model=openai_model, max_tokens=max_tokens, temperature=temperature,
    )
