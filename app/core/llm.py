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
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("core.llm")

_DEFAULT_MAX_TOKENS = 4096

# Shared AsyncOpenAI clients keyed by the running event loop. Like the HTTP client
# pool, an SDK client binds to the loop it was created on, so the web process gets
# one reused client while Celery's per-task loops each get their own.
_openai_clients: dict[int, tuple[asyncio.AbstractEventLoop, Any]] = {}


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

    loop = asyncio.get_running_loop()
    entry = _openai_clients.get(id(loop))
    if entry is not None and entry[0] is loop:
        return entry[1]
    for key, (cached_loop, _client) in list(_openai_clients.items()):
        if cached_loop.is_closed():
            _openai_clients.pop(key, None)
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    _openai_clients[id(loop)] = (loop, client)
    return client


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
    settings = get_settings()
    from anthropic import AsyncAnthropic

    system = system_prompt
    if want_json:
        system = (
            system_prompt
            + "\n\nReturn ONLY a single valid JSON object. "
            "No markdown, no code fences, no prose."
        )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
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
