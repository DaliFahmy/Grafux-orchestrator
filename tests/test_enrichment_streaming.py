from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from app.core import llm
from app.modules.session import enrichment


# ── enrich_actions_streaming ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_yields_per_action_as_ready(monkeypatch):
    """A fast enricher's result is yielded before a slow one finishes (not batched)."""

    async def slow(action, session_id):
        await asyncio.sleep(0.05)
        action["output_ports"] = [{"port_name": "p"}]
        return ("ok", "")

    async def fast(action, session_id):
        action["code"] = "x = 1"
        return ("ok", "")

    monkeypatch.setitem(enrichment._ENRICHERS, "topics", slow)
    monkeypatch.setitem(enrichment._ENRICHERS, "tools", fast)

    actions = [
        {"type": "create_block", "block_type": "topics", "block_name": "slow"},
        {"type": "create_block", "block_type": "tools", "block_name": "fast"},
    ]
    order = []
    async for action, status, _detail in enrichment.enrich_actions_streaming(actions, "sid"):
        order.append(action["block_name"])
        assert status == "ok"

    assert order[0] == "fast"  # fast completed and was yielded first
    assert set(order) == {"fast", "slow"}


@pytest.mark.asyncio
async def test_streaming_reports_failure_reason(monkeypatch):
    """A soft-fail enricher (no content) surfaces status=failed with a reason; a raise too."""

    async def empty(action, session_id):
        return ("failed", "AI not configured")

    async def boom(action, session_id):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(enrichment._ENRICHERS, "topics", empty)
    monkeypatch.setitem(enrichment._ENRICHERS, "code", boom)

    actions = [
        {"type": "create_block", "block_type": "topics", "block_name": "t"},
        {"type": "create_block", "block_type": "code", "block_name": "c"},
    ]
    results = {
        action["block_name"]: (status, detail)
        async for action, status, detail in enrichment.enrich_actions_streaming(actions, "sid")
    }
    assert results["t"] == ("failed", "AI not configured")
    assert results["c"][0] == "failed" and "kaboom" in results["c"][1]


@pytest.mark.asyncio
async def test_streaming_skips_non_enrichable(monkeypatch):
    """Non-create actions and unknown block types are not dispatched."""

    async def track(action, session_id):
        return ("ok", "")

    monkeypatch.setitem(enrichment._ENRICHERS, "tools", track)
    actions = [
        {"type": "set_port", "block_type": "tools", "block_name": "skip-set"},
        {"type": "create_block", "block_type": "unknown_kind", "block_name": "skip-unknown"},
        {"type": "create_block", "block_type": "tools", "block_name": "do-me"},
    ]
    names = [a["block_name"] async for a, _s, _d in enrichment.enrich_actions_streaming(actions, "sid")]
    assert names == ["do-me"]


def test_is_enrichable():
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "tools"})
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "image"})
    assert not enrichment.is_enrichable({"type": "create_block", "block_type": "nope"})
    assert not enrichment.is_enrichable({"type": "set_port", "block_type": "tools"})
    assert not enrichment.is_enrichable("not a dict")


# ── tool-code generation cache ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_tool_code_caches_identical_requests(monkeypatch):
    from app.modules.blocks import router as blocks_router

    calls = {"n": 0}

    async def fake_text(system_prompt, user_message, temperature=0.2, *, model=None, max_tokens=4096):
        calls["n"] += 1
        return "import os\n# generated"

    monkeypatch.setattr(blocks_router, "_call_openai_text", fake_text)
    monkeypatch.setattr(blocks_router, "get_system_prompt", lambda _s: "PROMPT")
    blocks_router._GEN_CACHE.clear()

    a = await blocks_router.generate_tool_code("Tool: adder\nadd two ints")
    b = await blocks_router.generate_tool_code("Tool: adder\nadd two ints")
    assert a == b == "import os\n# generated"
    assert calls["n"] == 1  # second call served from cache

    # A different requirement is a cache miss.
    await blocks_router.generate_tool_code("Tool: subber\nsubtract")
    assert calls["n"] == 2


# ── client pooling ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_client_pooled_within_loop(monkeypatch):
    created = []

    class FakeAnthropic:
        def __init__(self, **kwargs):
            created.append(kwargs)

    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropic)
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(anthropic_api_key="sk-a"))
    llm._anthropic_clients.clear()

    a = llm.get_async_anthropic()
    b = llm.get_async_anthropic()
    assert a is b  # same instance reused within one loop
    assert len(created) == 1


@pytest.mark.asyncio
async def test_gemini_client_pooled_within_loop(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    from google import genai as google_genai

    monkeypatch.setattr(google_genai, "Client", FakeClient)
    llm._gemini_clients.clear()

    a = llm.get_gemini_client("k")
    b = llm.get_gemini_client("k")
    assert a is b
    assert len(created) == 1
