"""Provider-agnostic tool calling (app.core.llm.call_llm_tools).

No real SDK is ever reached: each test installs a fake client that records the
kwargs it was handed, so the assertions are about the REQUEST we build as much as
the ToolTurn we parse. Several of them guard rules that fail loudly in
production and silently in review -- sampling parameters on models that reject
them, an unstable tool order that kills prompt caching, and a refusal read as if
it were an answer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from app.core import llm


# ── Fakes ────────────────────────────────────────────────────────────────────


# Kwargs the real ``AsyncMessages.create`` accepts, read off the installed SDK.
# A double looser than the thing it stands in for hides exactly the drift these
# tests exist to catch: the branch below builds `output_config`, which anthropic
# 0.71 does not accept and which a **kwargs-swallowing fake reports as a pass.
# The hard-coded set is the floor from requirements.txt, used only when the SDK
# is not installed at all.
_FALLBACK_CREATE_KWARGS = frozenset({
    "max_tokens", "messages", "model", "container", "context_management",
    "mcp_servers", "metadata", "output_config", "service_tier", "stop_sequences",
    "stream", "system", "temperature", "thinking", "tool_choice", "tools",
    "top_k", "top_p", "betas", "extra_headers", "extra_query", "extra_body",
    "timeout",
})


def _sdk_create_kwargs() -> frozenset[str]:
    try:
        import inspect

        from anthropic.resources.messages import AsyncMessages
    except Exception:                                   # SDK not installed
        return _FALLBACK_CREATE_KWARGS
    return frozenset(inspect.signature(AsyncMessages.create).parameters) - {"self"}


class _FakeAnthropic:
    """A double as strict as the SDK: unknown kwargs raise, as they really do.

    ``error`` is raised AFTER the call is recorded, so a test can count attempts
    (what the auth latch is about) and not merely successes.
    """

    def __init__(self, response=None, error: BaseException | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        unknown = set(kwargs) - _sdk_create_kwargs()
        if unknown:
            raise TypeError(
                f"create() got an unexpected keyword argument {sorted(unknown)!r} "
                f"-- the installed anthropic SDK would reject this request"
            )
        for required in ("model", "max_tokens", "messages"):
            if required not in kwargs:
                raise TypeError(f"create() missing required argument: {required!r}")
        if self._error is not None:
            raise self._error
        return self._response


def _auth_error(status: int = 401):
    """An exception shaped like anthropic.AuthenticationError, without the SDK."""
    return type("FakeAPIStatusError", (Exception,), {"status_code": status})(
        f"Error code: {status} - invalid x-api-key"
    )


class _FakeOpenAI:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _anthropic_response(*, text="", tool_uses=(), stop_reason="end_turn", stop_details=None):
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    for tid, name, payload in tool_uses:
        content.append(SimpleNamespace(type="tool_use", id=tid, name=name, input=payload))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=7,
            cache_creation_input_tokens=0,
        ),
    )


def _openai_response(*, text=None, tool_calls=()):
    calls = [
        SimpleNamespace(
            id=tid,
            function=SimpleNamespace(name=name, arguments=args),
        )
        for tid, name, args in tool_calls
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=calls or None),
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
    )


def _settings(*, anthropic_key="ak", openai_key="ok", openai_model="gpt-4o"):
    return SimpleNamespace(
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
        openai_model=openai_model,
    )


def _install(monkeypatch, *, anthropic=None, openai=None, settings=None):
    monkeypatch.setattr(llm, "get_settings", lambda: settings or _settings())
    if anthropic is not None:
        monkeypatch.setattr(llm, "get_async_anthropic", lambda: anthropic)
    if openai is not None:
        monkeypatch.setattr(llm, "get_async_openai", lambda: openai)



def _raise_no_key():
    raise ValueError("ANTHROPIC_API_KEY not configured")


def _raise_no_openai_key():
    raise ValueError("OPENAI_API_KEY not configured")


_TOOLS = [{
    "name": "read_port_value",
    "description": "Read a port.",
    "parameters": {
        "type": "object",
        "properties": {"port_name": {"type": "string", "description": "Port."}},
        "required": ["port_name"],
    },
}]


# ── Schema conversion ────────────────────────────────────────────────────────


def test_to_openai_tools_shape():
    [tool] = llm.to_openai_tools(_TOOLS)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "read_port_value"
    assert tool["function"]["parameters"]["required"] == ["port_name"]


def test_to_anthropic_tools_shape():
    [tool] = llm.to_anthropic_tools(_TOOLS)
    assert tool["name"] == "read_port_value"
    assert tool["input_schema"]["properties"]["port_name"]["type"] == "string"
    # strict is deliberately absent -- it would require every property to be
    # listed in `required`, which the canvas declarations do not do.
    assert "strict" not in tool


def test_converters_tolerate_a_declaration_with_no_parameters():
    decl = [{"name": "finish", "description": "Stop."}]
    assert llm.to_openai_tools(decl)[0]["function"]["parameters"]["type"] == "object"
    assert llm.to_anthropic_tools(decl)[0]["input_schema"]["type"] == "object"


# ── Transcript translation ───────────────────────────────────────────────────


def test_anthropic_coalesces_consecutive_tool_results():
    """Parallel tool results must arrive as ONE user message.

    Splitting them is rejected by the API and teaches the model to stop making
    parallel calls, so this is the translation's whole reason for existing.
    """
    calls = [llm.ToolCall("t1", "a", {}), llm.ToolCall("t2", "b", {})]
    out = llm._to_anthropic_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "working", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "t1", "content": "one"},
        {"role": "tool", "tool_call_id": "t2", "content": "two", "is_error": True},
    ])
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    results = out[-1]["content"]
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    # The assistant turn keeps its text block ahead of the tool_use blocks.
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]


def test_anthropic_empty_tool_result_is_never_blank():
    [msg] = llm._to_anthropic_messages([{"role": "tool", "tool_call_id": "t", "content": ""}])
    assert msg["content"][0]["content"] == "(no output)"


def test_anthropic_passes_mid_conversation_system_through():
    out = llm._to_anthropic_messages([
        {"role": "user", "content": "go"},
        {"role": "system", "content": "CANVAS: ..."},
    ])
    assert out[-1] == {"role": "system", "content": "CANVAS: ..."}


def test_openai_serializes_tool_calls_and_results():
    calls = [llm.ToolCall("t1", "read_port_value", {"port_name": "code"})]
    out = llm._to_openai_messages([
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "t1", "content": "module x;"},
    ])
    assert out[0]["content"] is None          # empty text must be null, not ""
    fn = out[0]["tool_calls"][0]["function"]
    assert json.loads(fn["arguments"]) == {"port_name": "code"}
    assert out[1] == {"role": "tool", "tool_call_id": "t1", "content": "module x;"}


def test_loads_args_degrades_instead_of_raising():
    assert llm._loads_args("{not json") == {}
    assert llm._loads_args("[1,2]") == {}
    assert llm._loads_args(None) == {}
    assert llm._loads_args({"a": 1}) == {"a": 1}


# ── Anthropic branch ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_sends_no_sampling_parameters(monkeypatch):
    """temperature/top_p/top_k are REMOVED on these models and return a 400."""
    fake = _FakeAnthropic(_anthropic_response(text="hi"))
    _install(monkeypatch, anthropic=fake)

    await llm.call_llm_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS,
                             model="claude-opus-5", temperature=0.7)

    sent = fake.calls[0]
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "top_k" not in sent


@pytest.mark.asyncio
async def test_anthropic_caches_the_system_block(monkeypatch):
    fake = _FakeAnthropic(_anthropic_response(text="hi"))
    _install(monkeypatch, anthropic=fake)

    await llm.call_llm_tools("frozen prompt", [], _TOOLS, model="claude-opus-5")

    [block] = fake.calls[0]["system"]
    assert block["text"] == "frozen prompt"
    assert block["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_anthropic_adaptive_thinking_is_model_gated(monkeypatch):
    """Adaptive thinking + effort on models that take it, neither on Haiku."""
    fake = _FakeAnthropic(_anthropic_response(text="hi"))
    _install(monkeypatch, anthropic=fake)

    await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5", effort="xhigh")
    assert fake.calls[-1]["thinking"] == {"type": "adaptive"}
    assert fake.calls[-1]["output_config"] == {"effort": "xhigh"}

    await llm.call_llm_tools("s", [], _TOOLS, model="claude-haiku-4-5")
    assert "thinking" not in fake.calls[-1]
    assert "output_config" not in fake.calls[-1]
    # budget_tokens is never sent either -- it is a 400 on the adaptive models
    # and this path simply does not configure thinking for the older ones.
    assert "budget_tokens" not in json.dumps(fake.calls[-1])


@pytest.mark.asyncio
async def test_anthropic_parses_tool_uses(monkeypatch):
    fake = _FakeAnthropic(_anthropic_response(
        text="I will read it.",
        tool_uses=[("tu_1", "read_port_value", {"port_name": "code"})],
    ))
    _install(monkeypatch, anthropic=fake)

    turn = await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert turn.stop_reason == "tool_use"
    assert turn.text == "I will read it."
    assert turn.tool_calls == [llm.ToolCall("tu_1", "read_port_value", {"port_name": "code"})]
    assert turn.usage["cache_read_tokens"] == 7
    assert (turn.provider, turn.model) == ("anthropic", "claude-opus-5")


@pytest.mark.asyncio
async def test_anthropic_refusal_is_reported_not_read_as_an_answer(monkeypatch):
    fake = _FakeAnthropic(_anthropic_response(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="declined"),
    ))
    _install(monkeypatch, anthropic=fake)

    turn = await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert turn.stop_reason == "refusal"
    assert turn.tool_calls == []
    assert "declined" in turn.text


# ── Routing ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gemini_falls_back_to_openai_and_says_so(monkeypatch):
    """The Agent dropdown's old default was gemini; the fallback must be visible."""
    fake = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, openai=fake)

    turn = await llm.call_llm_tools("s", [], _TOOLS, model="gemini-2.5-flash")

    assert turn.provider == "openai"
    assert turn.model == "gpt-4o"          # the effective model, not the request
    assert fake.calls[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_claude_without_a_key_falls_back_to_openai(monkeypatch):
    fake = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, openai=fake, settings=_settings(anthropic_key=""))

    turn = await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert (turn.provider, turn.model) == ("openai", "gpt-4o")


# ── A rejected key ───────────────────────────────────────────────────────────
#
# A key that is PRESENT but wrong used to be strictly worse than no key at all:
# the missing-key gate never fired, the 401 raised out of the provider branch,
# and a block agent died on step 1 of 24 with the raw JSON error in its chat.


@pytest.mark.asyncio
async def test_a_rejected_anthropic_key_falls_back_to_openai(monkeypatch):
    anthropic = _FakeAnthropic(error=_auth_error(401))
    openai = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, anthropic=anthropic, openai=openai)

    turn = await llm.call_llm_tools("s", [{"role": "user", "content": "go"}], _TOOLS,
                                    model="claude-opus-5")

    assert (turn.provider, turn.model) == ("openai", "gpt-4o")
    # The transcript reached the fallback translated, not dropped.
    assert openai.calls[0]["messages"][-1] == {"role": "user", "content": "go"}


@pytest.mark.asyncio
async def test_the_fallback_latches_so_a_run_pays_the_401_once(monkeypatch):
    """24 steps must not mean 24 doomed round-trips."""
    anthropic = _FakeAnthropic(error=_auth_error(401))
    openai = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, anthropic=anthropic, openai=openai)

    for _ in range(3):
        await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert len(anthropic.calls) == 1
    assert len(openai.calls) == 3


@pytest.mark.asyncio
async def test_a_403_latches_too(monkeypatch):
    anthropic = _FakeAnthropic(error=_auth_error(403))
    openai = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, anthropic=anthropic, openai=openai)

    turn = await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert turn.provider == "openai"
    assert not llm._anthropic_available("anthropic")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 429, 500, 529])
async def test_a_non_auth_failure_is_neither_hidden_nor_latched(monkeypatch, status):
    """A contract break (400) or an overload (429/5xx) must not become a GPT answer.

    Answering a 400 from another vendor is how an API-contract break ships;
    latching on a transient blip would downgrade the whole process for the
    cooldown over one bad minute.
    """
    anthropic = _FakeAnthropic(error=_auth_error(status))
    openai = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, anthropic=anthropic, openai=openai)

    with pytest.raises(Exception) as caught:
        await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert getattr(caught.value, "status_code", None) == status
    assert openai.calls == []
    assert llm._anthropic_available("anthropic")          # latch still clear


@pytest.mark.asyncio
async def test_a_client_that_cannot_be_built_takes_the_same_fallback(monkeypatch):
    """get_async_anthropic raises for an unconfigured key; that is a fallback, not a 500."""
    openai = _FakeOpenAI(_openai_response(text="hello"))
    _install(monkeypatch, openai=openai)
    monkeypatch.setattr(llm, "get_async_anthropic", _raise_no_key)

    turn = await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert turn.provider == "openai"


@pytest.mark.asyncio
async def test_with_no_provider_left_the_error_says_which_key_to_set(monkeypatch):
    anthropic = _FakeAnthropic(error=_auth_error(401))
    _install(monkeypatch, anthropic=anthropic,
             settings=_settings(openai_key=""))
    monkeypatch.setattr(llm, "get_async_openai", _raise_no_openai_key)

    with pytest.raises(RuntimeError) as caught:
        await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    message = str(caught.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "grafux-orchestrator" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tools", "text", "json"])
async def test_every_entry_point_shares_the_fallback(monkeypatch, entry):
    anthropic = _FakeAnthropic(error=_auth_error(401))
    openai = _FakeOpenAI(_openai_response(text='{"ok": true}'))
    _install(monkeypatch, anthropic=anthropic, openai=openai)

    if entry == "tools":
        result = await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")
        assert result.provider == "openai"
    elif entry == "text":
        assert await llm.call_llm_text("s", "u", model="claude-opus-5") == '{"ok": true}'
    else:
        assert await llm.call_llm_json("s", "u", model="claude-opus-5") == {"ok": True}

    assert len(anthropic.calls) == 1
    assert len(openai.calls) == 1


@pytest.mark.asyncio
async def test_a_bad_json_answer_is_not_mistaken_for_an_auth_failure(monkeypatch):
    """call_llm_json parses INSIDE the attempt, so a parse error still propagates."""
    anthropic = _FakeAnthropic(_anthropic_response(text="not json"))
    openai = _FakeOpenAI(_openai_response(text='{"ok": true}'))
    _install(monkeypatch, anthropic=anthropic, openai=openai)

    with pytest.raises(ValueError):
        await llm.call_llm_json("s", "u", model="claude-opus-5")
    assert openai.calls == []


def test_make_chat_model_honours_the_latch(monkeypatch):
    """Once this process has seen the key rejected, LangChain callers stop choosing it."""
    _install(monkeypatch)
    llm._disable_anthropic("401")
    sentinel = object()
    monkeypatch.setitem(
        __import__("sys").modules, "langchain_openai",
        SimpleNamespace(ChatOpenAI=lambda **kw: sentinel),
    )

    assert llm.make_chat_model("claude-opus-5") is sentinel


@pytest.mark.asyncio
async def test_the_fake_rejects_a_kwarg_the_real_sdk_would_reject():
    """Meta-test: the double's strictness IS the guard, so pin it."""
    with pytest.raises(TypeError, match="not_a_real_parameter"):
        await _FakeAnthropic()._create(
            model="m", max_tokens=1, messages=[], not_a_real_parameter=1,
        )


@pytest.mark.asyncio
async def test_the_request_only_uses_kwargs_the_installed_sdk_accepts(monkeypatch):
    """The guard behind the anthropic pin in requirements.txt.

    An SDK too old for output_config raises TypeError before any HTTP call --
    the same "stopped after an internal error" symptom as a bad key, by a
    completely different route.
    """
    import inspect

    pytest.importorskip("anthropic")
    from anthropic.resources.messages import AsyncMessages

    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    fake = _FakeAnthropic(_anthropic_response(text="hi"))
    _install(monkeypatch, anthropic=fake)

    await llm.call_llm_tools("s", [], _TOOLS, model="claude-opus-5")

    assert set(fake.calls[0]) <= accepted


@pytest.mark.asyncio
async def test_openai_branch_parses_tool_calls_and_keeps_temperature(monkeypatch):
    fake = _FakeOpenAI(_openai_response(
        tool_calls=[("call_1", "read_port_value", '{"port_name": "code"}')],
    ))
    _install(monkeypatch, openai=fake)

    turn = await llm.call_llm_tools("s", [{"role": "user", "content": "go"}], _TOOLS,
                                    model="gpt-5", temperature=0.4)

    assert fake.calls[0]["temperature"] == 0.4
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": "s"}
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].arguments == {"port_name": "code"}


@pytest.mark.asyncio
async def test_a_transcript_survives_switching_provider_mid_run(monkeypatch):
    """The per-block model dropdown can change between steps.

    The neutral transcript is what makes that safe: the same history must be
    translatable by either branch without being rebuilt.
    """
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [llm.ToolCall("t1", "read_port_value", {"port_name": "code"})]},
        {"role": "tool", "tool_call_id": "t1", "content": "module x;"},
    ]
    anth = _FakeAnthropic(_anthropic_response(text="done"))
    oai = _FakeOpenAI(_openai_response(text="done"))
    _install(monkeypatch, anthropic=anth, openai=oai)

    await llm.call_llm_tools("s", history, _TOOLS, model="claude-opus-5")
    await llm.call_llm_tools("s", history, _TOOLS, model="gpt-5")

    assert [m["role"] for m in anth.calls[0]["messages"]] == ["user", "assistant", "user"]
    assert [m["role"] for m in oai.calls[0]["messages"]] == ["system", "user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_no_tools_means_no_tools_key(monkeypatch):
    """A final summary step runs without tools; an empty list must not be sent."""
    fake = _FakeAnthropic(_anthropic_response(text="summary"))
    _install(monkeypatch, anthropic=fake)

    await llm.call_llm_tools("s", [], [], model="claude-opus-5")

    assert "tools" not in fake.calls[0]


# ── Thinking blocks ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_anthropic_turn_is_replayed_verbatim(monkeypatch):
    """Thinking blocks must go back UNCHANGED, so the turn is replayed as-is.

    With adaptive thinking on -- the default for the Opus 5 family, which is what
    a block agent runs on -- the response carries thinking blocks alongside the
    tool_use. Rebuilding the assistant turn from text + tool_calls drops them,
    and the API is explicit that dropping or editing them breaks the turn and can
    trigger ordering/signature 400s. Under display:"omitted" their text is empty
    but they still carry signatures, so they look safe to drop and are not.
    """
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig-abc")
    fake = _FakeAnthropic(SimpleNamespace(
        content=[thinking,
                 SimpleNamespace(type="text", text="reading it"),
                 SimpleNamespace(type="tool_use", id="tu_1", name="read_port_value",
                                 input={"port_name": "code"})],
        stop_reason="tool_use", stop_details=None,
        usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    ))
    _install(monkeypatch, anthropic=fake)

    turn = await llm.call_llm_tools("s", [{"role": "user", "content": "go"}], _TOOLS,
                                    model="claude-opus-5")
    assert turn.raw_content is not None

    # Replay it the way the agent loop does.
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls,
         "raw_content": turn.raw_content},
        {"role": "tool", "tool_call_id": "tu_1", "content": "module x;"},
    ]
    await llm.call_llm_tools("s", history, _TOOLS, model="claude-opus-5")

    assistant = fake.calls[-1]["messages"][1]
    assert assistant["content"] is turn.raw_content, "the turn was rebuilt, not replayed"
    assert thinking in assistant["content"], "the thinking block was dropped"


def test_a_turn_without_raw_content_is_still_reconstructed():
    """A turn from another provider has no native content; rebuild it."""
    out = llm._to_anthropic_messages([
        {"role": "assistant", "content": "hi",
         "tool_calls": [llm.ToolCall("t1", "run_block", {})]},
    ])
    assert [b["type"] for b in out[0]["content"]] == ["text", "tool_use"]
