"""Tests for the code_fix BLOCK type: general-purpose code repair in any language.

Named ``..._block`` because ``test_blocks_code_fix.py`` is already taken by the
``code`` block's ``[fix_rtl]`` repair MODE, which is a different thing that happens
to share the word: that one repairs HDL and keeps the module interface
byte-identical, this one repairs a program in any language and freezes nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.core.constants import BLOCK_TYPE_SECTION, BlockType
from app.modules.blocks import directives as D
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import (
    CodeFixGenerateRequest as FixRequest,  # alias: pytest must not collect it
)
from app.modules.session import enrichment
from app.prompts import get_system_prompt

BROKEN = """def read_rows(path):
    with open(path) as fh:
        return fh.read().strip().split("\\n")
"""

FIXED = """def read_rows(path):
    with open(path) as fh:
        body = fh.read().strip()
    if not body:
        return []
    return body.split("\\n")
"""

# What lands on the port: _strip_code_fences strips the model's answer, the same as
# it does for the code and code_hdl blocks, so the trailing newline does not survive
# the round trip. Named rather than inlined so the four assertions that depend on it
# say which behaviour they are pinning.
FIXED_ON_PORT = FIXED.strip()

ORDER = "it returns [''] for an empty file; return an empty list instead"

CODE_FIX_IN = ["block_description", "code", "fix", "language"]
CODE_FIX_OUT = [
    "code", "code_change", "explanation", "improvements", "status", "errors",
]


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(
        openai_api_key=openai, anthropic_api_key=anthropic, openai_model="gpt-test"
    )


def _ports(params, side):
    return {p["port_name"]: p for p in params[f"{side}_ports"]}


def _responder(*payloads):
    """A fake LLM returning each payload in turn, recording what it was asked."""
    calls = []

    async def fake_llm(
        system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096
    ):
        calls.append((system_prompt, user_message))
        return payloads[min(len(calls) - 1, len(payloads) - 1)]

    return fake_llm, calls


def _answer(code=FIXED, **over):
    payload = {
        "code": code,
        "explanation": "Root cause: str.split on an empty string yields [''].",
        "improvements": "",
        "language": "python",
        "compliance": f"{ORDER} -> APPLIED: the early return",
    }
    payload.update(over)
    return payload


# ── The diff is derived, not trusted ──────────────────────────────────────────


def test_code_change_is_a_real_unified_diff_of_the_two_versions():
    diff = blocks_router.code_change_diff("a = 1\nb = 2\n", "a = 1\nb = 3\n")
    assert diff.startswith("--- before")
    assert "+++ after" in diff
    assert "-b = 2" in diff
    assert "+b = 3" in diff
    # Only the line that changed is marked.
    assert "-a = 1" not in diff


def test_an_unchanged_program_produces_an_empty_diff_not_an_empty_hunk():
    assert blocks_router.code_change_diff("x = 1\n", "x = 1\n") == ""
    assert blocks_router.code_change_diff("", "x = 1\n") == ""
    assert blocks_router.code_change_diff("x = 1\n", "") == ""


def test_a_huge_diff_is_capped_and_says_so():
    before = "\n".join(f"line_{i} = {i}" for i in range(4000))
    after = "\n".join(f"line_{i} = {i + 1}" for i in range(4000))
    diff = blocks_router.code_change_diff(before, after)
    assert len(diff) <= blocks_router._CODE_CHANGE_MAX_CHARS + 200
    assert diff.endswith(blocks_router._CODE_CHANGE_TRUNCATED)


# ── Shape problems: the two failures that look like success ───────────────────


def test_an_empty_answer_is_a_problem():
    assert blocks_router.code_fix_shape_problems("", BROKEN)
    assert blocks_router.code_fix_shape_problems("   \n", BROKEN)


def test_an_unchanged_answer_is_a_problem_because_the_fix_was_not_applied():
    problems = blocks_router.code_fix_shape_problems(BROKEN, BROKEN)
    assert problems and "identical" in problems[0]
    # Whitespace-only difference still counts as unchanged.
    assert blocks_router.code_fix_shape_problems("\n" + BROKEN + "  ", BROKEN)


def test_a_real_change_has_no_shape_problems():
    assert blocks_router.code_fix_shape_problems(FIXED, BROKEN) == []


# ── Port contract (must stay byte-identical to the Qt dialog and executor) ────


@pytest.mark.asyncio
async def test_scaffold_code_fix_exact_ports_and_seeds():
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_fix",
        block_name="fix row reader",
        description="mend the csv reader",
        seeds={"code": BROKEN, "fix": ORDER, "language": "python"},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "code_fix"
    assert params["name"] == "fix_row_reader"
    assert [p["port_name"] for p in params["input_ports"]] == CODE_FIX_IN
    assert [p["port_name"] for p in params["output_ports"]] == CODE_FIX_OUT
    ins = _ports(params, "input")
    assert ins["code"]["port_content"] == BROKEN.strip()
    assert ins["fix"]["port_content"] == ORDER
    assert ins["language"]["port_content"] == "python"
    assert ins["fix"]["port_path"] == "data/code_fix/general/fix_row_reader/inputs/fix.txt"


@pytest.mark.asyncio
async def test_the_fix_port_falls_back_to_the_description():
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_fix", block_name="f", description="make it retry three times",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["fix"]["port_content"] == "make it retry three times"


@pytest.mark.asyncio
async def test_language_has_no_default_so_it_can_be_inferred_from_the_code():
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_fix", block_name="f", description="d",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["language"]["port_content"] == ""


@pytest.mark.asyncio
async def test_code_is_on_both_sides_and_is_a_deliberate_exception():
    """The input is the program to repair; the output is that program repaired.

    Pinned because the rule everywhere else (see the ``collect_coverage`` note in
    edaports.h) is that a name in both lists wrongly claims it is echoed through.
    """
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_fix", block_name="f",
    )
    params = result["tool_calls"][0]["params"]
    assert "code" in _ports(params, "input")
    assert "code" in _ports(params, "output")
    # ...and the two live in different files, which is what makes it safe.
    assert _ports(params, "input")["code"]["port_path"].endswith("/inputs/code.txt")
    assert _ports(params, "output")["code"]["port_path"].endswith("/outputs/code.txt")


@pytest.mark.asyncio
async def test_it_has_no_feedback_port_because_fix_is_that_port():
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_fix", block_name="f",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert "feedback" not in ins
    assert "fix" in ins


@pytest.mark.asyncio
async def test_a_verilator_repair_order_can_be_wired_into_it():
    fix_block = await blocks_router.generate_scaffold_payload(
        block_type="code_fix", block_name="f")
    ver = await blocks_router.generate_scaffold_payload(
        block_type="verilator", block_name="v")
    code_block = await blocks_router.generate_scaffold_payload(
        block_type="code", block_name="c") or {"tool_calls": [{"params": {}}]}
    ver_out = set(_ports(ver["tool_calls"][0]["params"], "output"))
    fix_in = set(_ports(fix_block["tool_calls"][0]["params"], "input"))
    # verilator.fix_rtl -> code_fix.fix, and code_hdl.code -> code_fix.code
    assert "fix_rtl" in ver_out
    assert {"fix", "code"} <= fix_in
    assert code_block is not None


# ── AI generation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_fills_every_port_and_derives_the_diff(monkeypatch):
    fake, calls = _responder(_answer())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_fix_payload(
        block_name="fix_reader", code=BROKEN, fix=ORDER, language="python",
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "code_fix"
    assert [p["port_name"] for p in params["output_ports"]] == CODE_FIX_OUT
    outs = _ports(params, "output")
    assert outs["code"]["port_content"] == FIXED_ON_PORT
    assert outs["explanation"]["port_content"].startswith("Root cause:")
    assert outs["status"]["port_content"] == "ok"
    assert outs["errors"]["port_content"] == ""
    # The diff is computed here, and matches the two versions it claims to describe.
    diff = outs["code_change"]["port_content"]
    assert "+    if not body:" in diff
    assert diff.startswith("--- before")
    # The inputs are echoed back so the block shows what it was asked.
    ins = _ports(params, "input")
    assert ins["code"]["port_content"] == BROKEN
    assert ins["fix"]["port_content"] == ORDER
    # One call: nothing was rejected.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_prompt_carries_the_code_the_order_and_the_instructions(monkeypatch):
    fake, calls = _responder(_answer())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix="use `pathlib`", language="python",
    )
    _, message = calls[0]
    assert BROKEN.strip() in message
    assert "use `pathlib`" in message
    assert "Programming language: python" in message
    # `fix` is mined into a mandatory instruction, not just pasted.
    assert D._INSTRUCTION_HEADER.splitlines()[0] in message


@pytest.mark.asyncio
async def test_a_blank_language_asks_the_model_to_infer_it(monkeypatch):
    fake, calls = _responder(_answer())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix=ORDER,
    )
    _, message = calls[0]
    assert "infer it from the code" in message
    # ...and nothing is invented on the port.
    assert _ports(result["tool_calls"][0]["params"], "input")["language"]["port_content"] == ""


@pytest.mark.asyncio
async def test_an_unchanged_answer_costs_a_repair_round(monkeypatch):
    # First the model returns the input untouched, then it actually changes it.
    fake, calls = _responder(_answer(code=BROKEN), _answer())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix=ORDER, language="python",
    )
    assert len(calls) == 2
    assert "REJECTED" in calls[1][1]
    assert _ports(result["tool_calls"][0]["params"], "output")["code"]["port_content"] == FIXED_ON_PORT


@pytest.mark.asyncio
async def test_it_never_blanks_the_program_it_was_given(monkeypatch):
    fake, _ = _responder(_answer(code=""))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix=ORDER, language="python",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    # The app writes every port it is handed, so an empty `code` here is data loss.
    assert outs["code"]["port_content"] == BROKEN
    assert outs["code_change"]["port_content"] == ""
    assert outs["status"]["port_content"] == "needs_review"


@pytest.mark.asyncio
async def test_an_unmet_instruction_is_reported_in_the_users_own_words(monkeypatch):
    # The model changes the code but ignores the literal it was told to include.
    fake, calls = _responder(_answer(compliance=""))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix="set MAX_ROWS to 500", language="python",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    improvements = outs["improvements"]["port_content"]
    assert D.UNMET_HEADING.splitlines()[0] in improvements
    assert "MAX_ROWS" in improvements
    # It cost the repair rounds before being reported rather than failing the run.
    assert len(calls) == blocks_router._CODE_FIX_BLOCK_REPAIR_ROUNDS + 1
    # ...and the repair is still written out.
    assert outs["code"]["port_content"] == FIXED_ON_PORT


@pytest.mark.asyncio
async def test_a_prohibition_met_only_in_a_comment_is_not_a_violation(monkeypatch):
    """The language-aware stripper's reason for existing.

    Without it a Python program whose comment says "no longer calls eval" reads as
    still calling it -- a FORBIDDEN directive is binding, so that would burn every
    repair round and end in a false failure.
    """
    answer = _answer(
        code='# we no longer call eval here\nrows = read_rows("x")\n',
        compliance="Do not use `eval`. -> APPLIED: the call is gone",
    )
    fake, calls = _responder(answer)
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix="Do not use `eval`.", language="python",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_both_halves_are_required(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    fake, _ = _responder(_answer())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    with pytest.raises(ValueError, match="code"):
        await blocks_router.generate_code_fix_payload(
            block_name="f", code="  ", fix=ORDER)
    with pytest.raises(ValueError, match="fix"):
        await blocks_router.generate_code_fix_payload(
            block_name="f", code=BROKEN, fix="")


@pytest.mark.asyncio
async def test_no_key_returns_none_so_callers_can_fall_back(monkeypatch):
    monkeypatch.setattr(
        blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic=""))
    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix=ORDER)
    assert result is None


@pytest.mark.asyncio
async def test_an_anthropic_only_deployment_still_generates(monkeypatch):
    fake, calls = _responder(_answer())
    monkeypatch.setattr(
        blocks_router, "get_settings",
        lambda: _fake_settings(openai="", anthropic="sk-ant-test"))
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_code_fix_payload(
        block_name="f", code=BROKEN, fix=ORDER)
    assert result is not None
    assert len(calls) == 1


# ── The no-key / failure stub ─────────────────────────────────────────────────


def test_the_stub_is_port_complete_and_echoes_the_program():
    body = FixRequest(block_name="f", code=BROKEN, fix=ORDER, language="python")
    params = blocks_router._simple_code_fix_block_response(
        body, error="AI not configured")["tool_calls"][0]["params"]
    assert [p["port_name"] for p in params["input_ports"]] == CODE_FIX_IN
    assert [p["port_name"] for p in params["output_ports"]] == CODE_FIX_OUT
    outs = _ports(params, "output")
    # Echoed, never blanked: the app writes these ports over the block's own.
    assert outs["code"]["port_content"] == BROKEN
    assert outs["code_change"]["port_content"] == ""
    assert outs["errors"]["port_content"] == "AI not configured"
    assert outs["status"]["port_content"] == "error"
    assert "could not be applied" in outs["improvements"]["port_content"]


def test_the_stub_says_nothing_when_there_was_no_program_to_keep():
    body = FixRequest(block_name="f", fix=ORDER)
    outs = _ports(
        blocks_router._simple_code_fix_block_response(body)["tool_calls"][0]["params"],
        "output",
    )
    assert outs["code"]["port_content"] == ""
    assert outs["improvements"]["port_content"] == ""


# ── Vocabulary and prompt wiring ──────────────────────────────────────────────


def test_the_block_type_resolves_to_its_prompt_section():
    assert BlockType.CODE_FIX.value == "code_fix"
    assert BLOCK_TYPE_SECTION["code_fix"] == "create_code_fix"
    prompt = get_system_prompt("create_code_fix")
    assert prompt
    for key in ("code", "explanation", "improvements", "language", "compliance"):
        assert f'"{key}"' in prompt
    # The three rules that make this block different from [create_code].
    assert "MANDATORY INSTRUCTIONS" in prompt
    assert "MINIMAL CHANGE" in prompt
    assert "Do not return the program unchanged" in prompt


def test_it_is_not_the_hdl_repair_mode():
    """``code_fix`` the TYPE and ``fix_rtl`` the MODE must stay distinct."""
    from app.core.constants import CODE_FIX_RTL_SECTION

    assert CODE_FIX_RTL_SECTION == "fix_rtl"
    assert CODE_FIX_RTL_SECTION not in BLOCK_TYPE_SECTION.values()
    assert BLOCK_TYPE_SECTION["code_fix"] != CODE_FIX_RTL_SECTION


# ── The chat / voice create path ───────────────────────────────────────────────


def test_code_fix_is_enrichable():
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "code_fix"})
    assert enrichment._ENRICHERS["code_fix"] is enrichment._enrich_code_fix_block


@pytest.mark.asyncio
async def test_enrich_generates_when_the_chat_supplied_both_halves(monkeypatch):
    fake, calls = _responder(_answer())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    action = {
        "type": "create_block", "block_type": "code_fix", "block_name": "fix_reader",
        "description": "mend the reader", "source_code": BROKEN, "fix": ORDER,
        "language": "python",
    }
    assert await enrichment._enrich_code_fix_block(action, "s1") == ("ok", "")
    assert len(calls) == 1
    outs = {p["port_name"]: p["port_content"] for p in action["output_ports"]}
    assert outs["code"] == FIXED_ON_PORT
    assert outs["code_change"].startswith("--- before")


@pytest.mark.asyncio
async def test_enrich_only_scaffolds_when_the_program_is_coming_by_wire(monkeypatch):
    """The normal chat case: the order is stated, the program is on another block."""
    fake, calls = _responder(_answer())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    action = {
        "type": "create_block", "block_type": "code_fix", "block_name": "f",
        "description": "fix the retry logic",
    }
    assert await enrichment._enrich_code_fix_block(action, "s1") == ("ok", "")
    assert calls == []  # nothing to repair yet, so no money was spent
    assert [p["port_name"] for p in action["input_ports"]] == CODE_FIX_IN
    assert [p["port_name"] for p in action["output_ports"]] == CODE_FIX_OUT
    ins = {p["port_name"]: p["port_content"] for p in action["input_ports"]}
    assert ins["fix"] == "fix the retry logic"
    assert ins["code"] == ""


@pytest.mark.asyncio
async def test_enrich_scaffolds_without_a_key_and_says_so(monkeypatch):
    monkeypatch.setattr(
        blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic=""))
    action = {
        "type": "create_block", "block_type": "code_fix", "block_name": "f",
        "description": "d", "source_code": BROKEN, "fix": ORDER,
    }
    assert await enrichment._enrich_code_fix_block(action, "s1") == (
        "failed", "AI not configured")
    # Still port-complete, and the program it was given is kept (the scaffold
    # fallback strips its seeds, so compare the stripped form).
    assert [p["port_name"] for p in action["output_ports"]] == CODE_FIX_OUT
    ins = {p["port_name"]: p["port_content"] for p in action["input_ports"]}
    assert ins["code"] == BROKEN.strip()


def test_the_create_block_tool_declares_and_forwards_both_seeds():
    from app.modules.session.canvas_tools import (
        CANVAS_FUNCTION_DECLARATIONS,
        function_call_to_action,
    )

    decl = next(d for d in CANVAS_FUNCTION_DECLARATIONS if d["name"] == "create_block")
    props = decl["parameters"]["properties"]
    assert "fix" in props
    assert "source_code" in props
    assert "code_fix" in props["block_type"]["description"]

    action = function_call_to_action("create_block", {
        "block_type": "code_fix", "block_name": "f", "description": "d",
        "fix": ORDER, "source_code": BROKEN,
    })
    assert action["fix"] == ORDER
    assert action["source_code"] == BROKEN
    # `code` still means a tools block's Python and is not hijacked here.
    assert "code" not in action
