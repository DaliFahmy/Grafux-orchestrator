"""
Tests for the RTL repair path of the code block — the half of the verification
loop that turns a verilator block's failing-test report back into working RTL.

The contract being pinned here is narrow and load-bearing: a repair may change the
design and nothing else. If it renames the module or touches the port list, the
testbench it was meant to satisfy no longer binds and every downstream block on
the canvas loses its wire — so the interface check, and the fact that a violation
is REJECTED rather than accepted, is the point of most of these tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.modules.blocks import router as blocks_router
from app.modules.blocks.hdl import validate_rtl_fix
from app.modules.blocks.schemas import CodeGenerateRequest

BUGGY = """module sync_fifo #(parameter WIDTH = 8, parameter DEPTH = 8) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] data_in,
    output wire             full
);
    reg [3:0] count;
    assign full = (count == DEPTH + 1);
endmodule
"""

FIXED = BUGGY.replace("(count == DEPTH + 1)", "(count == DEPTH)")

FAILURES = """1 of 5 cocotb tests failed.

FAILED test_full_asserts_at_depth
  full must assert after 8 writes (spec 5: full iff count == 8), got 0 with count=8
"""


def _fake_settings(key: str = "sk-test"):
    return SimpleNamespace(openai_api_key=key, openai_model="gpt-test")


def _responder(*payloads):
    """A fake LLM that returns each payload in turn, recording every call."""
    calls: list[dict] = []
    remaining = list(payloads)

    async def fake_call(system_prompt, user_message, temperature=0.3, **kwargs):
        calls.append({"system_prompt": system_prompt, "user_message": user_message,
                      "kwargs": kwargs})
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return fake_call, calls


def _payload(code: str, **over) -> dict:
    base = {"code": code, "explanation": "Root cause: off-by-one in the full flag.",
            "improvements": "", "dependencies": "None", "language": "verilog"}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# code_prompt_section — which prompt a request gets
# ---------------------------------------------------------------------------

def test_feedback_plus_previous_code_selects_the_rtl_fixer():
    assert blocks_router.code_prompt_section(
        feedback=FAILURES, previous_code=BUGGY, language="verilog") == "fix_rtl"


@pytest.mark.parametrize("lang", ["verilog", "v", "sv", "systemverilog", "vhdl"])
def test_every_hdl_spelling_reaches_the_fixer(lang):
    assert blocks_router.code_prompt_section(
        feedback=FAILURES, previous_code=BUGGY, language=lang) == "fix_rtl"


def test_feedback_alone_is_not_a_repair():
    """There is nothing to minimally edit without the design that failed."""
    assert blocks_router.code_prompt_section(
        feedback=FAILURES, previous_code="", language="verilog") == "create_code"


def test_previous_code_alone_is_not_a_repair():
    """A plain Run on a block that already has code must stay a plain Run."""
    assert blocks_router.code_prompt_section(
        feedback="", previous_code=BUGGY, language="verilog") == "create_code"


def test_a_python_block_with_feedback_still_writes_fresh_code():
    """
    The frozen-interface rule that makes [fix_rtl] safe is meaningless outside
    HDL, and [create_code] already treats feedback as extra requirement text.
    """
    assert blocks_router.code_prompt_section(
        feedback="it crashes on empty input", previous_code="print(1)",
        language="python") == "create_code"


# ---------------------------------------------------------------------------
# validate_rtl_fix — the enforcement behind the prompt's promise
# ---------------------------------------------------------------------------

def test_a_minimal_fix_is_accepted():
    assert validate_rtl_fix(FIXED, BUGGY, "sync_fifo") == []


def test_a_renamed_module_is_rejected():
    renamed = FIXED.replace("module sync_fifo", "module fifo8")
    problems = validate_rtl_fix(renamed, BUGGY, "sync_fifo")
    assert any("renamed" in p for p in problems)


def test_a_removed_port_is_rejected_and_named():
    dropped = FIXED.replace("    input  wire             wr_en,\n", "")
    problems = validate_rtl_fix(dropped, BUGGY, "sync_fifo")
    assert any("removed wr_en" in p for p in problems)


def test_an_added_port_is_rejected_and_named():
    grown = FIXED.replace("    output wire             full\n",
                          "    output wire             full,\n"
                          "    output wire             empty\n")
    problems = validate_rtl_fix(grown, BUGGY, "sync_fifo")
    assert any("added empty" in p for p in problems)


def test_an_unchanged_design_is_rejected():
    """
    Otherwise the client loop re-provisions a pod to reproduce the identical
    failure until the iteration budget runs out.
    """
    problems = validate_rtl_fix(BUGGY, BUGGY, "sync_fifo")
    assert any("identical" in p for p in problems)


def test_a_truncated_design_is_rejected():
    problems = validate_rtl_fix("module sync_fifo(input clk);", BUGGY, "sync_fifo")
    assert any("endmodule" in p for p in problems)


def test_an_empty_response_is_rejected():
    assert validate_rtl_fix("", BUGGY, "sync_fifo") == ["the fixer returned no code"]


def test_an_unparsable_interface_does_not_condemn_the_fix():
    """
    An empty port list means "could not tell", never "no ports" — rejecting a
    good fix because a regex failed is worse than accepting one that is fine.
    """
    opaque = "`include \"weird.vh\"\nmodule sync_fifo `PORTS ;\nendmodule\n"
    assert validate_rtl_fix(FIXED, opaque, "sync_fifo") == []


# ---------------------------------------------------------------------------
# generate_code_payload on the repair path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repair_uses_the_fix_prompt_and_shows_the_frozen_interface(monkeypatch):
    fake, calls = _responder(_payload(FIXED))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_payload(
        block_name="fifo", description="an 8-entry FIFO", language="verilog",
        feedback=FAILURES, previous_code=BUGGY,
    )

    assert len(calls) == 1
    prompt = calls[0]["system_prompt"]
    assert "FROZEN" in prompt                      # the [fix_rtl] section, not [create_code]
    message = calls[0]["user_message"]
    # The interface line is what makes "keep the header identical" checkable.
    assert "clk, rst_n, wr_en, data_in, full" in message
    assert FAILURES.strip() in message
    assert BUGGY.strip() in message
    ports = {p["port_name"]: p["port_content"]
             for p in result["tool_calls"][0]["params"]["output_ports"]}
    assert "(count == DEPTH)" in ports["code"]


@pytest.mark.asyncio
async def test_repair_asks_for_enough_tokens_to_return_a_whole_file(monkeypatch):
    """A truncated design is worse than no design; the default cap is too low."""
    fake, calls = _responder(_payload(FIXED))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    await blocks_router.generate_code_payload(
        block_name="fifo", language="verilog", feedback=FAILURES, previous_code=BUGGY)

    assert calls[0]["kwargs"].get("max_tokens", 0) >= 8192


@pytest.mark.asyncio
async def test_a_broken_interface_triggers_exactly_one_repair_round(monkeypatch):
    renamed = FIXED.replace("module sync_fifo", "module fifo8")
    fake, calls = _responder(_payload(renamed), _payload(FIXED))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_payload(
        block_name="fifo", language="verilog", feedback=FAILURES, previous_code=BUGGY)

    assert len(calls) == 2
    assert "REJECTED" in calls[1]["user_message"]
    assert "renamed" in calls[1]["user_message"]
    ports = {p["port_name"]: p["port_content"]
             for p in result["tool_calls"][0]["params"]["output_ports"]}
    assert "module sync_fifo" in ports["code"]
    assert not ports["improvements"].startswith("Validation:")


@pytest.mark.asyncio
async def test_repair_rounds_are_bounded_and_the_problem_is_surfaced(monkeypatch):
    """
    A second violation means the model has lost the thread. The user is better
    served by the design plus the complaint than by another minute of retries.
    """
    renamed = FIXED.replace("module sync_fifo", "module fifo8")
    fake, calls = _responder(_payload(renamed))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_payload(
        block_name="fifo", language="verilog", feedback=FAILURES, previous_code=BUGGY)

    assert len(calls) == 2
    ports = {p["port_name"]: p["port_content"]
             for p in result["tool_calls"][0]["params"]["output_ports"]}
    assert ports["improvements"].startswith("Validation:")
    assert "renamed" in ports["improvements"]


@pytest.mark.asyncio
async def test_a_plain_generation_still_uses_the_create_prompt(monkeypatch):
    """The repair path must not change what an ordinary code block does."""
    fake, calls = _responder(_payload("fn main() {}", language="rust"))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    await blocks_router.generate_code_payload(
        block_name="hello", description="say hi", language="rust")

    assert len(calls) == 1
    assert "FROZEN" not in calls[0]["system_prompt"]
    assert "Requirement (what the code should do)" in calls[0]["user_message"]


@pytest.mark.asyncio
async def test_every_code_block_has_a_feedback_input_port(monkeypatch):
    """The wire has to be drawable before there is anything to say on it."""
    fake, _ = _responder(_payload("print(1)", language="python"))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_payload(
        block_name="hello", description="say hi", language="python")

    names = [p["port_name"] for p in result["tool_calls"][0]["params"]["input_ports"]]
    assert "feedback" in names


# ---------------------------------------------------------------------------
# The no-key fallback
# ---------------------------------------------------------------------------

def test_the_fallback_never_wipes_a_working_design():
    """
    The app writes these ports over the block's own. Returning an empty `code`
    on a repair request would destroy the design the loop exists to improve,
    every time a key was missing or a call failed.
    """
    body = CodeGenerateRequest(block_name="fifo", language="verilog",
                               feedback=FAILURES, previous_code=BUGGY)
    ports = {p["port_name"]: p["port_content"]
             for p in blocks_router._simple_code_response(body)
             ["tool_calls"][0]["params"]["output_ports"]}
    assert ports["code"].strip() == BUGGY.strip()
    assert "unchanged" in ports["improvements"]


def test_the_fallback_for_a_fresh_block_is_still_empty():
    body = CodeGenerateRequest(block_name="hello", language="python",
                               description="say hi")
    ports = {p["port_name"]: p["port_content"]
             for p in blocks_router._simple_code_response(body)
             ["tool_calls"][0]["params"]["output_ports"]}
    assert ports["code"] == ""
    assert ports["improvements"] == ""
    assert ports["language"] == "python"
