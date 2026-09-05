from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from app.modules.blocks import hdl
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import (
    SpecHdlGenerateRequest as SpecRequest,  # alias: pytest must not collect it
)
from app.modules.session import canvas_tools, enrichment

EXPLANATION = (
    "a small fifo between the sensor and the bus; it should hold 8 words and not "
    "lose anything when the reader is slow"
)

SPEC_TEXT = (
    "The module is a synchronous 8-entry FIFO of DATA_WIDTH-bit words with "
    "full and empty flags. A write presented while full is ignored."
)

REQUIREMENTS = (
    "REQ-1: After reset, empty is 1 and full is 0.\n"
    "REQ-2: full asserts on the cycle after the eighth write.\n"
    "REQ-3: A write asserted while full must not modify any stored word."
)

INTERFACE = json.dumps([
    {"name": "clk", "direction": "input", "width": "1", "description": "clock"},
    {"name": "rst_n", "direction": "input", "width": "1", "description": "reset"},
    {"name": "wr_en", "direction": "input", "width": "1", "description": "write strobe"},
    {"name": "full", "direction": "output", "width": "1", "description": "full flag"},
])

SIGNALS = (
    "clk: input, 1 bit, single domain, free-running.\n"
    "rst_n: input, 1 bit, synchronous release.\n"
    "wr_en: input, 1 bit, sampled on the rising edge.\n"
    "full: output, 1 bit, registered, resets to 0."
)

EXISTING_RTL = """
module sync_fifo (
    input  wire clk,
    input  wire rst_n,
    input  wire wr_en,
    output wire full
);
    assign full = 1'b0;
endmodule
"""


def _payload(**over):
    """A well-formed model answer; override one key to break exactly one rule."""
    base = {
        "spec": SPEC_TEXT,
        "requirements": REQUIREMENTS,
        "interface": INTERFACE,
        "signals_analysis": SIGNALS,
        "parameters": "DATA_WIDTH = 16  # word width",
        "timing": "Rising edge of clk; synchronous active-low reset.",
        "assumptions": "Assumed a single clock domain.",
        "improvements": "Decide whether an overflow should be reported.",
        "explanation": "A small synchronous FIFO.",
        "top": "sync_fifo",
    }
    base.update(over)
    return base


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


# ── spec helpers (pure) ───────────────────────────────────────────────────────


def test_requirement_ids_reads_the_enumeration():
    assert hdl.requirement_ids(REQUIREMENTS) == ["REQ-1", "REQ-2", "REQ-3"]
    assert hdl.requirement_ids("no numbering at all") == []


def test_parse_interface_accepts_a_bare_array_and_a_wrapped_one():
    assert hdl.interface_signals(INTERFACE) == ["clk", "rst_n", "wr_en", "full"]
    wrapped = json.dumps({"ports": json.loads(INTERFACE)})
    assert hdl.interface_signals(wrapped) == ["clk", "rst_n", "wr_en", "full"]
    # Unparsable is "cannot tell", never a crash.
    assert hdl.interface_signals("clk, rst_n, wr_en") == []
    assert hdl.parse_interface("") == []


def test_validate_spec_accepts_a_complete_contract():
    problems, warnings = hdl.validate_spec(
        SPEC_TEXT, REQUIREMENTS, INTERFACE, signals_analysis=SIGNALS
    )
    assert problems == []
    assert warnings == []


def test_unnumbered_requirements_are_a_problem_because_tests_cite_them():
    problems, _ = hdl.validate_spec(
        SPEC_TEXT, "The FIFO holds eight words and flags when it is full.",
        INTERFACE, signals_analysis=SIGNALS,
    )
    assert any("REQ-" in p for p in problems)


def test_duplicate_requirement_numbers_are_a_problem():
    dupes = "REQ-1: a.\nREQ-2: b must not happen.\nREQ-1: c."
    problems, _ = hdl.validate_spec(SPEC_TEXT, dupes, INTERFACE, signals_analysis=SIGNALS)
    assert any("duplicate requirement numbers" in p and "REQ-1" in p for p in problems)


def test_unparsable_interface_is_a_problem_because_a_block_reads_it_as_json():
    problems, _ = hdl.validate_spec(
        SPEC_TEXT, REQUIREMENTS, "clk (in), rst_n (in), full (out)"
    )
    assert any("JSON" in p for p in problems)


def test_a_bad_direction_is_caught_by_name():
    iface = json.dumps([{"name": "clk", "direction": "in", "width": "1"}])
    problems, _ = hdl.validate_spec(SPEC_TEXT, REQUIREMENTS, iface)
    assert any("'clk'" in p and "'in'" in p for p in problems)


def test_a_placeholder_in_the_contract_is_a_problem():
    problems, _ = hdl.validate_spec(
        SPEC_TEXT + " Overflow behaviour: TODO.", REQUIREMENTS, INTERFACE,
        signals_analysis=SIGNALS,
    )
    assert any("placeholder" in p for p in problems)


def test_a_contract_with_no_must_not_clause_is_only_a_warning():
    positive_only = "REQ-1: empty is 1 after reset.\nREQ-2: full asserts after 8 writes."
    problems, warnings = hdl.validate_spec(
        SPEC_TEXT, positive_only, INTERFACE, signals_analysis=SIGNALS
    )
    # A weak contract is still a usable one — this must not cost a repair round.
    assert problems == []
    assert any("forbids" in w for w in warnings)


def test_an_unanalysed_signal_is_only_a_warning():
    _, warnings = hdl.validate_spec(
        SPEC_TEXT, REQUIREMENTS, INTERFACE, signals_analysis="clk: the clock."
    )
    assert any("rst_n" in w and "full" in w for w in warnings)


def test_drift_from_an_existing_design_is_a_warning_not_a_problem():
    iface = json.dumps([
        {"name": "clk", "direction": "input", "width": "1"},
        {"name": "almost_full", "direction": "output", "width": "1"},
    ])
    problems, warnings = hdl.validate_spec(
        SPEC_TEXT, REQUIREMENTS, iface, design=EXISTING_RTL, top="sync_fifo"
    )
    # The RTL may well be the thing that is wrong; this block states intent.
    assert problems == []
    assert any("almost_full" in w for w in warnings)


# ── full_spec: the whole contract as one document ─────────────────────────────


def _full(**over):
    """The output-port mapping compose_full_spec is handed, minus overrides."""
    base = _payload()
    base.update({"status": "ok", "errors": ""})
    base.update(over)
    return base


def test_full_spec_carries_every_substantive_port_in_document_order():
    doc = hdl.compose_full_spec(_full())
    assert doc.startswith("# Full Specification: sync_fifo")
    headings = [line for line in doc.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Top Module", "## Overview", "## Specification", "## Requirements",
        "## Interface", "## Signals Analysis", "## Parameters", "## Timing",
        "## Assumptions", "## Open Questions",
    ]
    assert SPEC_TEXT in doc
    assert "REQ-3: A write asserted while full must not modify" in doc
    assert SIGNALS in doc
    assert "DATA_WIDTH = 16" in doc


def test_full_spec_never_carries_the_run_bookkeeping_ports():
    # "needs_review" / "AI not configured" describe the RUN, not the contract;
    # a reader handed this document must not mistake them for requirements.
    doc = hdl.compose_full_spec(_full(status="needs_review", errors="boom"))
    assert "needs_review" not in doc
    assert "boom" not in doc
    assert "## Status" not in doc and "## Errors" not in doc


def test_an_empty_port_contributes_no_heading_at_all():
    doc = hdl.compose_full_spec(_full(assumptions="", improvements="   "))
    assert "## Assumptions" not in doc
    assert "## Open Questions" not in doc
    assert "## Specification" in doc


def test_the_interface_is_fenced_as_json():
    doc = hdl.compose_full_spec(_full())
    assert "## Interface\n```json\n[" in doc
    assert doc.count("```") == 2


def test_full_spec_is_empty_when_only_the_module_name_is_known():
    # A block that has not been run yet is not a title page.
    assert hdl.compose_full_spec({}) == ""
    assert hdl.compose_full_spec({"top": "sync_fifo", "status": "error"}) == ""


# ── scaffold (no AI, never needs a key) ───────────────────────────────────────

SPEC_HDL_IN = [
    "block_description", "explanation", "previous_code", "top", "language",
    "data_width", "addr_width", "parameters", "logic_style", "reset_style",
    "clocking", "protocol", "throughput", "constraints", "feedback",
]
SPEC_HDL_OUT = [
    "spec", "full_spec", "top", "interface", "signals_analysis", "parameters",
    "timing", "requirements", "assumptions", "explanation", "improvements",
    "status", "errors",
]


@pytest.mark.asyncio
async def test_scaffold_spec_hdl_exact_ports_seeds_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="spec_hdl",
        block_name="fifo spec",
        description="An 8-entry FIFO",
        seeds={"top": "sync_fifo", "spec": EXPLANATION},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "spec_hdl"
    assert params["name"] == "fifo_spec"
    assert [p["port_name"] for p in params["input_ports"]] == SPEC_HDL_IN
    assert [p["port_name"] for p in params["output_ports"]] == SPEC_HDL_OUT
    ins = _ports(params, "input")
    assert ins["top"]["port_content"] == "sync_fifo"
    assert ins["explanation"]["port_content"] == EXPLANATION
    assert ins["language"]["port_content"] == "systemverilog"
    assert ins["logic_style"]["port_content"] == "auto"
    assert ins["reset_style"]["port_content"] == "sync_active_high"
    assert ins["clocking"]["port_content"] == "single_clock"
    assert ins["explanation"]["port_path"] == (
        "data/spec_hdl/general/fifo_spec/inputs/explanation.txt"
    )


@pytest.mark.asyncio
async def test_scaffold_explanation_falls_back_to_the_description():
    result = await blocks_router.generate_scaffold_payload(
        block_type="spec_hdl", block_name="ctr", description="A counter that wraps at 15",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["explanation"]["port_content"] == "A counter that wraps at 15"


@pytest.mark.asyncio
async def test_spec_hdl_feeds_both_code_hdl_and_testbench():
    spec_block = await blocks_router.generate_scaffold_payload(
        block_type="spec_hdl", block_name="s")
    hdl_block = await blocks_router.generate_scaffold_payload(
        block_type="code_hdl", block_name="d")
    tb = await blocks_router.generate_scaffold_payload(block_type="testbench", block_name="t")
    out = set(_ports(spec_block["tool_calls"][0]["params"], "output"))
    hdl_in = set(_ports(hdl_block["tool_calls"][0]["params"], "input"))
    tb_in = set(_ports(tb["tool_calls"][0]["params"], "input"))
    # The whole point of the block: ONE spec text reaches both readers.
    assert {"spec", "top"} <= out
    assert {"spec", "top"} <= hdl_in
    assert {"spec", "top"} <= tb_in


@pytest.mark.asyncio
async def test_the_outer_loop_can_be_wired_back_to_the_spec():
    spec_block = await blocks_router.generate_scaffold_payload(
        block_type="spec_hdl", block_name="s")
    ver = await blocks_router.generate_scaffold_payload(block_type="verilator", block_name="v")
    spec_in = set(_ports(spec_block["tool_calls"][0]["params"], "input"))
    ver_in = set(_ports(ver["tool_calls"][0]["params"], "input"))
    ver_out = set(_ports(ver["tool_calls"][0]["params"], "output"))
    # verilator.improvements_spec -> spec_hdl.feedback, and
    # code_hdl.code -> spec_hdl.previous_code.
    assert {"feedback", "previous_code"} <= spec_in
    assert {"failures", "improvements_rtl", "improvements_spec"} <= ver_out
    # ...and the OUTWARD leg that makes the return leg worth reading:
    # spec_hdl.spec -> verilator.spec, so the review can cite REQ numbers
    # instead of only describing the ambiguity it found.
    assert "spec" in ver_in


def test_explanation_top_and_parameters_are_deliberately_on_both_sides():
    spec = blocks_router._SCAFFOLD_SPECS["spec_hdl"]
    # Rough description in, plain-language summary out; pinned in, resolved out.
    # Documented in the scaffold table because it breaks the "a shared name means
    # echoed through" convention the verilator block established.
    assert {"explanation", "top", "parameters"} <= set(spec.inputs)
    assert {"explanation", "top", "parameters"} <= set(spec.outputs)
    # interface is a PROPOSAL, never a request — out only, as on code_hdl.
    assert "interface" not in spec.inputs
    assert "interface" in spec.outputs


# ── design parameters ─────────────────────────────────────────────────────────


def test_pinned_parameters_become_prompt_lines_in_port_order():
    lines = blocks_router.spec_hdl_parameters({
        "data_width": "16", "addr_width": "3", "reset_style": "async_active_low",
        "clocking": "clock_domain_crossing",
    })
    assert lines == [
        "Data width: 16",
        "Address width: 3",
        "Reset style: async_active_low",
        "Clocking scheme: clock_domain_crossing",
    ]


def test_a_port_left_at_you_decide_is_not_forwarded_as_an_instruction():
    # "auto" is the logic_style default. Forwarding it would have the model
    # dutifully specify sequential logic for a combinational block.
    assert blocks_router.spec_hdl_parameters({"logic_style": "auto"}) == []
    assert blocks_router.spec_hdl_parameters({"protocol": "  "}) == []
    assert blocks_router.spec_hdl_parameters({"throughput": "none"}) == []
    assert blocks_router.spec_hdl_parameters({"logic_style": "combinational"}) == [
        "Logic style (combinational / sequential): combinational"
    ]


def test_the_generation_message_carries_the_pinned_parameters():
    msg = blocks_router.build_spec_hdl_gen_message(
        block_name="fifo_spec", explanation=EXPLANATION, language="systemverilog",
        top="sync_fifo", parameters=["Data width: 16"], constraints="sky130 target",
    )
    assert "Data width: 16" in msg
    assert "sync_fifo" in msg
    assert "sky130 target" in msg
    assert EXPLANATION in msg


def test_a_revision_message_carries_the_current_spec_and_the_feedback():
    msg = blocks_router.build_spec_hdl_gen_message(
        block_name="fifo_spec", explanation=EXPLANATION, language="verilog",
        top="sync_fifo", previous_spec=SPEC_TEXT, feedback="test_full_flag fails: off by one",
    )
    assert SPEC_TEXT in msg
    assert "off by one" in msg
    assert "keep the requirement numbering" in msg


# ── generation ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_writes_every_port_from_one_clean_answer(monkeypatch):
    fake, calls = _responder(_payload())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo spec", explanation=EXPLANATION, data_width="16",
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "spec_hdl"
    assert [p["port_name"] for p in params["input_ports"]] == SPEC_HDL_IN
    assert [p["port_name"] for p in params["output_ports"]] == SPEC_HDL_OUT
    outs = _ports(params, "output")
    assert outs["spec"]["port_content"] == SPEC_TEXT
    assert outs["requirements"]["port_content"] == REQUIREMENTS
    assert json.loads(outs["interface"]["port_content"])[0]["name"] == "clk"
    assert outs["signals_analysis"]["port_content"] == SIGNALS
    assert outs["timing"]["port_content"].startswith("Rising edge")
    assert outs["assumptions"]["port_content"] == "Assumed a single clock domain."
    assert outs["status"]["port_content"] == "ok"
    assert outs["errors"]["port_content"] == ""
    # A clean answer's improvements are the model's own, unprefixed.
    assert outs["improvements"]["port_content"] == (
        "Decide whether an overflow should be reported."
    )
    # full_spec is composed from the ports above, never asked of the model.
    doc = outs["full_spec"]["port_content"]
    assert doc.startswith("# Full Specification: sync_fifo")
    for fragment in (SPEC_TEXT, REQUIREMENTS, SIGNALS, "DATA_WIDTH = 16"):
        assert fragment in doc
    assert len(calls) == 1
    assert "Data width: 16" in calls[0][1]


@pytest.mark.asyncio
async def test_a_pinned_top_is_never_renamed_by_the_model(monkeypatch):
    fake, _ = _responder(_payload(top="something_else"))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION, top="sync_fifo",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    # The canvas is wired to this name; the model does not get to change it.
    assert outs["top"]["port_content"] == "sync_fifo"


@pytest.mark.asyncio
async def test_the_model_names_the_top_when_the_user_did_not(monkeypatch):
    fake, _ = _responder(_payload(top="sync_fifo"))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION,
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["top"]["port_content"] == "sync_fifo"


@pytest.mark.asyncio
async def test_a_rejected_draft_costs_exactly_one_repair_round(monkeypatch):
    bad = _payload(requirements="The FIFO holds eight words.")
    fake, calls = _responder(bad, _payload())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION,
    )
    assert len(calls) == 2
    assert "REJECTED" in calls[1][1]
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "ok"


@pytest.mark.asyncio
async def test_a_residual_problem_lands_on_improvements_and_needs_review(monkeypatch):
    bad = _payload(requirements="The FIFO holds eight words.")
    fake, calls = _responder(bad)  # never improves
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION,
    )
    assert len(calls) == 2  # one draft + one repair, then it gives up
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    improvements = outs["improvements"]["port_content"]
    assert improvements.startswith("Validation: ")
    assert "REQ-" in improvements
    # The model's own advice survives the prefix rather than being replaced.
    assert "Decide whether an overflow should be reported." in improvements


@pytest.mark.asyncio
async def test_a_warning_marks_needs_review_without_costing_a_repair(monkeypatch):
    weak = _payload(requirements="REQ-1: empty is 1 after reset.")
    fake, calls = _responder(weak)
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION,
    )
    assert len(calls) == 1  # a warning is not worth a second call
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    assert "forbids" in outs["improvements"]["port_content"]


@pytest.mark.asyncio
async def test_a_revision_that_returns_nothing_keeps_the_contract(monkeypatch):
    fake, _ = _responder(_payload(spec=""))
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION, previous_spec=SPEC_TEXT,
        feedback="test_full_flag fails",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    # The app writes every port it is handed: blanking this destroys the design
    # and the tests along with it.
    assert outs["spec"]["port_content"] == SPEC_TEXT


@pytest.mark.asyncio
async def test_an_existing_design_alone_is_enough_to_recover_a_spec(monkeypatch):
    fake, calls = _responder(_payload())
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_spec_hdl_payload(
        block_name="sync_fifo", previous_code=EXISTING_RTL, top="sync_fifo",
    )
    assert result is not None
    assert "Existing design" in calls[0][1]
    assert "Recover the specification" in calls[0][1]


@pytest.mark.asyncio
async def test_no_explanation_and_no_design_is_refused(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    with pytest.raises(ValueError, match="explanation"):
        await blocks_router.generate_spec_hdl_payload(block_name="empty")


@pytest.mark.asyncio
async def test_a_software_language_is_refused(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    with pytest.raises(ValueError, match="not a hardware description language"):
        await blocks_router.generate_spec_hdl_payload(
            block_name="fifo_spec", explanation=EXPLANATION, language="python",
        )


@pytest.mark.asyncio
async def test_no_key_returns_none_so_the_caller_can_scaffold(monkeypatch):
    monkeypatch.setattr(
        blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic="")
    )
    assert await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION) is None


@pytest.mark.asyncio
async def test_an_anthropic_key_alone_is_enough(monkeypatch):
    fake, _ = _responder(_payload())
    monkeypatch.setattr(
        blocks_router, "get_settings",
        lambda: _fake_settings(openai="", anthropic="sk-ant-test"),
    )
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_spec_hdl_payload(
        block_name="fifo_spec", explanation=EXPLANATION)
    assert result is not None


def test_stub_is_port_complete_and_never_blanks_a_specification():
    body = SpecRequest(
        block_name="fifo spec", explanation=EXPLANATION, top="sync_fifo",
        language="verilog", feedback="FAILED", previous_spec=SPEC_TEXT,
    )
    params = blocks_router._simple_spec_hdl_response(
        body, error="AI not configured")["tool_calls"][0]["params"]
    assert params["block_type"] == "spec_hdl"
    assert [p["port_name"] for p in params["input_ports"]] == SPEC_HDL_IN
    assert [p["port_name"] for p in params["output_ports"]] == SPEC_HDL_OUT
    outs = _ports(params, "output")
    assert outs["spec"]["port_content"] == SPEC_TEXT
    assert outs["top"]["port_content"] == "sync_fifo"
    assert outs["errors"]["port_content"] == "AI not configured"
    assert outs["status"]["port_content"] == "error"
    assert "returned unchanged" in outs["improvements"]["port_content"]
    # The document is preserved with the spec: a blank full_spec beside a kept
    # spec would read as a contract that had lost half of itself.
    assert SPEC_TEXT in outs["full_spec"]["port_content"]


def test_stub_on_a_fresh_block_says_nothing_about_an_unchanged_spec():
    body = SpecRequest(block_name="fifo_spec", explanation=EXPLANATION)
    params = blocks_router._simple_spec_hdl_response(
        body, error="AI not configured")["tool_calls"][0]["params"]
    outs = _ports(params, "output")
    assert outs["spec"]["port_content"] == ""
    assert outs["improvements"]["port_content"] == ""
    # With no pinned top the block name still gives downstream blocks a handle.
    assert outs["top"]["port_content"] == "fifo_spec"
    # ...but a name alone is not a specification, so the document stays empty.
    assert outs["full_spec"]["port_content"] == ""


# ── voice / text creation ─────────────────────────────────────────────────────


def test_spec_hdl_is_enrichable():
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "spec_hdl"})


def test_the_chat_tool_offers_the_type_and_the_explanation_seed():
    create = [
        f for f in canvas_tools.CANVAS_FUNCTION_DECLARATIONS if f["name"] == "create_block"
    ][0]
    props = create["parameters"]["properties"]
    assert "spec_hdl" in props["block_type"]["description"]
    assert "explanation" in props
    action = canvas_tools.function_call_to_action("create_block", {
        "block_type": "spec_hdl", "block_name": "fifo_spec", "description": "a fifo",
        "explanation": EXPLANATION, "top": "sync_fifo", "language": "systemverilog",
    })
    assert action["explanation"] == EXPLANATION
    assert action["top"] == "sync_fifo"


@pytest.mark.asyncio
async def test_enrich_spec_hdl_block_fills_ports(monkeypatch):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"tool_calls": [{"params": {
            "output_ports": [{"port_name": "spec", "port_content": SPEC_TEXT}],
            "input_ports": [{"port_name": "explanation", "port_content": EXPLANATION}],
        }}]}

    monkeypatch.setattr(blocks_router, "generate_spec_hdl_payload", fake_generate)
    action = {
        "type": "create_block", "block_type": "spec_hdl", "block_name": "fifo_spec",
        "description": "An 8-entry FIFO", "explanation": EXPLANATION, "language": "verilog",
    }
    assert await enrichment._enrich_spec_hdl_block(action, "s1") == ("ok", "")
    assert action["output_ports"][0]["port_content"] == SPEC_TEXT
    assert captured["explanation"] == EXPLANATION
    assert captured["language"] == "verilog"


@pytest.mark.asyncio
async def test_the_description_is_the_explanation_when_nothing_else_is_given(monkeypatch):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"tool_calls": [{"params": {"output_ports": [], "input_ports": []}}]}

    monkeypatch.setattr(blocks_router, "generate_spec_hdl_payload", fake_generate)
    action = {
        "type": "create_block", "block_type": "spec_hdl", "block_name": "ctr",
        "description": "a counter that wraps at 15",
    }
    await enrichment._enrich_spec_hdl_block(action, "s1")
    assert captured["explanation"] == "a counter that wraps at 15"


@pytest.mark.asyncio
async def test_the_spec_seed_key_is_accepted_as_an_alias(monkeypatch):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"tool_calls": [{"params": {"output_ports": [], "input_ports": []}}]}

    monkeypatch.setattr(blocks_router, "generate_spec_hdl_payload", fake_generate)
    # The chat schema tells the model to pass the shared spec as "spec" for the
    # whole HDL family; a user saying "with this spec: ..." must not get an
    # empty block for using the word the tool uses.
    action = {
        "type": "create_block", "block_type": "spec_hdl", "block_name": "fifo_spec",
        "description": "a fifo", "spec": EXPLANATION,
    }
    await enrichment._enrich_spec_hdl_block(action, "s1")
    assert captured["explanation"] == EXPLANATION


@pytest.mark.asyncio
async def test_a_non_hdl_language_scaffolds_instead_of_losing_the_block(monkeypatch):
    async def boom(**kwargs):
        raise ValueError("'python' is not a hardware description language")

    monkeypatch.setattr(blocks_router, "generate_spec_hdl_payload", boom)
    action = {
        "type": "create_block", "block_type": "spec_hdl", "block_name": "fifo_spec",
        "description": "a fifo", "language": "python",
    }
    status = await enrichment._enrich_spec_hdl_block(action, "s1")
    assert status[0] == "failed"
    # The block still arrives, port-complete, with a language the user can fix.
    assert [p["port_name"] for p in action["input_ports"]] == SPEC_HDL_IN
    assert [p["port_name"] for p in action["output_ports"]] == SPEC_HDL_OUT
    ins = {p["port_name"]: p["port_content"] for p in action["input_ports"]}
    assert ins["language"] == "systemverilog"


@pytest.mark.asyncio
async def test_without_a_key_the_block_is_scaffolded_and_the_failure_reported(monkeypatch):
    async def no_key(**kwargs):
        return None

    monkeypatch.setattr(blocks_router, "generate_spec_hdl_payload", no_key)
    action = {
        "type": "create_block", "block_type": "spec_hdl", "block_name": "fifo_spec",
        "description": "a fifo",
    }
    assert await enrichment._enrich_spec_hdl_block(action, "s1") == (
        "failed", "AI not configured"
    )
    assert [p["port_name"] for p in action["output_ports"]] == SPEC_HDL_OUT


# ── prompt wiring ─────────────────────────────────────────────────────────────


def test_the_block_type_resolves_to_its_prompt_section():
    from app.core.constants import BLOCK_TYPE_SECTION, BlockType
    from app.prompts import get_system_prompt

    assert BlockType.SPEC_HDL.value == "spec_hdl"
    section = BLOCK_TYPE_SECTION["spec_hdl"]
    assert section == "create_spec_hdl"
    prompt = get_system_prompt(section)
    assert prompt, "create_spec_hdl section missing from Msg_config"
    # The contract the router and the validators are written against.
    for key in ("spec", "requirements", "interface", "signals_analysis", "top"):
        assert f'"{key}"' in prompt
    assert "REQ-" in prompt
