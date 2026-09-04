from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from app.modules.blocks import hdl
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import (
    CodeHdlGenerateRequest as HdlRequest,  # alias: pytest must not collect it
)
from app.modules.session import enrichment

SPEC = (
    "8-entry 16-bit synchronous FIFO; full/empty flags; a write when full is "
    "ignored; one-cycle read latency."
)

GOOD_RTL = """
module sync_fifo #(parameter WIDTH = 16, DEPTH = 8) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] wr_data,
    input  wire             rd_en,
    output reg  [WIDTH-1:0] rd_data,
    output wire             full,
    output wire             empty
);
    reg [3:0] wr_ptr, rd_ptr;
    assign full  = (wr_ptr - rd_ptr) == DEPTH;
    assign empty = wr_ptr == rd_ptr;
    always @(posedge clk) begin
        if (!rst_n) wr_ptr <= 4'd0;
        else if (wr_en && !full) wr_ptr <= wr_ptr + 4'd1;
    end
endmodule
"""

# Stops inside the always block: `endmodule` is present, so the check
# validate_rtl_fix does would pass it, but the design cannot elaborate.
TRUNCATED_RTL = """
module sync_fifo (input wire clk, input wire rst_n, output wire full);
    reg [3:0] wr_ptr;
    always @(posedge clk) begin
        if (!rst_n) wr_ptr <= 4'd0;
endmodule
"""

GOOD_VHDL = """
entity gray_counter is
  port (clk : in std_logic; rst : in std_logic; q : out std_logic_vector(3 downto 0));
end entity;

architecture rtl of gray_counter is
begin
end architecture;
"""

PORTS = ["clk", "rst_n", "wr_en", "wr_data", "rd_en", "rd_data", "full", "empty"]


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


# ── hdl helpers (pure) ────────────────────────────────────────────────────────


def test_hdl_family_collapses_the_verilog_spellings():
    assert hdl.hdl_family("verilog") == "verilog"
    assert hdl.hdl_family("SystemVerilog") == "verilog"
    assert hdl.hdl_family("sv") == "verilog"
    assert hdl.hdl_family("VHDL") == "vhdl"
    assert hdl.hdl_family("python") == ""
    assert hdl.hdl_family("") == ""


def test_design_interface_reads_the_generated_header():
    assert hdl.design_interface(GOOD_RTL, "systemverilog", "sync_fifo") == (
        "sync_fifo",
        PORTS,
    )


def test_design_interface_names_the_vhdl_entity_but_lists_no_ports():
    # No VHDL port parser on purpose: verilator cannot simulate VHDL, so the
    # interface would serve a path that dead-ends before a testbench.
    assert hdl.design_interface(GOOD_VHDL, "vhdl", "") == ("gray_counter", [])


def test_validate_hdl_design_accepts_a_clean_design():
    assert hdl.validate_hdl_design(GOOD_RTL, "verilog", "sync_fifo") == ([], [])


def test_validate_hdl_design_catches_truncation_that_endmodule_alone_would_miss():
    problems, _ = hdl.validate_hdl_design(TRUNCATED_RTL, "verilog", "sync_fifo")
    assert any("begin" in p and "end" in p for p in problems)
    # The check validate_rtl_fix relies on would have let this through.
    assert "endmodule" in TRUNCATED_RTL


def test_validate_hdl_design_rejects_a_renamed_top():
    problems, _ = hdl.validate_hdl_design(GOOD_RTL, "verilog", "other_name")
    assert any("other_name" in p for p in problems)


def test_validate_hdl_design_rejects_a_testbench():
    tb = "import cocotb\n\n@cocotb.test()\nasync def t(dut):\n    pass\nmodule x(); endmodule\n"
    problems, _ = hdl.validate_hdl_design(tb, "verilog", "x")
    assert any("testbench" in p for p in problems)


def test_simulation_only_constructs_are_warnings_not_problems():
    # SystemVerilog FPGA code legitimately initialises registers this way;
    # rejecting it would throw away a correct design.
    src = GOOD_RTL.replace("    reg [3:0] wr_ptr", "    initial $display(1);\n    reg [3:0] wr_ptr")
    problems, warnings = hdl.validate_hdl_design(src, "systemverilog", "sync_fifo")
    assert problems == []
    assert len(warnings) == 2


def test_validate_hdl_design_checks_vhdl_structurally():
    assert hdl.validate_hdl_design(GOOD_VHDL, "vhdl") == ([], [])
    problems, _ = hdl.validate_hdl_design(GOOD_VHDL.replace("of gray_counter", "of other"), "vhdl")
    assert any("architecture" in p for p in problems)


# ── scaffold contract ─────────────────────────────────────────────────────────
#
# These lists are duplicated in Grafux-app's EdaPorts::kCodeHdlInputs/Outputs and
# asserted there by tst_purehelpers. They are ordered comparisons on purpose: a
# reordering means an AI-created and a hand-created block disagree.

CODE_HDL_IN = ["block_description", "spec", "language", "top", "constraints", "feedback"]
CODE_HDL_OUT = [
    "code", "top", "interface", "language", "explanation", "improvements",
    "status", "errors",
]


@pytest.mark.asyncio
async def test_scaffold_code_hdl_exact_ports_seeds_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_hdl",
        block_name="sync fifo",
        description="An 8-entry FIFO",
        seeds={"top": "sync_fifo", "spec": SPEC},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "code_hdl"
    assert params["name"] == "sync_fifo"
    assert [p["port_name"] for p in params["input_ports"]] == CODE_HDL_IN
    assert [p["port_name"] for p in params["output_ports"]] == CODE_HDL_OUT
    ins = _ports(params, "input")
    assert ins["top"]["port_content"] == "sync_fifo"
    assert ins["spec"]["port_content"] == SPEC
    assert ins["language"]["port_content"] == "systemverilog"
    assert ins["spec"]["port_path"] == "data/code_hdl/general/sync_fifo/inputs/spec.txt"


@pytest.mark.asyncio
async def test_scaffold_code_hdl_spec_falls_back_to_description():
    result = await blocks_router.generate_scaffold_payload(
        block_type="code_hdl", block_name="ctr", description="A counter that wraps at 15",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["spec"]["port_content"] == "A counter that wraps at 15"


@pytest.mark.asyncio
async def test_code_hdl_wires_into_testbench_and_verilator():
    hdl_block = await blocks_router.generate_scaffold_payload(
        block_type="code_hdl", block_name="d")
    tb = await blocks_router.generate_scaffold_payload(block_type="testbench", block_name="t")
    ver = await blocks_router.generate_scaffold_payload(block_type="verilator", block_name="v")
    out = set(_ports(hdl_block["tool_calls"][0]["params"], "output"))
    tb_in = set(_ports(tb["tool_calls"][0]["params"], "input"))
    ver_in = set(_ports(ver["tool_calls"][0]["params"], "input"))
    # code_hdl.code -> testbench.rtl and verilator.rtl; code_hdl.top -> both tops
    assert {"code", "top"} <= out
    assert {"rtl", "top"} <= tb_in
    assert {"rtl", "top"} <= ver_in
    # ...and the loop's return leg: verilator.failures -> code_hdl.feedback
    ver_out = set(_ports(ver["tool_calls"][0]["params"], "output"))
    assert "failures" in ver_out
    assert "feedback" in set(_ports(hdl_block["tool_calls"][0]["params"], "input"))


# ── AI generation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_code_hdl_fills_ports_and_derives_the_interface(monkeypatch):
    fake, calls = _responder({
        "code": "```systemverilog\n" + GOOD_RTL + "\n```",
        "explanation": "A pointer-based FIFO.",
        "improvements": "",
        "language": "systemverilog",
        # The model claims a different top; the server must ignore it and read
        # the source, because [fix_rtl] has no such key and a claim that
        # disagreed with the code would be worse than no claim.
        "top": "not_the_real_top",
    })
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_hdl_payload(
        block_name="sync_fifo", spec=SPEC, language="systemverilog", top="sync_fifo",
        constraints="Active-low synchronous reset.",
    )
    params = result["tool_calls"][0]["params"]
    assert [p["port_name"] for p in params["input_ports"]] == CODE_HDL_IN
    assert [p["port_name"] for p in params["output_ports"]] == CODE_HDL_OUT
    outs = _ports(params, "output")
    assert outs["code"]["port_content"].startswith("module sync_fifo")
    assert "```" not in outs["code"]["port_content"]
    assert outs["top"]["port_content"] == "sync_fifo"
    assert json.loads(outs["interface"]["port_content"]) == PORTS
    assert outs["status"]["port_content"] == "ok"
    ins = _ports(params, "input")
    assert ins["constraints"]["port_content"] == "Active-low synchronous reset."
    # The spec and the constraints both reach the model.
    assert SPEC in calls[0][1]
    assert "Active-low synchronous reset." in calls[0][1]


@pytest.mark.asyncio
async def test_generate_code_hdl_repairs_once_then_reports(monkeypatch):
    fake, calls = _responder(
        {"code": TRUNCATED_RTL, "explanation": "", "improvements": ""},
        {"code": TRUNCATED_RTL, "explanation": "", "improvements": ""},
    )
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_hdl_payload(
        block_name="sync_fifo", spec=SPEC, top="sync_fifo",
    )
    assert len(calls) == 2, "one repair round, not more"
    assert "REJECTED" in calls[1][1]
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    assert outs["improvements"]["port_content"].startswith("Validation: ")


@pytest.mark.asyncio
async def test_feedback_plus_previous_code_selects_the_shared_rtl_fixer(monkeypatch):
    fixed = GOOD_RTL.replace("wr_ptr + 4'd1", "wr_ptr + 4'd2")
    fake, calls = _responder({"code": fixed, "explanation": "ROOT CAUSE: increment.",
                              "improvements": "", "language": "verilog"})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_hdl_payload(
        block_name="sync_fifo", language="verilog", top="sync_fifo",
        feedback="test_write_when_full FAILED", previous_code=GOOD_RTL,
    )
    system_prompt = calls[0][0]
    assert system_prompt == blocks_router.get_system_prompt(
        blocks_router._CODE_FIX_RTL_SECTION)
    assert "test_write_when_full FAILED" in calls[0][1]
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["code"]["port_content"] == fixed.strip()
    # The interface survived the repair, which is the whole contract of [fix_rtl].
    assert json.loads(outs["interface"]["port_content"]) == PORTS


@pytest.mark.asyncio
async def test_a_repair_that_returns_nothing_keeps_the_previous_design(monkeypatch):
    # The app overwrites every port it is handed, so an empty `code` here would
    # destroy the design the loop exists to improve.
    fake, _ = _responder({"code": "", "explanation": "", "improvements": ""})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_hdl_payload(
        block_name="sync_fifo", top="sync_fifo",
        feedback="everything FAILED", previous_code=GOOD_RTL,
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["code"]["port_content"].strip() == GOOD_RTL.strip()
    assert outs["status"]["port_content"] == "needs_review"


@pytest.mark.asyncio
async def test_vhdl_says_the_loop_is_unavailable_rather_than_leaving_a_blank_port(monkeypatch):
    fake, _ = _responder({"code": GOOD_VHDL, "explanation": "", "improvements": ""})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)

    result = await blocks_router.generate_code_hdl_payload(
        block_name="gray_counter", spec="4-bit gray counter", language="vhdl",
        top="gray_counter",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["interface"]["port_content"] == ""
    assert "verilator" in outs["improvements"]["port_content"]
    assert outs["status"]["port_content"] == "needs_review"


@pytest.mark.asyncio
async def test_a_non_hdl_language_is_refused(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    with pytest.raises(ValueError):
        await blocks_router.generate_code_hdl_payload(
            block_name="x", spec=SPEC, language="python")


@pytest.mark.asyncio
async def test_a_fresh_generation_needs_a_spec(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    with pytest.raises(ValueError):
        await blocks_router.generate_code_hdl_payload(block_name="x")


@pytest.mark.asyncio
async def test_no_llm_key_returns_none_so_callers_scaffold(monkeypatch):
    monkeypatch.setattr(
        blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic=""))
    assert await blocks_router.generate_code_hdl_payload(
        block_name="x", spec=SPEC) is None


@pytest.mark.asyncio
async def test_an_anthropic_only_deployment_still_generates(monkeypatch):
    fake, _ = _responder({"code": GOOD_RTL, "explanation": "", "improvements": ""})
    monkeypatch.setattr(
        blocks_router, "get_settings",
        lambda: _fake_settings(openai="", anthropic="sk-ant-test"))
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_code_hdl_payload(
        block_name="sync_fifo", spec=SPEC, top="sync_fifo")
    assert result is not None


def test_stub_is_port_complete_and_never_blanks_a_design():
    body = HdlRequest(
        block_name="sync fifo", spec=SPEC, top="sync_fifo", language="verilog",
        feedback="FAILED", previous_code=GOOD_RTL,
    )
    params = blocks_router._simple_code_hdl_response(
        body, error="AI not configured")["tool_calls"][0]["params"]
    assert params["block_type"] == "code_hdl"
    assert [p["port_name"] for p in params["input_ports"]] == CODE_HDL_IN
    assert [p["port_name"] for p in params["output_ports"]] == CODE_HDL_OUT
    outs = _ports(params, "output")
    assert outs["code"]["port_content"] == GOOD_RTL.strip()
    assert outs["errors"]["port_content"] == "AI not configured"
    assert outs["status"]["port_content"] == "error"
    assert json.loads(outs["interface"]["port_content"]) == PORTS


# ── voice / text creation ─────────────────────────────────────────────────────


def test_code_hdl_is_enrichable():
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "code_hdl"})


@pytest.mark.asyncio
async def test_enrich_code_hdl_block_fills_ports(monkeypatch):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"tool_calls": [{"params": {
            "output_ports": [{"port_name": "code", "port_content": GOOD_RTL}],
            "input_ports": [{"port_name": "spec", "port_content": SPEC}],
        }}]}

    monkeypatch.setattr(blocks_router, "generate_code_hdl_payload", fake_generate)
    action = {
        "type": "create_block", "block_type": "code_hdl", "block_name": "sync_fifo",
        "description": "An 8-entry FIFO", "spec": SPEC, "language": "verilog",
    }
    assert await enrichment._enrich_code_hdl_block(action, "s1") == ("ok", "")
    assert action["output_ports"][0]["port_content"] == GOOD_RTL
    assert captured["spec"] == SPEC
    # The top defaults to the block name so downstream blocks have something to
    # address before the user has typed anything.
    assert captured["top"] == "sync_fifo"


@pytest.mark.asyncio
async def test_enrich_code_hdl_block_scaffolds_without_a_key(monkeypatch):
    async def no_key(**kwargs):
        return None

    monkeypatch.setattr(blocks_router, "generate_code_hdl_payload", no_key)
    action = {
        "type": "create_block", "block_type": "code_hdl", "block_name": "ctr",
        "description": "A counter", "spec": "A 4-bit counter",
    }
    status = await enrichment._enrich_code_hdl_block(action, "s1")
    assert status == ("failed", "AI not configured")
    assert [p["port_name"] for p in action["input_ports"]] == CODE_HDL_IN
    assert [p["port_name"] for p in action["output_ports"]] == CODE_HDL_OUT
