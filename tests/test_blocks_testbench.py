from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from app.modules.blocks import hdl
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import (
    TestbenchGenerateRequest as TbRequest,  # alias: pytest must not collect it
)
from app.modules.session import enrichment

FIFO_RTL = """
// 8-entry synchronous FIFO
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
endmodule
"""

GOOD_TB = '''
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.rd_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.empty.value) == 1, "empty must be 1 after reset"
    assert int(dut.full.value) == 0, "full must be 0 after reset"


@cocotb.test()
async def test_write_when_full_must_not_be_accepted(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    for i in range(8):
        dut.wr_en.value = 1
        dut.wr_data.value = i
        await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.full.value) == 1, "full must assert after 8 writes"
'''

# Same tests, but peeking at an internal register the spec never mentions.
PEEKING_TB = GOOD_TB.replace(
    'assert int(dut.full.value) == 1, "full must assert after 8 writes"',
    'assert int(dut.wr_ptr.value) == 8, "internal pointer"',
)


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(openai_api_key=openai, anthropic_api_key=anthropic, openai_model="gpt-test")


def _ports(params, side):
    return {p["port_name"]: p for p in params[f"{side}_ports"]}


# ── hdl helpers (pure) ────────────────────────────────────────────────────────


def test_module_ports_ansi_with_parameters():
    assert hdl.module_ports(FIFO_RTL, "sync_fifo") == [
        "clk", "rst_n", "wr_en", "wr_data", "rd_en", "rd_data", "full", "empty",
    ]


def test_module_ports_non_ansi_and_unknown_module():
    rtl = "module m(a, b, y);\n input a, b;\n output y;\nendmodule"
    assert hdl.module_ports(rtl, "m") == ["a", "b", "y"]
    assert hdl.module_ports(rtl, "nope") == []
    assert hdl.module_ports("", "m") == []


def test_infer_top_module_is_last_declared():
    rtl = "module sub(); endmodule\nmodule top_mod(input a); endmodule"
    assert hdl.infer_top_module(rtl) == "top_mod"
    assert hdl.infer_top_module("") == "top"


def test_dut_signals_and_hallucinated():
    ports = hdl.module_ports(FIFO_RTL, "sync_fifo")
    assert hdl.hallucinated_signals(GOOD_TB, ports) == []
    assert hdl.hallucinated_signals(PEEKING_TB, ports) == ["wr_ptr"]
    # Unknown interface must not condemn the testbench.
    assert hdl.hallucinated_signals(PEEKING_TB, []) == []
    # dut._log is a cocotb handle helper, not a signal.
    assert "_log" not in hdl.dut_signals("dut._log.info('x'); dut.clk.value = 1")


def test_validate_testbench_reports_each_problem():
    assert hdl.validate_testbench("", FIFO_RTL, "sync_fifo") == ["empty testbench"]
    problems = hdl.validate_testbench("def broken(:\n  pass", FIFO_RTL, "sync_fifo")
    assert any("syntax" in p for p in problems)
    assert any("@cocotb.test" in p for p in problems)
    problems = hdl.validate_testbench(PEEKING_TB, FIFO_RTL, "sync_fifo")
    assert len(problems) == 1 and "wr_ptr" in problems[0]
    assert hdl.validate_testbench(GOOD_TB, FIFO_RTL, "sync_fifo") == []


# ── scaffold contract ─────────────────────────────────────────────────────────

TESTBENCH_IN = [
    "block_description", "spec", "rtl", "top", "framework", "style",
    "coverage_goals", "extra_tests", "feedback",
]
TESTBENCH_OUT = [
    "testbench", "sva", "test_plan", "top", "explanation", "improvements",
    "status", "errors",
]


@pytest.mark.asyncio
async def test_scaffold_testbench_exact_ports_seeds_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="testbench",
        block_name="fifo tests",
        description="Tests for the FIFO",
        seeds={"top": "sync_fifo", "spec": "8-entry FIFO; writes when full are ignored"},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "testbench"
    assert params["name"] == "fifo_tests"
    assert [p["port_name"] for p in params["input_ports"]] == TESTBENCH_IN
    assert [p["port_name"] for p in params["output_ports"]] == TESTBENCH_OUT
    ins = _ports(params, "input")
    assert ins["top"]["port_content"] == "sync_fifo"
    assert ins["spec"]["port_content"] == "8-entry FIFO; writes when full are ignored"
    assert ins["framework"]["port_content"] == "cocotb"
    assert ins["style"]["port_content"] == "directed+random"
    # category-based layout, like code and the other EDA types
    assert ins["spec"]["port_path"] == "data/testbench/general/fifo_tests/inputs/spec.txt"


@pytest.mark.asyncio
async def test_scaffold_testbench_spec_falls_back_to_description():
    result = await blocks_router.generate_scaffold_payload(
        block_type="testbench", block_name="t", description="A counter that wraps at 15",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["spec"]["port_content"] == "A counter that wraps at 15"


@pytest.mark.asyncio
async def test_testbench_wires_into_verilator():
    tb = await blocks_router.generate_scaffold_payload(block_type="testbench", block_name="t")
    ver = await blocks_router.generate_scaffold_payload(block_type="verilator", block_name="v")
    tb_out = set(_ports(tb["tool_calls"][0]["params"], "output"))
    ver_in = set(_ports(ver["tool_calls"][0]["params"], "input"))
    # testbench.testbench -> verilator.testbench, testbench.top -> verilator.top
    assert {"testbench", "top"} <= tb_out
    assert {"testbench", "top", "rtl"} <= ver_in


# ── AI generation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_testbench_payload_interface_only_in_prompt_and_ports_filled(monkeypatch):
    captured = []

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096):
        captured.append((system_prompt, user_message))
        return {
            "testbench": "```python\n" + GOOD_TB + "\n```",
            "test_plan": [{"name": "test_reset", "intent": "reset", "spec_ref": "reset"}],
            "sva": "",
            "explanation": "Spec-derived tests.",
            "improvements": "- add a wraparound test",
            "top": "sync_fifo",
        }

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    result = await blocks_router.generate_testbench_payload(
        block_name="fifo tests",
        description="Tests for the FIFO",
        spec="8-entry FIFO; writes when full are ignored",
        rtl=FIFO_RTL,
        top="",
    )
    assert len(captured) == 1
    system_prompt, user_message = captured[0]
    assert "verification engineer" in system_prompt.lower()
    # Top inferred from the RTL; interface listed; RTL BODY not leaked to the model.
    assert "Top module: sync_fifo" in user_message
    assert "clk, rst_n, wr_en, wr_data, rd_en, rd_data, full, empty" in user_message
    assert "assign full" not in user_message
    assert "writes when full are ignored" in user_message

    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "testbench"
    assert [p["port_name"] for p in params["input_ports"]] == TESTBENCH_IN
    assert [p["port_name"] for p in params["output_ports"]] == TESTBENCH_OUT
    outs = _ports(params, "output")
    assert "```" not in outs["testbench"]["port_content"]
    assert "@cocotb.test()" in outs["testbench"]["port_content"]
    assert json.loads(outs["test_plan"]["port_content"])[0]["name"] == "test_reset"
    assert outs["status"]["port_content"] == "ok"
    assert outs["top"]["port_content"] == "sync_fifo"
    assert outs["improvements"]["port_content"] == "- add a wraparound test"
    ins = _ports(params, "input")
    assert ins["rtl"]["port_content"] == FIFO_RTL
    assert ins["top"]["port_content"] == "sync_fifo"
    assert ins["framework"]["port_content"] == "cocotb"


@pytest.mark.asyncio
async def test_generate_testbench_payload_repairs_hallucinated_signal_once(monkeypatch):
    calls = []

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096):
        calls.append(user_message)
        tb = PEEKING_TB if len(calls) == 1 else GOOD_TB
        return {"testbench": tb, "test_plan": "[]", "explanation": "", "improvements": "", "top": "sync_fifo"}

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    result = await blocks_router.generate_testbench_payload(
        block_name="t", spec="spec", rtl=FIFO_RTL, top="sync_fifo",
    )
    assert len(calls) == 2
    # The repair round names the offending signal and carries the rejected draft.
    assert "wr_ptr" in calls[1]
    assert "REJECTED" in calls[1]
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "ok"
    assert "wr_ptr" not in outs["testbench"]["port_content"]


@pytest.mark.asyncio
async def test_generate_testbench_payload_surfaces_residual_problems(monkeypatch):
    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096):
        return {"testbench": PEEKING_TB, "test_plan": "[]", "improvements": "x", "top": "sync_fifo"}

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    result = await blocks_router.generate_testbench_payload(
        block_name="t", spec="spec", rtl=FIFO_RTL, top="sync_fifo",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    assert outs["improvements"]["port_content"].startswith("Validation: ")
    assert "wr_ptr" in outs["improvements"]["port_content"]
    assert outs["improvements"]["port_content"].endswith("\nx")
    # The testbench is still delivered so the user can fix it by hand.
    assert "@cocotb.test()" in outs["testbench"]["port_content"]


@pytest.mark.asyncio
async def test_generate_testbench_payload_without_rtl_asks_model_to_infer_interface(monkeypatch):
    seen = {}

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096):
        seen["msg"] = user_message
        return {"testbench": GOOD_TB, "test_plan": "[]", "top": "sync_fifo"}

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    result = await blocks_router.generate_testbench_payload(block_name="t", spec="FIFO spec")
    assert "unknown" in seen["msg"]
    outs = _ports(result["tool_calls"][0]["params"], "output")
    # No interface → no hallucination check → accepted, top echoed from the model.
    assert outs["status"]["port_content"] == "ok"
    assert outs["top"]["port_content"] == "sync_fifo"


@pytest.mark.asyncio
async def test_generate_testbench_payload_feedback_and_previous_reach_the_prompt(monkeypatch):
    seen = {}

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096):
        seen["msg"] = user_message
        return {"testbench": GOOD_TB, "test_plan": "[]"}

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    await blocks_router.generate_testbench_payload(
        block_name="t", spec="spec", rtl=FIFO_RTL, top="sync_fifo",
        feedback="test_reset checks empty one cycle too early",
        previous_testbench="# old\n" + GOOD_TB,
    )
    assert "Reviewer feedback" in seen["msg"]
    assert "one cycle too early" in seen["msg"]
    assert "Previous testbench:" in seen["msg"]


@pytest.mark.asyncio
async def test_generate_testbench_payload_no_key_returns_none(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings(openai=""))
    assert await blocks_router.generate_testbench_payload(block_name="t", spec="s") is None


@pytest.mark.asyncio
async def test_generate_testbench_payload_requires_a_spec(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    with pytest.raises(ValueError):
        await blocks_router.generate_testbench_payload(block_name="t")


def test_simple_testbench_response_shape():
    body = TbRequest(block_name="fifo tests", spec="s", top="sync_fifo")
    params = blocks_router._simple_testbench_response(body, error="AI not configured")["tool_calls"][0]["params"]
    assert params["block_type"] == "testbench"
    assert params["name"] == "fifo_tests"
    assert [p["port_name"] for p in params["input_ports"]] == TESTBENCH_IN
    assert [p["port_name"] for p in params["output_ports"]] == TESTBENCH_OUT
    outs = _ports(params, "output")
    assert outs["errors"]["port_content"] == "AI not configured"
    assert outs["status"]["port_content"] == "error"
    assert outs["testbench"]["port_content"] == ""
    ins = _ports(params, "input")
    assert ins["framework"]["port_content"] == "cocotb"


# ── voice/text enrichment ─────────────────────────────────────────────────────


def test_testbench_is_enrichable():
    assert enrichment.is_enrichable({"type": "create_block", "block_type": "testbench"})


@pytest.mark.asyncio
async def test_enrich_testbench_generates_from_description_as_spec(monkeypatch):
    seen = {}

    async def fake_generate(**kwargs):
        seen.update(kwargs)
        return await blocks_router.generate_scaffold_payload(
            block_type="testbench", block_name=kwargs["block_name"], seeds={"spec": kwargs["spec"]},
        )

    monkeypatch.setattr(blocks_router, "generate_testbench_payload", fake_generate)
    action = {
        "type": "create_block", "block_type": "testbench", "block_name": "fifo_tests",
        "description": "Tests for the 8-entry FIFO", "top": "sync_fifo",
    }
    status = await enrichment._enrich_testbench_block(action, "s1")
    assert status == ("ok", "")
    assert seen["spec"] == "Tests for the 8-entry FIFO"
    assert seen["top"] == "sync_fifo"
    assert [p["port_name"] for p in action["input_ports"]] == TESTBENCH_IN
    assert [p["port_name"] for p in action["output_ports"]] == TESTBENCH_OUT


@pytest.mark.asyncio
async def test_enrich_testbench_falls_back_to_scaffold_without_key(monkeypatch):
    async def none(**kwargs):
        return None

    monkeypatch.setattr(blocks_router, "generate_testbench_payload", none)
    action = {"type": "create_block", "block_type": "testbench", "block_name": "t",
              "description": "spec here", "spec": "explicit spec"}
    status = await enrichment._enrich_testbench_block(action, "s1")
    assert status == ("failed", "AI not configured")
    ins = {p["port_name"]: p["port_content"] for p in action["input_ports"]}
    assert ins["spec"] == "explicit spec"


@pytest.mark.asyncio
async def test_enrich_testbench_without_spec_scaffolds_ok():
    action = {"type": "create_block", "block_type": "testbench", "block_name": "t"}
    status = await enrichment._enrich_testbench_block(action, "s1")
    assert status == ("ok", "")
    assert [p["port_name"] for p in action["output_ports"]] == TESTBENCH_OUT
