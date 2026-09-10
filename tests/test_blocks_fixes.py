"""Tests for the failure-triage endpoint that fills the fix_rtl / fix_tb / fix_spec ports.

Two contracts run through every case here, and they are what make this endpoint
different from its sibling ``/blocks/run/improvements``:

* **It never raises, and it never puts prose on a port.**  The run has already
  finished and already been reported.  Worse, ``fix_rtl`` is read straight back
  out as an instruction to the design block, so an error sentence written there
  would become a MANDATORY INSTRUCTION to implement an HTTP 503.  Every failure
  path therefore returns HTTP 200 with EMPTY buckets.
* **A failing test belongs to exactly one bucket.**  Three different blocks read
  these three ports and each does what its port says; a failure routed to two of
  them gets "fixed" in two artifacts at once and the next run is worse.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import FixesRequest
from app.prompts import get_system_prompt

FIFO_RTL = """
module sync_fifo #(parameter WIDTH = 16, DEPTH = 8) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             wr_en,
    output wire             full,
    output wire             empty
);
    reg [3:0] wr_ptr, rd_ptr;
    assign full  = (wr_ptr - rd_ptr) == DEPTH - 1;
    assign empty = wr_ptr == rd_ptr;
endmodule
"""

FIFO_TB = """
import cocotb
from cocotb.triggers import RisingEdge

@cocotb.test()
async def test_full_flag_asserts_at_depth(dut):
    assert int(dut.full.value) == 1
"""

FAILING_RUN = {
    "status": "error",
    "passed": "false",
    "failures": (
        "1 of 6 cocotb tests failed.\n"
        "FAILED test_full_flag_asserts_at_depth\n"
        "  WHY: AssertionError: full must assert after 8 writes (expected 1, got 0)\n"
        "  WHERE: test_sync_fifo.py:76\n"
    ),
    "results": '{"total": 6, "passed": 5, "failed": 1, "skipped": 0}',
    "errors": "",
    "lint": "Warning-WIDTH: sync_fifo.v:11: Operator EQ expects 4 bits",
    "rtl": FIFO_RTL,
    "testbench": FIFO_TB,
    "spec": "REQ-3: `full` shall assert once DEPTH entries are held.",
    "top": "sync_fifo",
    "iterations": '[{"i": 1, "passed": false, "failed": 1}]',
}

PASSING_RUN = {
    "status": "ok",
    "passed": "true",
    "results": '{"total": 6, "passed": 6, "failed": 0, "skipped": 0}',
    "rtl": FIFO_RTL,
    "testbench": FIFO_TB,
    "top": "sync_fifo",
}


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(openai_api_key=openai, anthropic_api_key=anthropic,
                           openai_model="gpt-test")


def _capture():
    """A fake _call_openai_json that records what it was handed."""
    seen: dict[str, object] = {"calls": 0}

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None,
                       max_tokens=4096):
        seen["calls"] = int(seen["calls"]) + 1
        seen["system"] = system_prompt
        seen["user"] = user_message
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        seen["temperature"] = temperature
        return {
            "fix_rtl": (
                "1. In `sync_fifo`, compare `(wr_ptr - rd_ptr) == DEPTH`, not "
                "`DEPTH - 1`.\n"
                "   [must appear: (wr_ptr - rd_ptr) == DEPTH]\n"
                "   Fixes: test_full_flag_asserts_at_depth"
            ),
            "fix_tb": "",
            "fix_spec": "",
            "summary": "one off-by-one in the `full` comparison",
        }

    return fake_llm, seen


def _run(coro):
    return asyncio.run(coro)


# ── the prompt section ────────────────────────────────────────────────────────


def test_triage_section_exists():
    prompt = get_system_prompt("triage_failures")
    assert prompt
    assert "verilator" in prompt


def test_triage_section_declares_the_four_output_keys():
    prompt = get_system_prompt("triage_failures")
    for key in ('"fix_rtl"', '"fix_tb"', '"fix_spec"', '"summary"'):
        assert key in prompt


def test_triage_section_pins_the_entry_wire_format():
    """The numbered entry, its optional literal hint and its attribution line.

    ``[must appear: ...]`` is not decoration: directives.extract_directives mines
    it into a checkable KIND_LITERAL, and ``Fixes:`` is what carries the
    single-attribution rule into something a reader can audit.
    """
    prompt = get_system_prompt("triage_failures")
    assert "[must appear:" in prompt
    assert "Fixes:" in prompt
    assert "numbered from 1" in prompt


def test_triage_section_states_the_single_attribution_rule():
    prompt = get_system_prompt("triage_failures")
    assert "EXACTLY ONE bucket" in prompt
    assert "Do not put the same test name on two" in prompt


def test_triage_section_carries_the_fix_tb_allow_list():
    """A test may be called wrong only for reasons that do not depend on the design."""
    prompt = get_system_prompt("triage_failures")
    assert "THE ONLY ADMISSIBLE REASONS" in prompt
    assert "not in the design's port list" in prompt
    assert "cocotb 1.x API" in prompt


def test_triage_section_forbids_weakening_a_test():
    """The failure mode this whole feature could otherwise introduce.

    A loop that is allowed to edit its own judge converges on a green run by
    deleting the only independent check anyone has.
    """
    prompt = get_system_prompt("triage_failures")
    assert "Relax, weaken or delete an assertion" in prompt
    assert "Change an expected value to whatever the design happens to produce" in prompt


def test_triage_section_keeps_the_spec_bucket_clause_addressed():
    prompt = get_system_prompt("triage_failures")
    assert "REQ-4" in prompt
    assert "Never renumber" in prompt


def test_triage_section_wants_everything_empty_when_nothing_can_be_fixed():
    prompt = get_system_prompt("triage_failures")
    assert "WHEN EVERY BUCKET IS EMPTY" in prompt
    # A build error is NOT the environmental case: it is located and it is an
    # RTL edit like any other.
    assert "A build error is NOT environmental" in prompt


def test_triage_section_writes_no_code():
    prompt = get_system_prompt("triage_failures")
    assert "Do not return code, a diff, a patch or a corrected file" in prompt


def test_triage_section_does_not_duplicate_the_review_ports():
    """The two port families must not be the same answer written twice."""
    prompt = get_system_prompt("triage_failures")
    assert "improvements_rtl" in prompt
    assert "Do not order refactors" in prompt


# ── the payload generator ─────────────────────────────────────────────────────


def test_payload_returns_every_bucket(monkeypatch):
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    out = _run(blocks_router.generate_fixes_payload(
        block_name="fifo_sim", kind="verilator", verdict="failed", run=FAILING_RUN))
    assert set(out) == {"fix_rtl", "fix_tb", "fix_spec", "summary"}
    assert seen["system"] == get_system_prompt("triage_failures")
    assert "VERDICT: failed" in str(seen["user"])
    assert out["fix_rtl"]
    # Two of three empty is the normal, correct answer.
    assert out["fix_tb"] == "" and out["fix_spec"] == ""


def test_a_passing_run_is_never_sent_to_the_model(monkeypatch):
    """The short-circuit, and why it is on the SERVER as well as in the app.

    A passing run has nothing to repair, and billing one for a repair order is a
    cost the user cannot see and did not ask for.
    """
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    out = _run(blocks_router.generate_fixes_payload(
        block_name="fifo_sim", kind="verilator", verdict="passed", run=PASSING_RUN))
    assert seen["calls"] == 0
    assert out == {"fix_rtl": "", "fix_tb": "", "fix_spec": "", "summary": ""}


def test_an_absent_verdict_is_also_short_circuited(monkeypatch):
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    out = _run(blocks_router.generate_fixes_payload(block_name="b", verdict="", run={}))
    assert seen["calls"] == 0
    assert out["fix_rtl"] == ""


def test_the_short_circuit_beats_the_missing_key(monkeypatch):
    """A passing run answers with empties, not with "no AI key configured".

    Nothing was wanted, so nothing missing is an outage.
    """
    monkeypatch.setattr(blocks_router, "get_settings",
                        lambda: _fake_settings(openai="", anthropic=""))
    out = _run(blocks_router.generate_fixes_payload(
        block_name="b", verdict="passed", run=PASSING_RUN))
    assert out is not None
    assert out["fix_rtl"] == ""


def test_payload_is_none_without_a_key(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings",
                        lambda: _fake_settings(openai="", anthropic=""))
    out = _run(blocks_router.generate_fixes_payload(
        block_name="b", verdict="failed", run=FAILING_RUN))
    assert out is None


def test_payload_uses_anthropic_key_alone(monkeypatch):
    fake_llm, _ = _capture()
    monkeypatch.setattr(blocks_router, "get_settings",
                        lambda: _fake_settings(openai="", anthropic="sk-ant-test"))
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)
    out = _run(blocks_router.generate_fixes_payload(
        block_name="b", verdict="failed", run=FAILING_RUN))
    assert out is not None


def test_missing_model_keys_become_empty_strings(monkeypatch):
    async def partial(*args, **kwargs):
        return {"fix_rtl": "1. do the thing\n   Fixes: test_x"}

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", partial)
    out = _run(blocks_router.generate_fixes_payload(
        block_name="b", verdict="failed", run=FAILING_RUN))
    assert out["fix_tb"] == "" and out["fix_spec"] == "" and out["summary"] == ""


def test_the_triage_is_colder_and_cheaper_than_the_review(monkeypatch):
    """A repair order is not an opinion, and it carries no per-test diagnosis."""
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)
    _run(blocks_router.generate_fixes_payload(
        block_name="b", verdict="failed", run=FAILING_RUN))
    assert seen["temperature"] == 0.2
    assert seen["max_tokens"] == 4096


def test_the_evidence_carries_the_design_the_tests_and_the_contract(monkeypatch):
    """All three artifacts, or the attribution decision cannot be made at all."""
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)
    _run(blocks_router.generate_fixes_payload(
        block_name="b", kind="verilator", verdict="failed", run=FAILING_RUN))
    user = str(seen["user"])
    for label in ("--- rtl ---", "--- testbench ---", "--- spec ---",
                  "--- failures ---", "--- iterations ---"):
        assert label in user


# ── the endpoint ──────────────────────────────────────────────────────────────


def test_endpoint_returns_ok_envelope(monkeypatch):
    fake_llm, _ = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    body = FixesRequest(block_name="fifo_sim", kind="verilator",
                        verdict="failed", run=FAILING_RUN)
    out = _run(blocks_router.run_fixes(body, user=None))
    assert out["status"] == "ok"
    assert out["errors"] == ""
    assert out["fix_rtl"] and out["summary"]


def test_endpoint_is_graceful_without_a_key(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings",
                        lambda: _fake_settings(openai="", anthropic=""))
    body = FixesRequest(block_name="fifo_sim", kind="verilator",
                        verdict="failed", run=FAILING_RUN)
    out = _run(blocks_router.run_fixes(body, user=None))
    assert out["status"] == "error"
    assert out["fix_rtl"] == "" and out["fix_tb"] == "" and out["fix_spec"] == ""
    assert "no AI key" in out["errors"]


def test_endpoint_is_graceful_when_the_model_raises(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", boom)

    body = FixesRequest(block_name="fifo_sim", kind="verilator",
                        verdict="failed", run=FAILING_RUN)
    out = _run(blocks_router.run_fixes(body, user=None))
    assert out["status"] == "error"
    assert "upstream 503" in out["errors"]
    # The error text goes in `errors`, NEVER into a bucket: the app writes these
    # buckets to ports that feed other blocks' feedback inputs.
    assert out["fix_rtl"] == ""


def test_every_error_envelope_carries_every_bucket():
    assert set(blocks_router._EMPTY_FIXES) == set(blocks_router._FIXES_KEYS)
    assert all(v == "" for v in blocks_router._EMPTY_FIXES.values())


def test_run_llm_model_is_forwarded(monkeypatch):
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)
    body = FixesRequest(block_name="b", kind="verilator", verdict="failed",
                        run=FAILING_RUN, run_llm_model="claude-opus-5")
    _run(blocks_router.run_fixes(body, user=None))
    assert seen["model"] == "claude-opus-5"


# ── the port contract ─────────────────────────────────────────────────────────


def test_verilator_scaffold_declares_all_three_fix_ports():
    """Mirrors EdaPorts::kVerilatorOutputs in
    Grafux-app/src/clients/grafux-devices/edaports.h."""
    outputs = blocks_router._SCAFFOLD_SPECS["verilator"].outputs
    for name in ("fix_rtl", "fix_tb", "fix_spec"):
        assert name in outputs


def test_the_fix_ports_are_outputs_only():
    """Nothing wires INTO them; they are the block's answer, not a setting."""
    spec = blocks_router._SCAFFOLD_SPECS["verilator"]
    for name in ("fix_rtl", "fix_tb", "fix_spec"):
        assert name not in spec.inputs
        assert name not in spec.defaults


def test_the_review_ports_survive_alongside_them():
    """Coexistence is the whole design: a review is not a repair order."""
    outputs = blocks_router._SCAFFOLD_SPECS["verilator"].outputs
    for name in ("improvements_rtl", "improvements_test", "improvements_spec"):
        assert name in outputs


@pytest.mark.parametrize("block_type", ["yosys", "openroad", "devices"])
def test_no_other_block_type_has_fix_ports(block_type):
    """Only verilator runs tests, so only verilator can attribute a failure."""
    spec = blocks_router._SCAFFOLD_SPECS.get(block_type)
    if spec is None:
        pytest.skip(f"no scaffold for {block_type}")
    assert not any(o.startswith("fix_") for o in spec.outputs)


def test_triage_section_keeps_each_order_on_one_line():
    """directives.extract_directives mines one instruction PER LINE.

    An order wrapped across two lines arrives at code_hdl / testbench / spec_hdl
    as two unrelated MANDATORY INSTRUCTIONS, the second of which is a sentence
    fragment nobody wrote.
    """
    prompt = get_system_prompt("triage_failures")
    assert "Keep each numbered order on ONE line" in prompt
    assert "one instruction PER LINE" in prompt
