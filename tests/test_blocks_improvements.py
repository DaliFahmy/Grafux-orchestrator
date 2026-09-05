"""Tests for the run-review endpoint that fills the improvements ports.

The contract under test is unusual in one way that every case here leans on:
this endpoint runs AFTER the run has already been reported to the user, so it
must NEVER raise and must NEVER return a shape the app would treat as a failure
of the run itself.  A review outage leaves the port alone; it does not redden a
green block.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import ImprovementsRequest
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
    assign full  = (wr_ptr - rd_ptr) == DEPTH;
    assign empty = wr_ptr == rd_ptr;
endmodule
"""

FAILING_RUN = {
    "status": "ok",
    "passed": "false",
    "failures": "test_wrap_around: expected rd_data 0x00, got 0x01\ntest_full_flag: full never asserted",
    "results": '{"total": 6, "passed": 4, "failed": 2, "skipped": 0}',
    "coverage": '{"lines": {"pct": 41.2}}',
    "lint": "Warning-WIDTH: sync_fifo.v:11: Operator EQ expects 4 bits",
    "rtl": FIFO_RTL,
    "top": "sync_fifo",
}

PASSING_RUN = {
    "status": "ok",
    "passed": "true",
    "results": '{"total": 6, "passed": 6, "failed": 0, "skipped": 0}',
    "coverage": '{"lines": {"pct": 41.2}}',
    "rtl": FIFO_RTL,
    "top": "sync_fifo",
}

OPENROAD_RUN = {
    "status": "ok",
    "stage": "route",
    "metrics": '{"worst_slack_ns": -0.42, "utilization": 0.81, "drc_violations": 3}',
    "reports": "5_route_drc.rpt\n4_cts_timing.rpt",
}


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(openai_api_key=openai, anthropic_api_key=anthropic, openai_model="gpt-test")


def _capture():
    """A fake _call_openai_json that records the messages it was handed."""
    seen: dict[str, str] = {}

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None, max_tokens=4096):
        seen["system"] = system_prompt
        seen["user"] = user_message
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return {
            "design": "- `wr_ptr` wraps one cycle early (test_wrap_around)",
            "tests": "- Add a test that writes while `full` is high",
            "spec": "- REQ-3 does not say whether `full` blocks a same-cycle write",
            "summary": "2 tests fail on the pointer wrap",
        }

    return fake_llm, seen


def _run(coro):
    return asyncio.run(coro)


# ── the prompt section ────────────────────────────────────────────────────────


def test_improve_run_section_exists_and_covers_every_kind():
    prompt = get_system_prompt("improve_run")
    assert prompt, "improve_run section missing from Msg_config"
    for kind in ("verilator", "openroad", "device"):
        assert kind in prompt, f"prompt gives no lens for kind={kind}"


def test_improve_run_section_declares_the_four_output_keys():
    prompt = get_system_prompt("improve_run")
    for key in ("design", "tests", "spec", "summary"):
        assert f'"{key}"' in prompt


def test_improve_run_keeps_the_spec_bucket_off_the_kinds_with_no_contract():
    # openroad and device have no spec_hdl block upstream, so a spec bullet for
    # them would be advice about a file that does not exist.
    prompt = get_system_prompt("improve_run")
    assert "no contract block on the canvas" in prompt


def test_improve_run_allows_an_empty_spec_bucket():
    # The one bucket that is CORRECTLY empty most of the time - the general
    # "never return empty buckets" rule must carve it out, or every run reports
    # a contract fault and the outer loop revises the spec on nothing.
    prompt = get_system_prompt("improve_run")
    assert '"spec" is the one exception' in prompt


def test_improve_run_forbids_an_empty_review_of_a_passing_run():
    # The single most likely way this feature disappoints: a model handed a green
    # run writes nothing. The prohibition is load-bearing, so it is pinned.
    prompt = get_system_prompt("improve_run")
    assert "Do not return empty buckets for a passing run" in prompt
    assert 'NEVER say "looks good" and stop' in prompt


# ── message building (pure) ───────────────────────────────────────────────────


def test_message_carries_verdict_failures_and_rtl():
    msg = blocks_router.build_improvements_message(
        kind="verilator", block_description="8-entry sync FIFO",
        verdict="failed", run=FAILING_RUN,
    )
    assert "KIND: verilator" in msg
    assert "VERDICT: failed" in msg
    assert "DESCRIPTION: 8-entry sync FIFO" in msg
    assert "test_wrap_around" in msg
    assert "sync_fifo" in msg
    # Every section is labelled with its PORT NAME - that label is what the
    # prompt tells the model to cite as evidence.
    assert "--- failures ---" in msg
    assert "--- rtl ---" in msg


def test_passing_run_says_passed_and_keeps_coverage():
    # The prompt's pass branch is only reachable if the verdict actually arrives,
    # and coverage is the evidence that branch is built on.
    msg = blocks_router.build_improvements_message(
        kind="verilator", verdict="passed", run=PASSING_RUN
    )
    assert "VERDICT: passed" in msg
    assert "--- coverage ---" in msg
    assert "41.2" in msg


def test_openroad_message_carries_metrics_and_no_verilator_labels():
    msg = blocks_router.build_improvements_message(
        kind="openroad", verdict="passed", run=OPENROAD_RUN
    )
    assert "KIND: openroad" in msg
    assert "--- metrics ---" in msg
    assert "worst_slack_ns" in msg
    assert "--- failures ---" not in msg
    assert "--- coverage ---" not in msg


def test_empty_sections_are_omitted_entirely():
    msg = blocks_router.build_improvements_message(
        kind="device", verdict="passed",
        run={"command": "./a.out", "errors": "", "warnings": "   "},
    )
    assert "--- command ---" in msg
    assert "--- errors ---" not in msg
    assert "--- warnings ---" not in msg


def test_oversized_evidence_is_capped_and_drops_the_least_specific_first():
    msg = blocks_router.build_improvements_message(
        kind="verilator", verdict="failed",
        run={
            "failures": "test_wrap_around: KEEP THIS",
            "results": '{"total": 6}',
            "log": "L" * 120_000,
            "sim_output": "S" * 120_000,
        },
    )
    assert len(msg) <= blocks_router._IMPROVEMENTS_MAX_CHARS
    # The sections the review is actually built from survive; the bulk goes.
    assert "KEEP THIS" in msg
    assert "--- results ---" in msg
    assert "--- log ---" not in msg
    assert "--- sim_output ---" not in msg


def test_hard_truncation_marks_itself_when_evidence_cannot_be_dropped():
    # `rtl` is not droppable, so an absurd one must still be cut - and say so,
    # because the prompt tells the model not to conclude anything from the cut.
    msg = blocks_router.build_improvements_message(
        kind="verilator", verdict="failed", run={"rtl": "R" * 200_000}
    )
    assert len(msg) <= blocks_router._IMPROVEMENTS_MAX_CHARS + len("\n...[truncated]")
    assert "[truncated]" in msg


# ── payload + endpoint ────────────────────────────────────────────────────────


def test_payload_returns_the_four_buckets(monkeypatch):
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    out = _run(blocks_router.generate_improvements_payload(
        block_name="fifo_sim", kind="verilator", verdict="failed", run=FAILING_RUN,
    ))
    assert set(out) == {"design", "tests", "spec", "summary"}
    assert "wr_ptr" in out["design"]
    assert "full" in out["tests"]
    assert "REQ-3" in out["spec"]
    assert seen["system"] == get_system_prompt("improve_run")
    assert "VERDICT: failed" in seen["user"]


def test_payload_is_none_without_a_key(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic=""))
    out = _run(blocks_router.generate_improvements_payload(
        block_name="fifo_sim", kind="verilator", run=FAILING_RUN,
    ))
    assert out is None


def test_payload_uses_anthropic_key_alone(monkeypatch):
    fake_llm, _ = _capture()
    monkeypatch.setattr(blocks_router, "get_settings",
                        lambda: _fake_settings(openai="", anthropic="sk-ant-test"))
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)
    out = _run(blocks_router.generate_improvements_payload(block_name="b", kind="verilator"))
    assert out is not None


def test_endpoint_returns_ok_envelope(monkeypatch):
    fake_llm, _ = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    body = ImprovementsRequest(block_name="fifo_sim", kind="verilator",
                               verdict="failed", run=FAILING_RUN)
    out = _run(blocks_router.run_improvements(body, user=None))
    assert out["status"] == "ok"
    assert out["errors"] == ""
    assert out["design"] and out["tests"] and out["summary"]


def test_endpoint_is_graceful_without_a_key(monkeypatch):
    # HTTP 200 with empty buckets, NOT a raise: the run already succeeded.
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic=""))
    body = ImprovementsRequest(block_name="fifo_sim", kind="verilator", run=FAILING_RUN)
    out = _run(blocks_router.run_improvements(body, user=None))
    assert out["status"] == "error"
    assert out["design"] == "" and out["tests"] == ""
    assert "no AI key" in out["errors"]


def test_endpoint_is_graceful_when_the_model_raises(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", boom)

    body = ImprovementsRequest(block_name="fifo_sim", kind="verilator", run=FAILING_RUN)
    out = _run(blocks_router.run_improvements(body, user=None))
    assert out["status"] == "error"
    assert "upstream 503" in out["errors"]
    assert out["design"] == ""


def test_unknown_kind_is_accepted_not_rejected(monkeypatch):
    # The app is the authority on kinds. A 422 here would fail a run that worked.
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    body = ImprovementsRequest(block_name="b", kind="klingon", run={"errors": "x"})
    out = _run(blocks_router.run_improvements(body, user=None))
    assert out["status"] == "ok"
    assert "KIND: klingon" in seen["user"]


def test_missing_model_keys_become_empty_strings(monkeypatch):
    async def sparse(*args, **kwargs):
        return {"design": "- do the thing"}      # no "tests"/"spec"/"summary"

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", sparse)

    out = _run(blocks_router.generate_improvements_payload(block_name="b", kind="verilator"))
    assert out == {"design": "- do the thing", "tests": "", "spec": "", "summary": ""}


def test_run_llm_model_is_forwarded(monkeypatch):
    fake_llm, seen = _capture()
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_llm)

    body = ImprovementsRequest(block_name="b", kind="verilator",
                               run={"errors": "x"}, run_llm_model="claude-opus-5")
    _run(blocks_router.run_improvements(body, user=None))
    assert seen["model"] == "claude-opus-5"


# ── the port contract this endpoint writes into ───────────────────────────────


def test_verilator_scaffold_declares_all_three_improvement_ports():
    """Mirrors EdaPorts::kVerilatorOutputs in
    Grafux-app/src/clients/grafux-devices/edaports.h. A drift here means an
    AI-created block and a toolbar-created block get different ports.
    """
    outputs = blocks_router._SCAFFOLD_SPECS["verilator"].outputs
    assert "improvements_rtl" in outputs
    assert "improvements_test" in outputs
    # The contract advice, and the outer loop's return leg into spec_hdl.feedback.
    assert "improvements_spec" in outputs
    # The single generic port is GONE on verilator - it was dead, and keeping it
    # beside the three real ones would leave a fourth port that never fills.
    assert "improvements" not in outputs


def test_verilator_scaffold_takes_the_spec_as_a_wire_only_input():
    """`spec` is evidence for the review, never a run parameter.

    It must be an input port (so spec_hdl.spec can be wired in) and must NOT be
    seeded or defaulted: a seeded verilator spec is a SECOND copy of the
    contract, and a review citing a contract nobody implemented is worse than
    no review.
    """
    spec = blocks_router._SCAFFOLD_SPECS["verilator"]
    assert "spec" in spec.inputs
    assert "spec" not in spec.outputs        # never echoed through
    assert "spec" not in (spec.seed_map or {}).values()
    assert "spec" not in (spec.defaults or {})


def test_openroad_and_devices_keep_the_single_improvements_port():
    assert "improvements" in blocks_router._SCAFFOLD_SPECS["openroad"].outputs
    assert "improvements" in blocks_router._SCAFFOLD_SPECS["devices"].outputs


def test_yosys_has_no_improvements_port():
    # Nothing reviews a synthesis run, so it must not pay for one.
    outputs = blocks_router._SCAFFOLD_SPECS["yosys"].outputs
    assert not any(o.startswith("improvements") for o in outputs)
