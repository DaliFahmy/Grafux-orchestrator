"""The analogue_simulator block's orchestrator side: the SPICE netlist validator,
the netlist generator's repair loop and fallback, and the block's port contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.core.constants import EXPENSIVE_BLOCK_TYPES, BlockType
from app.modules.blocks import router as blocks_router
from app.modules.blocks import spice
from app.modules.blocks.schemas import (
    AnalogueNetlistGenerateRequest as NetlistRequest,  # alias: pytest must not collect it
)
from app.modules.session import enrichment
from app.prompts import get_system_prompt

# The decks below are the ones scripts/e2e_ngspice_smoke.py in Grafux-devices
# simulates for real on the ngspice image -- they are known to load and converge.
SKY130_INVERTER = """sky130 inverter
Vdd vdd 0 {vdd}
Vin in 0 PULSE(0 {vdd} 0 50p 50p 1n 2n)
XM1 out in 0 0 sky130_fd_pr__nfet_01v8 W=1 L=0.15
XM2 out in vdd vdd sky130_fd_pr__pfet_01v8 W=2 L=0.15
C1 out 0 10f
.tran 1p 4n
.end
"""

GF180_INVERTER = """gf180 inverter
Vdd vdd 0 {vdd}
Vin in 0 0
M1 out in 0 0 nfet_03v3 W=1u L=0.28u
M2 out in vdd vdd pfet_03v3 W=2u L=0.28u
.dc Vin 0 3.3 0.01
.end
"""

RC_GENERIC = """rc low pass
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1u
.ac dec 20 1 1meg
.end
"""

# Mirrors EdaPorts::kAnalogueSimInputs / kAnalogueSimOutputs in the Qt app.
_ANALOGUE_INPUTS = {
    "block_description", "netlist", "pdk", "corner", "temperature", "supply_voltage",
    "analyses", "meas_statements", "probes", "max_points", "extra_control", "files",
    "timeout", "instance_type", "image", "api_keys",
}
_ANALOGUE_OUTPUTS = {
    "status", "netlist", "measurements", "waveforms", "operating_point", "analyses",
    "stats", "errors", "warnings", "log", "raw", "artifacts", "eda_id", "cost",
    "improvements",
}


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(openai_api_key=openai, anthropic_api_key=anthropic,
                           openai_model="gpt-test")


def _responder(*payloads):
    calls = []

    async def fake_llm(system_prompt, user_message, temperature=0.3, *, model=None,
                       max_tokens=4096):
        calls.append((system_prompt, user_message))
        return payloads[min(len(calls) - 1, len(payloads) - 1)]

    return fake_llm, calls


def _ports(params, side):
    return {p["port_name"]: p for p in params[f"{side}_ports"]}


# ── validator (pure) ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("deck, pdk, supply", [
    (SKY130_INVERTER, "sky130A", "1.8"),
    (GF180_INVERTER, "gf180mcuD", "3.3"),
    (RC_GENERIC, "none", ""),
])
def test_known_good_decks_pass(deck, pdk, supply):
    assert spice.validate_netlist(deck, pdk, supply_voltage=supply) == []


def test_an_m_line_in_sky130_names_the_subcircuit_syntax():
    deck = SKY130_INVERTER.replace("XM1 out in 0 0 sky130_fd_pr__nfet_01v8",
                                   "M1 out in 0 0 nmos")
    problems = spice.validate_netlist(deck, "sky130A", supply_voltage="1.8")
    assert any("XM1" in p and "sky130_fd_pr__nfet_01v8" in p for p in problems)


def test_sky130_devices_in_a_gf180_deck_are_refused():
    problems = spice.validate_netlist(SKY130_INVERTER, "gf180mcuD", supply_voltage="3.3")
    assert any("nfet_03v3" in p for p in problems)


def test_a_gf180_mosfet_written_as_a_subcircuit_is_refused():
    deck = GF180_INVERTER.replace("M1 out in 0 0 nfet_03v3", "XM1 out in 0 0 nfet_03v3")
    problems = spice.validate_netlist(deck, "gf180mcuD", supply_voltage="3.3")
    assert any("MOSFET model" in p for p in problems)


def test_an_unknown_gf180_model_is_refused():
    deck = GF180_INVERTER.replace("nfet_03v3", "nfet_33v0")
    assert any("nfet_33v0" in p for p in
               spice.validate_netlist(deck, "gf180mcuD", supply_voltage="3.3"))


def test_the_server_owned_lines_are_refused():
    deck = SKY130_INVERTER.replace(".tran 1p 4n", '.lib "sky130.lib.spice" tt\n.control\nrun\n.endc')
    problems = spice.validate_netlist(deck, "sky130A", analyses="tran 1p 4n", supply_voltage="1.8")
    assert any(".lib" in p for p in problems)
    assert any(".control" in p for p in problems)


def test_no_analysis_anywhere_is_refused_but_the_port_satisfies_it():
    deck = RC_GENERIC.replace(".ac dec 20 1 1meg\n", "")
    assert any("no analysis" in p for p in spice.validate_netlist(deck, "none"))
    assert spice.validate_netlist(deck, "none", analyses="ac dec 20 1 1meg") == []


def test_an_undefined_subcircuit_is_refused_and_a_defined_one_is_not():
    deck = "t\nX1 a b amp\nV1 a 0 1\n.op\n.end\n"
    assert any("amp" in p for p in spice.validate_netlist(deck, "none"))
    defined = "t\n.subckt amp a b\nR1 a b 1k\n.ends\nX1 a b amp\nV1 a 0 1\n.op\n.end\n"
    assert spice.validate_netlist(defined, "none") == []


def test_unbalanced_subckt():
    deck = "t\n.subckt amp a b\nR1 a b 1k\nV1 a 0 1\n.op\n.end\n"
    assert any(".ends" in p for p in spice.validate_netlist(deck, "none"))


def test_vdd_needs_a_value_from_somewhere():
    assert any("{vdd}" in p for p in spice.validate_netlist(SKY130_INVERTER, "sky130A"))
    with_param = SKY130_INVERTER.replace("Vdd vdd", ".param vdd=1.8\nVdd vdd")
    assert spice.validate_netlist(with_param, "sky130A") == []


def test_a_generic_mosfet_needs_a_model_card():
    deck = "t\nV1 d 0 1\nM1 d d 0 0 nmos1\n.op\n.end\n"
    assert any(".model" in p for p in spice.validate_netlist(deck, "none"))
    assert spice.validate_netlist(deck.replace(".op", ".model nmos1 nmos level=1\n.op"),
                                  "none") == []


def test_a_deck_starting_with_a_card_has_no_title():
    assert any("title" in p for p in spice.validate_netlist(".op\nR1 a 0 1k\n.end\n", "none"))


def test_continuation_lines_are_joined():
    deck = "t\nXM1 out in 0 0\n+ sky130_fd_pr__nfet_01v8 W=1 L=0.15\nV1 in 0 1\n.op\n.end\n"
    assert spice.validate_netlist(deck, "sky130A") == []


def test_extract_spice_strips_fences():
    assert spice.extract_spice("```spice\nt\nR1 a 0 1\n```\n") == "t\nR1 a 0 1\n"


def test_pdk_aliases_match_the_devices_server():
    assert spice.normalize_pdk("SKY130") == "sky130A"
    assert spice.normalize_pdk("sky130hd") == "sky130A"
    assert spice.normalize_pdk("gf180") == "gf180mcuD"


# ── generator ─────────────────────────────────────────────────────────────────


def test_the_prompt_section_exists_and_forbids_the_server_owned_lines():
    prompt = get_system_prompt("create_analogue_netlist")
    assert prompt
    for needle in ("sky130_fd_pr__nfet_01v8", "nfet_03v3", ".control", "{vdd}"):
        assert needle in prompt


@pytest.mark.asyncio
async def test_a_valid_first_draft_is_returned_without_repair(monkeypatch):
    fake, calls = _responder({
        "netlist": SKY130_INVERTER, "analyses": "tran 1p 4n",
        "meas_statements": ".meas tran tpd TRIG v(in) VAL=0.9 RISE=1 TARG v(out) VAL=0.9 FALL=1",
        "probes": "v(in) v(out)", "explanation": "an inverter", "improvements": "10f load assumed",
    })
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_analogue_netlist_payload(NetlistRequest(
        block_name="inv", description="a sky130 inverter, measure tpd", supply_voltage="1.8"))
    assert len(calls) == 1
    assert result["status"] == "ok"
    assert result["netlist"] == SKY130_INVERTER
    assert result["probes"] == "v(in) v(out)"
    assert "PDK: sky130A" in calls[0][1]


@pytest.mark.asyncio
async def test_a_rejected_draft_is_repaired_with_the_reasons(monkeypatch):
    bad = SKY130_INVERTER.replace("XM1 out in 0 0 sky130_fd_pr__nfet_01v8", "M1 out in 0 0 nmos")
    fake, calls = _responder({"netlist": bad}, {"netlist": SKY130_INVERTER})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_analogue_netlist_payload(NetlistRequest(
        block_name="inv", description="inverter", supply_voltage="1.8"))
    assert len(calls) == 2
    assert "REJECTED" in calls[1][1] and "XM1" in calls[1][1]
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_problems_that_survive_every_round_are_surfaced_not_hidden(monkeypatch):
    bad = "t\nM1 d g 0 0 nmos\n.op\n.end\n"
    fake, calls = _responder({"netlist": bad, "improvements": "x"})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_analogue_netlist_payload(NetlistRequest(
        block_name="n", description="nmos"))
    assert len(calls) == 1 + blocks_router._ANALOGUE_REPAIR_ROUNDS
    assert result["status"] == "needs_review"
    assert result["improvements"].startswith("Validation:")
    assert result["netlist"] == bad


@pytest.mark.asyncio
async def test_the_users_ports_win_over_the_models_proposals(monkeypatch):
    fake, calls = _responder({"netlist": RC_GENERIC.replace(".ac dec 20 1 1meg\n", ""),
                              "analyses": "op", "probes": "v(in)"})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    result = await blocks_router.generate_analogue_netlist_payload(NetlistRequest(
        block_name="rc", description="rc", pdk="none", analyses="ac dec 20 1 1meg",
        probes="v(out)"))
    assert result["analyses"] == "ac dec 20 1 1meg"
    assert result["probes"] == "v(out)"
    assert "keep them" in calls[0][1]


@pytest.mark.asyncio
async def test_a_revision_offers_the_existing_deck_and_the_request(monkeypatch):
    fake, calls = _responder({"netlist": RC_GENERIC})
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake)
    await blocks_router.generate_analogue_netlist_payload(NetlistRequest(
        block_name="rc", pdk="none", previous_netlist=RC_GENERIC, feedback="make R1 2k"))
    assert "Current netlist" in calls[0][1] and "make R1 2k" in calls[0][1]


@pytest.mark.asyncio
async def test_no_key_falls_back_without_inventing_a_netlist(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings(openai=""))
    result = await blocks_router.generate_analogue_netlist(
        NetlistRequest(block_name="x", description="an amplifier"), user=None)
    assert result["status"] == "error"
    assert result["netlist"] == ""
    assert "AI not configured" in result["errors"]


@pytest.mark.asyncio
async def test_no_description_is_an_error_not_a_guess(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    result = await blocks_router.generate_analogue_netlist(
        NetlistRequest(block_name="x"), user=None)
    assert result["status"] == "error"
    assert "block_description" in result["errors"]


# ── block contract ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scaffold_analogue_simulator_exact_ports_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="analogue_simulator", block_name="inv delay",
        description="Delay of a sky130 inverter", seeds={"pdk": "gf180mcuD"},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "analogue_simulator"
    ins, outs = _ports(params, "input"), _ports(params, "output")
    assert set(ins) == _ANALOGUE_INPUTS
    assert set(outs) == _ANALOGUE_OUTPUTS
    assert ins["pdk"]["port_content"] == "gf180mcuD"
    assert ins["max_points"]["port_content"] == "2000"
    assert ins["netlist"]["port_content"] == ""
    assert outs["waveforms"]["port_path"] == \
        "data/analogue_simulator/general/inv_delay/outputs/waveforms.txt"


def test_the_measure_lines_and_their_values_do_not_share_a_name():
    spec = blocks_router._SCAFFOLD_SPECS["analogue_simulator"]
    assert "meas_statements" in spec.inputs and "meas_statements" not in spec.outputs
    assert "measurements" in spec.outputs and "measurements" not in spec.inputs


def test_analogue_simulator_is_a_block_type_and_an_expensive_one():
    assert BlockType.ANALOGUE_SIMULATOR.value == "analogue_simulator"
    assert "analogue_simulator" in EXPENSIVE_BLOCK_TYPES
    assert enrichment._ENRICHERS["analogue_simulator"] is enrichment._enrich_scaffold_block


def test_the_improve_run_prompt_covers_the_kind():
    prompt = get_system_prompt("improve_run")
    assert "- analogue_simulator - " in prompt
