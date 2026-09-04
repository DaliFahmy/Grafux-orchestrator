from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.modules.blocks import router as blocks_router


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(
        openai_api_key=openai, anthropic_api_key=anthropic, openai_model="gpt-test"
    )


def _ports(params, side):
    return {p["port_name"]: p for p in params[f"{side}_ports"]}


# ── scaffold builder (no AI, pure) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_scaffold_location_ports_paths_and_address_seed():
    result = await blocks_router.generate_scaffold_payload(
        block_type="location",
        block_name="eiffel tower",
        description="Eiffel Tower",
        seeds={"address": "Eiffel Tower, Paris"},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "location"
    assert params["name"] == "eiffel_tower"

    ins = _ports(params, "input")
    outs = _ports(params, "output")
    assert set(ins) == {
        "block_description", "name", "full_address", "street",
        "postal_code", "city", "country",
    }
    assert set(outs) == {
        "name", "full_address", "street", "postal_code", "city", "country", "status",
    }
    # Address seed lands on full_address; explicit seed beats the description fallback.
    assert ins["full_address"]["port_content"] == "Eiffel Tower, Paris"
    # location is NOT category-based → flat path.
    assert ins["full_address"]["port_path"] == "data/location/eiffel_tower/inputs/full_address.txt"
    assert ins["block_description"]["port_content"] == "Eiffel Tower"


@pytest.mark.asyncio
async def test_scaffold_location_seeds_full_address_from_description_when_no_seed():
    result = await blocks_router.generate_scaffold_payload(
        block_type="location", block_name="home", description="1600 Amphitheatre Pkwy",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["full_address"]["port_content"] == "1600 Amphitheatre Pkwy"


@pytest.mark.asyncio
async def test_scaffold_white_board_ports_paths_and_prompt_seed():
    result = await blocks_router.generate_scaffold_payload(
        block_type="white_board",
        block_name="q3 launch",
        description="map our Q3 launch plan",
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "white_board"
    assert params["name"] == "q3_launch"

    ins = _ports(params, "input")
    outs = _ports(params, "output")
    assert set(ins) == {"block_description", "board_name", "prompt", "notes", "board_id"}
    assert set(outs) == {"board_url", "embed_url", "board_id", "summary", "status"}
    # The description doubles as the board prompt, so the block is ready to Run.
    assert ins["prompt"]["port_content"] == "map our Q3 launch plan"
    # white_board is NOT category-based -> flat path.
    assert ins["prompt"]["port_path"] == "data/white_board/q3_launch/inputs/prompt.txt"


@pytest.mark.asyncio
async def test_scaffold_live_seeds_link_from_url():
    result = await blocks_router.generate_scaffold_payload(
        block_type="live", block_name="news",
        description="watch the news",
        seeds={"url": "https://youtube.com/watch?v=abc"},
    )
    params = result["tool_calls"][0]["params"]
    ins = _ports(params, "input")
    outs = _ports(params, "output")
    assert set(ins) == {"block_description", "live_stream_link", "question"}
    assert set(outs) == {"answer", "transcript", "status"}
    assert ins["live_stream_link"]["port_content"] == "https://youtube.com/watch?v=abc"
    # description does NOT bleed into the link.
    assert ins["block_description"]["port_content"] == "watch the news"


@pytest.mark.asyncio
async def test_scaffold_stream_minimal():
    result = await blocks_router.generate_scaffold_payload(
        block_type="stream", block_name="cam", description="broadcast me",
    )
    params = result["tool_calls"][0]["params"]
    assert set(_ports(params, "input")) == {"block_description", "question"}
    assert set(_ports(params, "output")) == {"answer", "transcript", "status"}


@pytest.mark.asyncio
async def test_scaffold_image_has_image_input_and_prompt_seed_and_category_path():
    result = await blocks_router.generate_scaffold_payload(
        block_type="image", block_name="sunset", category="art", description="a sunset",
    )
    params = result["tool_calls"][0]["params"]
    ins = _ports(params, "input")
    outs = _ports(params, "output")
    # The `image` input port (edit mode) must exist — the old payload omitted it.
    assert set(ins) == {"block_description", "prompt", "modification", "search_for", "image"}
    assert set(outs) == {"image", "image_name", "image_description", "improvements", "status"}
    assert ins["prompt"]["port_content"] == "a sunset"  # description seeds prompt
    # image IS category-based.
    assert ins["prompt"]["port_path"] == "data/image/art/sunset/inputs/prompt.txt"


@pytest.mark.asyncio
async def test_scaffold_gpu_seeds_model_and_language():
    result = await blocks_router.generate_scaffold_payload(
        block_type="gpu", block_name="bench", description="matmul",
        seeds={"gpu_model": "NVIDIA A100", "language": "cuda"},
    )
    params = result["tool_calls"][0]["params"]
    ins = _ports(params, "input")
    assert "gpu_model" in ins and "code" in ins and "benchmark" not in ins
    assert ins["gpu_model"]["port_content"] == "NVIDIA A100"
    assert ins["language"]["port_content"] == "cuda"
    assert "benchmark" in _ports(params, "output")
    assert ins["gpu_model"]["port_path"] == "data/gpu/general/bench/inputs/gpu_model.txt"


@pytest.mark.asyncio
async def test_scaffold_devices_seeds_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="devices", block_name="pi", description="blink led",
    )
    ins = _ports(result["tool_calls"][0]["params"], "input")
    assert ins["language"]["port_content"] == "cpp"
    assert ins["timeout"]["port_content"] == "130"


@pytest.mark.asyncio
async def test_scaffold_extra_inputs_outputs_appended_and_deduped():
    result = await blocks_router.generate_scaffold_payload(
        block_type="selection", block_name="pick",
        inputs=["option_a", "option_b", "criteria", "description"],
        outputs=["selected", "extra_out"],
    )
    params = result["tool_calls"][0]["params"]
    ins = _ports(params, "input")
    outs = _ports(params, "output")
    # criteria (spec) not duplicated; description/block_description not duplicated.
    assert set(ins) == {"block_description", "criteria", "option_a", "option_b"}
    assert set(outs) == {"selected", "analysis", "extra_out"}


# ── chip design (EDA): verilator → yosys → openroad ──────────────────────────
#
# These assert the EXACT port sets, not just "contains". The same lists are
# duplicated in the Qt dialog's finalize<Type>() and in the executor's
# build<Type>Outputs(); a drift between them means an AI-created block and a
# hand-created block of the same type get different ports, which then fails at
# Run in a way that points nowhere near the cause.

_VERILATOR_INPUTS = {
    "block_description", "rtl", "testbench", "sva", "top", "mode", "simulator",
    "tests", "seed", "collect_coverage", "max_iterations", "defines",
    "include_dirs", "files", "trace", "sim_args", "verilator_flags", "timeout",
    "instance_type", "image", "api_keys",
}
_VERILATOR_OUTPUTS = {
    "status", "passed", "results", "failures", "coverage", "coverage_report",
    "iterations", "sim_output", "lint", "errors", "warnings", "waveform",
    "rtl", "top", "log", "artifacts", "eda_id", "cost",
    # Split in two: the review of the DESIGN and the review of the TESTS land on
    # separate ports so each can be wired somewhere different.
    "improvements_rtl", "improvements_test",
}
_YOSYS_INPUTS = {
    "block_description", "rtl", "top", "pdk", "liberty", "synth_flags", "defines",
    "include_dirs", "files", "timeout", "instance_type", "image", "api_keys",
    "credentials",
}
_YOSYS_OUTPUTS = {
    "netlist", "status", "top", "pdk", "stats", "report", "errors", "warnings",
    "log", "artifacts", "eda_id", "cost",
}
_OPENROAD_INPUTS = {
    "block_description", "netlist", "rtl", "top", "pdk", "sdc", "clock_port",
    "clock_period", "core_utilization", "aspect_ratio", "die_area", "core_area",
    "place_density", "from_stage", "to_stage", "extra_config", "files", "timeout",
    "instance_type", "image", "api_keys",
}
_OPENROAD_OUTPUTS = {
    "status", "stage", "gds", "def", "netlist_final", "spef", "layout_png",
    "metrics", "reports", "errors", "warnings", "log", "artifacts", "eda_id",
    "cost", "improvements",
}


@pytest.mark.asyncio
async def test_scaffold_verilator_exact_ports_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="verilator", block_name="counter tb",
        description="Simulate a 4-bit counter", seeds={"top": "counter"},
    )
    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "verilator"
    assert params["name"] == "counter_tb"
    ins, outs = _ports(params, "input"), _ports(params, "output")
    assert set(ins) == _VERILATOR_INPUTS
    assert set(outs) == _VERILATOR_OUTPUTS
    assert ins["top"]["port_content"] == "counter"
    # mode stays "sim": a cocotb testbench is detected server-side, so wiring one
    # into an existing block must not depend on anyone changing this dropdown.
    assert ins["mode"]["port_content"] == "sim"
    assert ins["trace"]["port_content"] == "1"
    assert ins["simulator"]["port_content"] == "verilator"
    assert ins["collect_coverage"]["port_content"] == "1"
    # 1 = single-shot, today's behaviour; the fix loop is opt-in.
    assert ins["max_iterations"]["port_content"] == "1"
    assert ins["rtl"]["port_path"] == "data/verilator/general/counter_tb/inputs/rtl.txt"


@pytest.mark.asyncio
async def test_scaffold_yosys_exact_ports_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="yosys", block_name="counter synth",
        description="Synthesize the counter", seeds={"top": "counter"},
    )
    params = result["tool_calls"][0]["params"]
    ins, outs = _ports(params, "input"), _ports(params, "output")
    assert set(ins) == _YOSYS_INPUTS
    assert set(outs) == _YOSYS_OUTPUTS
    assert ins["top"]["port_content"] == "counter"
    assert ins["pdk"]["port_content"] == "sky130hd"
    assert ins["rtl"]["port_path"] == "data/yosys/general/counter_synth/inputs/rtl.txt"


@pytest.mark.asyncio
async def test_scaffold_openroad_exact_ports_and_defaults():
    result = await blocks_router.generate_scaffold_payload(
        block_type="openroad", block_name="counter layout",
        description="Place and route the counter",
        seeds={"top": "counter", "pdk": "sky130hd", "clock_period": "5"},
    )
    params = result["tool_calls"][0]["params"]
    ins, outs = _ports(params, "input"), _ports(params, "output")
    assert set(ins) == _OPENROAD_INPUTS
    assert set(outs) == _OPENROAD_OUTPUTS
    assert ins["clock_period"]["port_content"] == "5"   # seed beats the default
    assert ins["clock_port"]["port_content"] == "clk"
    assert ins["from_stage"]["port_content"] == "synth"
    assert ins["to_stage"]["port_content"] == "final"
    assert outs["gds"]["port_path"] ==         "data/openroad/general/counter_layout/outputs/gds.txt"


@pytest.mark.asyncio
async def test_eda_chain_ports_line_up_for_wiring():
    """
    The pipeline only works if the obvious wiring is the correct one:
    code.code -> verilator.rtl, verilator.rtl -> yosys.rtl, yosys.netlist ->
    openroad.netlist. Guard those four port names against a well-meaning rename.
    """
    ver = await blocks_router.generate_scaffold_payload(
        block_type="verilator", block_name="v")
    yos = await blocks_router.generate_scaffold_payload(
        block_type="yosys", block_name="y")
    orr = await blocks_router.generate_scaffold_payload(
        block_type="openroad", block_name="o")
    ver_out = _ports(ver["tool_calls"][0]["params"], "output")
    yos_in = _ports(yos["tool_calls"][0]["params"], "input")
    yos_out = _ports(yos["tool_calls"][0]["params"], "output")
    orr_in = _ports(orr["tool_calls"][0]["params"], "input")
    assert "rtl" in ver_out and "rtl" in yos_in
    assert "netlist" in yos_out and "netlist" in orr_in


@pytest.mark.asyncio
async def test_scaffold_eda_defaults_do_not_leak_across_types():
    """A verilator block has no PDK; an ORFS default there would be nonsense."""
    ver = await blocks_router.generate_scaffold_payload(
        block_type="verilator", block_name="v")
    ins = _ports(ver["tool_calls"][0]["params"], "input")
    assert "pdk" not in ins and "clock_period" not in ins


@pytest.mark.asyncio
async def test_scaffold_unknown_type_returns_none():
    assert await blocks_router.generate_scaffold_payload(
        block_type="nonsense", block_name="x"
    ) is None


# ── generalized grounded generator (block_type → prompt section + grounding) ──


def _patch_llm(monkeypatch, captured, grounded):
    def fake_get_system_prompt(section):
        captured["section"] = section
        return f"PROMPT[{section}]"

    async def fake_call_llm_json(system_prompt, user_message, model=None, temperature=0.2):
        captured["user_message"] = user_message
        return {"tool_calls": [{"params": {
            "output_ports": [
                {"port_name": "result", "port_content": "x", "port_path": "p"}
            ]
        }}]}

    async def fake_augment(user_message, query):
        grounded["called"] = True
        return user_message + "\nGROUNDED"

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "get_system_prompt", fake_get_system_prompt)
    monkeypatch.setattr(blocks_router, "get_json_schema", lambda: "")
    monkeypatch.setattr(blocks_router, "call_llm_json", fake_call_llm_json)
    monkeypatch.setattr(blocks_router, "_augment_with_grounding", fake_augment)


@pytest.mark.asyncio
async def test_commands_uses_create_cmd_section_and_is_not_grounded(monkeypatch):
    captured: dict = {}
    grounded = {"called": False}
    _patch_llm(monkeypatch, captured, grounded)

    await blocks_router.generate_topic_payload(
        topic_name="list_files", description="list files", block_type="commands",
    )
    assert captured["section"] == "create_cmd"
    assert grounded["called"] is False  # commands are not in GROUNDABLE_BLOCK_TYPES


@pytest.mark.asyncio
async def test_components_uses_section_and_is_grounded(monkeypatch):
    captured: dict = {}
    grounded = {"called": False}
    _patch_llm(monkeypatch, captured, grounded)

    await blocks_router.generate_topic_payload(
        topic_name="esp32", description="short query", block_type="components",
    )
    assert captured["section"] == "create_component"
    assert grounded["called"] is True  # short description on a groundable type → grounded


@pytest.mark.asyncio
async def test_default_block_type_is_topics(monkeypatch):
    captured: dict = {}
    grounded = {"called": False}
    _patch_llm(monkeypatch, captured, grounded)

    await blocks_router.generate_topic_payload(topic_name="ai", description="short")
    assert captured["section"] == "create_topic"
