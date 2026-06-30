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
