from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.modules.blocks import router as blocks_router
from app.modules.blocks.schemas import CodeGenerateRequest


def _fake_settings(key: str = "sk-test"):
    return SimpleNamespace(openai_api_key=key, openai_model="gpt-test")


@pytest.mark.asyncio
async def test_generate_code_payload_uses_language_and_fills_code_port(monkeypatch):
    captured = {}

    async def fake_call_openai_json(system_prompt, user_message, temperature=0.3):
        captured["system_prompt"] = system_prompt
        captured["user_message"] = user_message
        # Model may wrap the code value in fences; payload must strip them.
        return {
            "code": "```rust\nfn main() { println!(\"hi\"); }\n```",
            "explanation": "Prints hi.",
            "improvements": "- add args",
            "dependencies": "None",
            "language": "rust",
        }

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_call_openai_json)

    result = await blocks_router.generate_code_payload(
        block_name="reverse_string",
        category="general",
        description="a function that reverses a string",
        language="rust",
    )

    # The requested language is threaded into the prompt the model sees.
    assert "rust" in captured["user_message"].lower()

    params = result["tool_calls"][0]["params"]
    assert params["block_type"] == "code"

    ports = {p["port_name"]: p["port_content"] for p in params["output_ports"]}
    assert set(ports) == {"code", "explanation", "improvements", "dependencies", "language"}
    assert "status" not in ports and "errors" not in ports and "warnings" not in ports
    # Fences stripped, real code present.
    assert ports["code"] == 'fn main() { println!("hi"); }'
    assert "```" not in ports["code"]
    assert ports["explanation"] == "Prints hi."
    assert ports["dependencies"] == "None"
    assert ports["language"] == "rust"

    inputs = {p["port_name"]: p["port_content"] for p in params["input_ports"]}
    assert inputs["language"] == "rust"
    assert inputs["block_description"] == "a function that reverses a string"


@pytest.mark.asyncio
async def test_generate_code_payload_overrides_mismatched_language(monkeypatch):
    """If the model echoes a different language than requested, the port keeps the request."""

    async def fake_call_openai_json(system_prompt, user_message, temperature=0.3):
        return {"code": "print('hi')", "language": "python"}  # asked for go, got python

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_call_openai_json)

    result = await blocks_router.generate_code_payload(
        block_name="x", description="do thing", language="go"
    )
    ports = {p["port_name"]: p["port_content"] for p in result["tool_calls"][0]["params"]["output_ports"]}
    assert ports["language"] == "go"


@pytest.mark.asyncio
async def test_generate_code_payload_language_alias_not_a_mismatch(monkeypatch):
    """js vs javascript is an alias, not a mismatch — the returned value is preserved as the request."""

    async def fake_call_openai_json(system_prompt, user_message, temperature=0.3):
        return {"code": "console.log(1)", "language": "javascript"}

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_call_openai_json)

    result = await blocks_router.generate_code_payload(
        block_name="x", description="do thing", language="js"
    )
    ports = {p["port_name"]: p["port_content"] for p in result["tool_calls"][0]["params"]["output_ports"]}
    assert ports["language"] == "js"  # surfaces the requested spelling


@pytest.mark.asyncio
async def test_generate_code_payload_no_key_returns_none(monkeypatch):
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings(key=""))
    result = await blocks_router.generate_code_payload(
        block_name="x", description="do thing", language="python"
    )
    assert result is None


def test_simple_code_response_shape():
    body = CodeGenerateRequest(
        block_name="my code", category="util", description="d", language="go"
    )
    resp = blocks_router._simple_code_response(body)
    params = resp["tool_calls"][0]["params"]
    assert params["name"] == "my_code"
    assert params["block_type"] == "code"
    inputs = {p["port_name"]: p["port_content"] for p in params["input_ports"]}
    assert inputs["language"] == "go"
    out = {p["port_name"]: p["port_content"] for p in params["output_ports"]}
    assert set(out) == {"code", "explanation", "improvements", "dependencies", "language"}
    assert out["language"] == "go"
