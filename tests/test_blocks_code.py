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

    async def fake_call_openai_text(system_prompt, user_message, temperature=0.2):
        captured["system_prompt"] = system_prompt
        captured["user_message"] = user_message
        # Model may wrap output in fences; payload must strip them.
        return "```rust\nfn main() { println!(\"hi\"); }\n```"

    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(blocks_router, "_call_openai_text", fake_call_openai_text)

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
    assert set(ports) >= {"code", "status", "errors", "warnings"}
    # Fences stripped, real code present.
    assert ports["code"] == 'fn main() { println!("hi"); }'
    assert "```" not in ports["code"]
    assert ports["status"] == "generated"

    inputs = {p["port_name"]: p["port_content"] for p in params["input_ports"]}
    assert inputs["language"] == "rust"
    assert inputs["description"] == "a function that reverses a string"


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
    out_names = {p["port_name"] for p in params["output_ports"]}
    assert out_names == {"code", "status", "errors", "warnings"}
