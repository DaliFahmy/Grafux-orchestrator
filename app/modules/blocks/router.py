from __future__ import annotations

import json
import uuid

from fastapi import APIRouter

from app.config import get_settings
from app.core.logging import get_logger
from app.dependencies import CurrentUser
from app.modules.blocks.schemas import TopicGenerateRequest

log = get_logger("blocks.router")

router = APIRouter(prefix="/blocks", tags=["blocks"])


def _make_port_path(category: str, name: str, port_type: str, port_name: str) -> str:
    return f"data/topics/{category}/{name}/{port_type}/{port_name}.txt"


def _simple_topic_response(body: TopicGenerateRequest) -> dict:
    """Fallback: build a minimal block without AI when OpenAI is not configured."""
    name = body.topic_name.replace(" ", "_")
    cat = body.category

    ip = [
        {
            "port_name": "description",
            "port_content": body.description,
            "port_path": _make_port_path(cat, name, "inputs", "description"),
        }
    ]
    for inp in body.inputs:
        if inp and inp != "description":
            ip.append({
                "port_name": inp,
                "port_content": "",
                "port_path": _make_port_path(cat, name, "inputs", inp),
            })

    op = []
    for out in body.outputs:
        if out:
            op.append({
                "port_name": out,
                "port_content": "",
                "port_path": _make_port_path(cat, name, "outputs", out),
            })

    return {
        "tool_calls": [
            {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "block_id": uuid.uuid4().hex[:8],
                    "block_type": "topics",
                    "x": 0,
                    "y": 0,
                    "input_ports": ip,
                    "output_ports": op,
                },
            }
        ],
        "connections": [],
    }


@router.post("/generate/topic")
async def generate_topic_block(
    body: TopicGenerateRequest,
    user: CurrentUser,
) -> dict:
    """Generate a structured topic block using AI (OpenAI) or a simple template fallback."""
    settings = get_settings()

    if not settings.openai_api_key:
        log.info("blocks_generate_topic_fallback", reason="no_openai_key", topic=body.topic_name)
        return _simple_topic_response(body)

    name = body.topic_name.replace(" ", "_")
    cat = body.category

    system_prompt = (
        "You are a Grafux block generator. Given a topic description, generate a structured "
        "topic block as a JSON object in EXACTLY this format (no extra keys, no markdown):\n"
        '{\n'
        '  "tool_calls": [\n'
        '    {\n'
        '      "id": 1,\n'
        '      "jsonrpc": "2.0",\n'
        '      "method": "tools/call",\n'
        '      "params": {\n'
        f'        "name": "{name}",\n'
        f'        "block_id": "{uuid.uuid4().hex[:8]}",\n'
        '        "block_type": "topics",\n'
        '        "x": 0,\n'
        '        "y": 0,\n'
        '        "input_ports": [\n'
        '          {"port_name": "description", "port_content": "<the description text>", '
        f'"port_path": "data/topics/{cat}/{name}/inputs/description.txt"}\n'
        '        ],\n'
        '        "output_ports": [\n'
        '          {"port_name": "<output_name>", "port_content": "<actual value>", '
        f'"port_path": "data/topics/{cat}/{name}/outputs/<output_name>.txt"}\n'
        '        ]\n'
        '      }\n'
        '    }\n'
        '  ],\n'
        '  "connections": []\n'
        '}\n\n'
        "Rules:\n"
        "- Generate meaningful output_ports based on the description (extract real data values)\n"
        "- Use lowercase_with_underscores for all port names\n"
        f"- Keep port_path prefix as data/topics/{cat}/{name}/ substituting actual port names\n"
        "- Output ONLY valid JSON — no markdown fences, no explanation"
    )

    user_message = f"Topic name: {name}\nCategory: {cat}\n"
    if body.inputs:
        user_message += f"Requested input ports (besides 'description'): {', '.join(body.inputs)}\n"
    if body.outputs:
        user_message += f"Requested output ports: {', '.join(body.outputs)}\n"
    if body.description:
        user_message += f"\nDescription / content to extract data from:\n{body.description[:3000]}"

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        result: dict = json.loads(content)

        # Ensure block_id is filled
        if result.get("tool_calls"):
            params = result["tool_calls"][0].get("params", {})
            if not params.get("block_id"):
                params["block_id"] = uuid.uuid4().hex[:8]
                result["tool_calls"][0]["params"] = params

        log.info("blocks_generate_topic_ok", topic=body.topic_name)
        return result

    except Exception as exc:
        log.error("blocks_generate_topic_error", topic=body.topic_name, error=str(exc))
        # Graceful fallback to simple template
        return _simple_topic_response(body)
