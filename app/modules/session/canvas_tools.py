"""Gemini Live function declarations and mapping to Grafux canvas actions."""

from __future__ import annotations

from typing import Any


def _str_prop(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _func(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return {
        "name": name,
        "description": description,
        "parameters": schema,
    }


CANVAS_FUNCTION_DECLARATIONS: list[dict[str, Any]] = [
    _func(
        "set_port_value",
        "Set the value of an input or output port on a canvas block.",
        {
            "target_block": _str_prop("Block name on the canvas. Omit when one active block is in context."),
            "direction": _str_prop("Must be 'input' or 'output'."),
            "port_name": _str_prop("Port name (snake_case)."),
            "value": _str_prop("Value to write into the port."),
        },
        required=["direction", "port_name", "value"],
    ),
    _func(
        "read_port_value",
        "Read the FULL current content of a block's input or output port file. "
        "Use when the user asks what a port or block contains, to summarize or explain "
        "port data, or when the canvas context marks a value as truncated. This returns "
        "the content to you — it is not a canvas edit.",
        {
            "target_block": _str_prop("Block name on the canvas. Omit when one active block is in context."),
            "direction": _str_prop("Must be 'input' or 'output'."),
            "port_name": _str_prop("Port name (snake_case)."),
        },
        required=["direction", "port_name"],
    ),
    _func(
        "run_block",
        "Execute / run a block on the canvas.",
        {"target_block": _str_prop("Block name on the canvas.")},
        required=["target_block"],
    ),
    _func(
        "regenerate_block",
        "Regenerate a block using optional new instructions.",
        {
            "target_block": _str_prop("Block name on the canvas."),
            "description": _str_prop("Optional regeneration instructions."),
        },
        required=["target_block"],
    ),
    _func(
        "add_port",
        "Add an input or output port to a block.",
        {
            "target_block": _str_prop("Block name on the canvas."),
            "direction": _str_prop("Must be 'input' or 'output'."),
            "port_name": _str_prop("New port name (snake_case)."),
            "description": _str_prop("Optional port description."),
            "value": _str_prop("Optional initial port value."),
        },
        required=["target_block", "direction", "port_name"],
    ),
    _func(
        "remove_port",
        "Remove a port from a block.",
        {
            "target_block": _str_prop("Block name on the canvas."),
            "direction": _str_prop("Must be 'input' or 'output'."),
            "port_name": _str_prop("Port name to remove."),
        },
        required=["target_block", "direction", "port_name"],
    ),
    _func(
        "rename_port",
        "Rename a port on a block.",
        {
            "target_block": _str_prop("Block name on the canvas."),
            "direction": _str_prop("Must be 'input' or 'output'."),
            "old_port_name": _str_prop("Current port name."),
            "new_port_name": _str_prop("New port name (snake_case)."),
        },
        required=["target_block", "direction", "old_port_name", "new_port_name"],
    ),
    _func(
        "open_port",
        "Open a port's data file in the editor.",
        {
            "target_block": _str_prop("Block name on the canvas."),
            "direction": _str_prop("Must be 'input' or 'output'."),
            "port_name": _str_prop("Port name to open."),
        },
        required=["target_block", "direction", "port_name"],
    ),
    _func(
        "set_description",
        "Update a block's reference description.",
        {
            "target_block": _str_prop("Block name on the canvas."),
            "description": _str_prop("New description text."),
        },
        required=["target_block", "description"],
    ),
    _func(
        "connect_ports",
        "Connect an output port on one block to an input port on another.",
        {
            "from_block": _str_prop("Source block name."),
            "from_port": _str_prop("Source output port name."),
            "to_block": _str_prop("Destination block name."),
            "to_port": _str_prop("Destination input port name."),
        },
        required=["from_block", "from_port", "to_block", "to_port"],
    ),
    _func(
        "disconnect_ports",
        "Disconnect an existing port connection.",
        {
            "from_block": _str_prop("Source block name."),
            "from_port": _str_prop("Source output port name."),
            "to_block": _str_prop("Destination block name."),
            "to_port": _str_prop("Destination input port name."),
        },
        required=["from_block", "from_port", "to_block", "to_port"],
    ),
    _func(
        "delete_block",
        "Delete a block from the canvas (only after user confirmation).",
        {"target_block": _str_prop("Block name to delete.")},
        required=["target_block"],
    ),
    _func(
        "load_block",
        "Load a saved block from the project library onto the canvas (only after user confirmation).",
        {
            "block_type": _str_prop("Library block type, e.g. tools, topics."),
            "category": _str_prop("Category folder for category-based types (e.g. general). Omit for flat types."),
            "block_name": _str_prop("Saved block name (not the category name)."),
        },
        required=["block_type", "block_name"],
    ),
    _func(
        "create_block",
        "Create a new saved block (only after user confirmation).",
        {
            "block_type": _str_prop(
                "Canonical block type (plural where applicable). Pick the best fit:\n"
                "- topics: research/data on a subject (web-grounded entities).\n"
                "- components: a hardware/software component's specs (web-grounded).\n"
                "- procedures: step-by-step how-to (web-grounded).\n"
                "- commands: a shell/CLI command block.\n"
                "- tools: an MCP tool that runs Python (give a clear description).\n"
                "- code: generate source code in a chosen language (set 'language').\n"
                "- image: generate/edit/search an image (the description is the prompt).\n"
                "- location: geocode an address and show a map (set 'address').\n"
                "- live: watch/transcribe a YouTube video or live stream (set 'url').\n"
                "- stream: broadcast the user's camera/mic and transcribe it.\n"
                "- white_board: draw the result on a Miro whiteboard embedded in the "
                "block; the description becomes the prompt the AI designs it from.\n"
                "- gpu: compile/run code on a cloud GPU (optionally set 'gpu_model'/'language').\n"
                "- claw: an AI agent assembled from a description; can connect to external "
                "apps via Composio (set 'connections' to the apps it should act on).\n"
                "- devices: run a command/code on a connected hardware device.\n"
                "- memory: snapshot or sequence values from other blocks.\n"
                "- selection: pick one of several inputs by criteria.\n"
                "- filter: filter text/image/video input by criteria."
            ),
            "block_name": _str_prop("New block name (snake_case)."),
            "description": _str_prop("What the block does."),
            "category": _str_prop("Optional category folder for category-based types."),
            "language": _str_prop(
                "For a 'code' or 'gpu' block: the programming language "
                "(e.g. python, javascript, go, rust, c++, cuda). Ignored for other types."
            ),
            "address": _str_prop(
                "For a 'location' block: the address/place to geocode "
                "(e.g. 'Eiffel Tower, Paris'). Ignored for other types."
            ),
            "url": _str_prop(
                "For a 'live' block: the YouTube video or live-stream URL. "
                "Ignored for other types."
            ),
            "gpu_model": _str_prop(
                "For a 'gpu' block: the GPU model to provision (e.g. 'NVIDIA A100'). "
                "Optional; ignored for other types."
            ),
            "connections": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "For a 'claw' block: external app/service toolkit names the agent should "
                    "connect to and act on (Composio toolkits), lowercase slugs, e.g. "
                    "['googlesheets','gmail','telegram','slack']. Infer from the description "
                    "(e.g. 'send info to Google Sheets' → ['googlesheets']). Omit for other types."
                ),
            },
            "inputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Input port names.",
            },
            "outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Output port names.",
            },
        },
        required=["block_type", "block_name", "description"],
    ),
]


def get_live_tools_config() -> list[dict[str, Any]]:
    """Tools list for Gemini Live connect config."""
    return [{"function_declarations": CANVAS_FUNCTION_DECLARATIONS}]


def _default_target_block(active_blocks: list[Any]) -> str:
    if len(active_blocks) == 1:
        name = active_blocks[0].get("name", "")
        if name:
            return str(name)
    return ""


def function_call_to_action(
    name: str,
    args: dict[str, Any],
    active_blocks: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Map a Gemini function call to a Grafux ##ACTIONS##-compatible action dict."""
    active_blocks = active_blocks or []
    action: dict[str, Any] = {"type": name}

    if name in (
        "set_port_value",
        "run_block",
        "regenerate_block",
        "add_port",
        "remove_port",
        "rename_port",
        "open_port",
        "set_description",
        "delete_block",
    ):
        target = (args.get("target_block") or "").strip()
        if not target:
            target = _default_target_block(active_blocks)
        if target:
            action["target_block"] = target
        for key in (
            "direction",
            "port_name",
            "value",
            "description",
            "old_port_name",
            "new_port_name",
        ):
            if key in args and args[key] is not None:
                action[key] = args[key]

    elif name in ("connect_ports", "disconnect_ports"):
        for key in ("from_block", "from_port", "to_block", "to_port"):
            if key in args and args[key] is not None:
                action[key] = args[key]

    elif name == "load_block":
        for key in ("block_type", "category", "block_name"):
            if key in args and args[key] is not None:
                action[key] = str(args[key]).strip()

    elif name == "create_block":
        for key in (
            "block_type", "block_name", "description", "category",
            "language", "address", "url", "gpu_model",
        ):
            if key in args and args[key] is not None:
                action[key] = args[key]
        if "inputs" in args:
            action["inputs"] = args["inputs"]
        if "outputs" in args:
            action["outputs"] = args["outputs"]
        if "connections" in args:
            action["connections"] = args["connections"]

    else:
        return None

    return action


def tool_calls_to_actions(
    function_calls: list[Any],
    active_blocks: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert Gemini tool function_calls to action dicts for the Qt client."""
    actions: list[dict[str, Any]] = []
    for fc in function_calls:
        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
        raw_args = getattr(fc, "args", None)
        if raw_args is None and isinstance(fc, dict):
            raw_args = fc.get("args", {})
        if not name:
            continue
        if isinstance(raw_args, dict):
            args = raw_args
        elif raw_args is not None:
            try:
                args = dict(raw_args)
            except (TypeError, ValueError):
                args = {}
        else:
            args = {}
        action = function_call_to_action(str(name), args, active_blocks)
        if action:
            actions.append(action)
    return actions


def build_tool_function_responses(function_calls: list[Any]) -> list[dict[str, Any]]:
    """Acknowledge tool calls so Gemini can continue the live session."""
    responses: list[dict[str, Any]] = []
    for fc in function_calls:
        fc_id = getattr(fc, "id", None) or (fc.get("id") if isinstance(fc, dict) else None)
        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "")
        entry: dict[str, Any] = {
            "name": name,
            "response": {"result": "queued"},
        }
        if fc_id:
            entry["id"] = fc_id
        responses.append(entry)
    return responses
