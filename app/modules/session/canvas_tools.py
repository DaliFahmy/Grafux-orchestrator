"""Canvas tool declarations and their mapping to Grafux canvas actions.

These declarations are the SINGLE SOURCE for what an AI may do to a canvas.
Every consumer derives from them rather than restating them:

* Gemini Live   -> ``get_live_tools_config()`` (the native declaration shape);
* OpenAI        -> ``to_openai_tools()``;
* Anthropic     -> ``to_anthropic_tools()`` (the block agents);
* the canvas    -> ``function_call_to_action()``, whose output is applied by the
  Qt client's ``BlockAgentController``.

Adding a tool means adding it here once. The one remaining restatement is the
``##ACTIONS##`` grammar in ``Msg_config [chat_assistant]``, which serves the
legacy chat path only -- new work goes through tool calls.
"""

from __future__ import annotations

from typing import Any

from app.core.llm import to_anthropic_tools as _to_anthropic_tools
from app.core.llm import to_openai_tools as _to_openai_tools


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
        "Update a block's description (its block_description input port).",
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
                "- memory: remember values from other blocks - snapshot them, step "
                "through them in sequence, accumulate them into one growing record, "
                "or merge several inputs at once into one combined output, the last "
                "two with an AI review (memory_mode).\n"
                "- selection: pick one of several inputs by criteria.\n"
                "- filter: filter text/image/video input by criteria.\n"
                "- verilator: simulate or lint a Verilog/SystemVerilog design and "
                "report pass/fail plus a waveform (chip design; set 'top').\n"
                "- yosys: synthesize RTL into a gate-level netlist for a PDK "
                "(chip design; set 'top'/'pdk').\n"
                "- openroad: place and route a netlist into a chip layout/GDS "
                "(chip design; set 'top'/'pdk'/'clock_period').\n"
                "- testbench: write a cocotb testbench that verifies a Verilog module "
                "against its SPEC (chip verification; set 'top' and 'spec').\n"
                "- code_hdl: write a synthesizable RTL design (verilog, systemverilog "
                "or vhdl) from a SPEC (chip design; set 'top', 'spec' and 'language'). "
                "Prefer this over 'code' whenever the user wants hardware.\n"
                "- spec_hdl: turn a rough explanation of some hardware into a "
                "rigorous, numbered SPECIFICATION (chip design; set 'explanation', "
                "and 'top'/'language' when known). Its 'spec' output feeds BOTH a "
                "code_hdl and a testbench block, which is what makes a design and "
                "its tests agree about what correct means. Create it when the user "
                "describes hardware loosely, wants the requirements written down, or "
                "asks for a spec.\n"
                "For chip design the usual chain is spec_hdl -> code_hdl -> verilator "
                "-> yosys -> openroad, and the verification loop is spec_hdl feeding "
                "BOTH code_hdl and testbench, which both feed verilator; a plain code "
                "block with language=verilog still works for canvases built that way. "
                "Create only the blocks asked for."
            ),
            "block_name": _str_prop("New block name (snake_case)."),
            "description": _str_prop("What the block does."),
            "category": _str_prop("Optional category folder for category-based types."),
            "language": _str_prop(
                "For a 'code' or 'gpu' block: the programming language "
                "(e.g. python, javascript, go, rust, c++, cuda). For a 'code_hdl' or "
                "'spec_hdl' block: the hardware description language, one of verilog, "
                "systemverilog or vhdl (defaults to systemverilog). "
                "Ignored for other types."
            ),
            "address": _str_prop(
                "For a 'location' block: the address/place to geocode "
                "(e.g. 'Eiffel Tower, Paris'). Ignored for other types."
            ),
            "url": _str_prop(
                "For a 'live' block: the YouTube video or live-stream URL. "
                "Ignored for other types."
            ),
            "memory_mode": _str_prop(
                "For a 'memory' block: how it remembers. One of 'snapshot' (each run "
                "copies the input into a new timestamped output), 'sequential' (each "
                "run emits the next of several inputs, wrapping around) or "
                "'accumulate' (each run appends the 'data' input to a growing "
                "'accumulated_data' record and writes an AI review of how the new "
                "data relates to what is already there onto 'analysis') or 'merge' "
                "(each run combines EVERY connected input at once into 'merged_data' "
                "and writes an AI review of how those inputs agree, conflict and "
                "complement each other onto 'analysis'; the block grows a new input "
                "port whenever the last one is connected). Use 'accumulate' to build "
                "up one source over time and 'merge' to bring several sources "
                "together in one go. Defaults to snapshot; ignored for other types."
            ),
            "gpu_model": _str_prop(
                "For a 'gpu' block: the GPU model to provision (e.g. 'NVIDIA A100'). "
                "Optional; ignored for other types."
            ),
            "top": _str_prop(
                "For a 'verilator', 'yosys', 'openroad', 'testbench', 'code_hdl' or "
                "'spec_hdl' block: the top-level module name of the design (e.g. "
                "'counter'). "
                "Optional; ignored for other types."
            ),
            "spec": _str_prop(
                "For a 'testbench' block: the behavioural specification the tests must "
                "check. For a 'code_hdl' block: the same specification, which the "
                "design must implement — passing the identical text to both is what "
                "makes a design and its tests agree about what correct means. Either "
                "way it says what the module must and must not do. Falls back to the "
                "description when omitted; ignored for other types. Prefer creating a "
                "'spec_hdl' block and wiring its 'spec' output into both when the user "
                "has only described the hardware loosely."
            ),
            "explanation": _str_prop(
                "For a 'spec_hdl' block: the rough, plain-language description of the "
                "hardware to specify (e.g. 'a small FIFO between two clock domains that "
                "drops writes when it is full'). Unlike 'spec' this is the INPUT to be "
                "made rigorous, not an already-rigorous contract. Falls back to the "
                "description when omitted; ignored for other types."
            ),
            "pdk": _str_prop(
                "For a 'yosys' or 'openroad' block: the process design kit / OpenROAD "
                "platform (sky130hd, sky130hs, asap7, nangate45). Defaults to sky130hd; "
                "ignored for other types."
            ),
            "clock_period": _str_prop(
                "For an 'openroad' block: the target clock period in nanoseconds "
                "(e.g. '10'). Optional; ignored for other types."
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


# Declarations that coordinate AGENTS rather than touch the canvas. Kept apart
# from CANVAS_FUNCTION_DECLARATIONS because the voice/chat paths must not see
# them -- there is no agent for them to talk to.
AGENT_COORDINATION_DECLARATIONS: list[dict[str, Any]] = [
    _func(
        "post_note",
        "Share a short finding with the other active block agents. Non-blocking: "
        "it does not wait for a reply. Use it to say what you changed and why, so "
        "no one repeats your work.",
        {
            "note": _str_prop("What you want the other agents to know."),
            "to_block": _str_prop(
                "Optional block name to address this to. Omit to tell everyone."
            ),
        },
        required=["note"],
    ),
    _func(
        "request_agent",
        "Ask the agent of ANOTHER block to do something on its own block. This is "
        "the only way to change a block that has its own active agent.",
        {
            "target_block": _str_prop("Block name whose agent should act."),
            "request": _str_prop("What you need it to do, and the evidence for why."),
            "wait": _str_prop(
                "'true' to wait for its reply before continuing, 'false' to carry on."
            ),
        },
        required=["target_block", "request"],
    ),
    _func(
        "ask_user",
        "Ask the user a question and wait for their answer. Use it when you need a "
        "decision only they can make, or when a rule requires their confirmation.",
        {"question": _str_prop("The question, in one or two plain sentences.")},
        required=["question"],
    ),
    _func(
        "finish",
        "Declare your work on this block done and stop. Always end with this.",
        {
            "summary": _str_prop("What you changed and what you verified."),
            "goal_met": _str_prop("'true' if the objective is met, otherwise 'false'."),
            "blocking": _str_prop(
                "If not met, the one thing standing in the way."
            ),
        },
        required=["summary", "goal_met"],
    ),
]


def _by_name(names: list[str] | None) -> list[dict[str, Any]]:
    """Declarations filtered to ``names`` (all canvas + agent tools when None).

    Order follows the declaration lists, not the caller's ``names``, so the tool
    array is byte-stable across steps -- an unstable tool order silently kills
    prompt caching, which renders ``tools`` before ``system``.
    """
    everything = CANVAS_FUNCTION_DECLARATIONS + AGENT_COORDINATION_DECLARATIONS
    if names is None:
        return list(everything)
    wanted = set(names)
    return [d for d in everything if d["name"] in wanted]


def to_declarations(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Canvas/agent declarations in the neutral shape ``call_llm_tools`` takes."""
    return _by_name(names)


def get_live_tools_config() -> list[dict[str, Any]]:
    """Tools list for Gemini Live connect config."""
    return [{"function_declarations": CANVAS_FUNCTION_DECLARATIONS}]


def to_openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Canvas/agent declarations in OpenAI ``tools=`` shape."""
    return _to_openai_tools(_by_name(names))


def to_anthropic_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Canvas/agent declarations in Anthropic ``tools=`` shape."""
    return _to_anthropic_tools(_by_name(names))


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
            # EDA seeds — see _SCAFFOLD_SEED_KEYS in enrichment.py. Forwarding them
            # here is what lets "create a yosys block for my counter on sky130"
            # land with top/pdk already filled instead of empty ports.
            "top", "pdk", "clock_period",
            # testbench / code_hdl seed: the spec the tests and the design share,
            # and the rough explanation the spec_hdl block makes rigorous.
            "spec", "explanation",
            # The two params that pick a SUBTYPE. Both are declared above, and
            # leaving them out here does not fail -- the enricher normalises a
            # missing value to the default, so every voice-created memory block
            # silently became a snapshot one whatever the user asked for. Anything
            # declared to the model belongs in this tuple.
            "memory_mode", "filter_type",
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
