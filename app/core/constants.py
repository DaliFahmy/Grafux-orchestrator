"""Shared constants and enums that replace magic strings spread across modules.

Centralizing these keeps the canvas block vocabulary and the workflow tool-routing
rules in one auditable place instead of duplicated string literals in routers and
the graph engine.
"""

from __future__ import annotations

from enum import Enum


class BlockType(str, Enum):
    """
    Canvas block kinds. ``str`` subclass so values compare/serialize as plain strings.

    This must list every type the canvas supports.  It previously covered only 11 of
    them — gpu, claw, devices, memory, selection and filter were missing — which made
    it a trap for anyone treating it as the registry: the real port definitions live
    in ``_SCAFFOLD_SPECS`` (app/modules/blocks/router.py) and always had all of them.
    The gap is closed here alongside the three chip-design types.
    """

    # Search / AI-generated
    TOPICS = "topics"
    COMPONENTS = "components"
    COMMANDS = "commands"
    TOOLS = "tools"
    PROCEDURES = "procedures"
    CODE = "code"
    # Media & interaction
    IMAGE = "image"
    LOCATION = "location"
    LIVE = "live"
    STREAM = "stream"
    WHITE_BOARD = "white_board"
    # Compute & hardware
    GPU = "gpu"
    CLAW = "claw"
    DEVICES = "devices"
    # Canvas plumbing
    MEMORY = "memory"
    SELECTION = "selection"
    FILTER = "filter"
    # Chip design (EDA): code -> verilator -> yosys -> openroad
    VERILATOR = "verilator"
    YOSYS = "yosys"
    OPENROAD = "openroad"
    # Verification: an AI-written cocotb testbench derived from the SPEC (not the
    # RTL) that a verilator block runs against the design. The wedge block of the
    # spec -> RTL -> testbench -> simulate -> fix loop.
    TESTBENCH = "testbench"
    # The RTL source of that same loop: a spec-driven HDL generator. Kept apart
    # from CODE because a design needs a spec, a frozen module interface and
    # synthesizability rules that a general-purpose programming prompt has no
    # reason to know about. CODE still accepts an HDL language, so canvases built
    # before this type keep working.
    CODE_HDL = "code_hdl"
    # ...and the CONTRACT both of those derive from: a spec-driven specification
    # writer. It turns a rough explanation plus the parameters a design always
    # needs (widths, sync/async, combinational/sequential) into an enumerated,
    # testable spec, and feeds the SAME text to CODE_HDL and TESTBENCH — which is
    # the only thing that makes a design and its tests agree about what "correct"
    # means. It also closes the OUTER loop: [fix_rtl] can repair a design against
    # failing tests, but nothing could repair the contract they were both derived
    # from, which is the actual root cause whenever the two disagree.
    SPEC_HDL = "spec_hdl"


# block_type → Msg_config section name used for LLM prompt selection.
BLOCK_TYPE_SECTION: dict[str, str] = {
    BlockType.TOPICS.value: "create_topic",
    BlockType.COMPONENTS.value: "create_component",
    BlockType.COMMANDS.value: "create_cmd",
    BlockType.TOOLS.value: "create_tool",
    BlockType.PROCEDURES.value: "create_procedure",
    BlockType.CODE.value: "create_code",
    BlockType.IMAGE.value: "create_image",
    BlockType.TESTBENCH.value: "create_testbench",
    BlockType.CODE_HDL.value: "create_code_hdl",
    BlockType.SPEC_HDL.value: "create_spec_hdl",
}

# Msg_config section for REPAIRING an existing HDL design against failing tests.
#
# Deliberately NOT in BLOCK_TYPE_SECTION: it is a mode of the code block, chosen
# per request by router.code_prompt_section(), not a block type. That map is
# looked up as BLOCK_TYPE_SECTION.get(block_type, ...) on every AI block, and a
# non-type key in it would eventually route a real block to the RTL fixer.
CODE_FIX_RTL_SECTION = "fix_rtl"

# Search-block types whose Run/Regenerate output is grounded in live web data.
GROUNDABLE_BLOCK_TYPES: frozenset[str] = frozenset(
    {BlockType.TOPICS.value, BlockType.COMPONENTS.value, BlockType.PROCEDURES.value}
)


# ── Block agents ──────────────────────────────────────────────────────────────

# block_type -> Msg_config section holding that block agent's OBJECTIVE: what
# "done well" means for a block of this type, and the order it should work in.
#
# Deliberately sparse. Most blocks are served by the generic objective, and a
# per-type section earns its place only where the type has a real procedure to
# follow -- which today means the chip-design wedge, where the agent has to reason
# about a spec, a design, its tests and a simulator that judges all three.
# Everything absent here resolves to BLOCK_AGENT_DEFAULT_SECTION; see
# app.modules.session.block_agent.objective_section().
BLOCK_AGENT_DEFAULT_SECTION = "block_agent_default"

BLOCK_AGENT_SECTION: dict[str, str] = {
    BlockType.SPEC_HDL.value: "block_agent_spec_hdl",
    BlockType.CODE_HDL.value: "block_agent_code_hdl",
    BlockType.TESTBENCH.value: "block_agent_testbench",
    BlockType.VERILATOR.value: "block_agent_verilator",
}

# Block types whose Run provisions a cloud pod: real money, and minutes to hours
# of wall clock (an OpenROAD place-and-route can take 30-90 minutes). A block
# agent may run these, but they are metered separately from ordinary runs so a
# team of agents cannot provision a pod each -- see AgentPolicy.max_expensive_runs.
EXPENSIVE_BLOCK_TYPES: frozenset[str] = frozenset({
    BlockType.VERILATOR.value,
    BlockType.YOSYS.value,
    BlockType.OPENROAD.value,
    BlockType.GPU.value,
})


# ── Workflow tool routing ──────────────────────────────────────────────────────

# Graph node names the agent's tool calls are dispatched to.
NODE_MCP = "mcp"
NODE_RESEARCH = "research"
NODE_SANDBOX = "sandbox"
NODE_DEVICE = "device"


def route_tool_to_node(tool_name: str) -> str:
    """Map an agent tool-call name to the workflow graph node that handles it.

    Mirrors the historical prefix rules; ``mcp`` is the default fall-through so an
    unrecognized tool still routes somewhere rather than ending the run.
    """
    name = tool_name or ""
    if name.startswith(("mcp_", "grafux_")):
        return NODE_MCP
    if name.startswith("research_") or name == "web_search":
        return NODE_RESEARCH
    if name.startswith("sandbox_") or name == "execute_code":
        return NODE_SANDBOX
    if name.startswith("device_"):
        return NODE_DEVICE
    return NODE_MCP
