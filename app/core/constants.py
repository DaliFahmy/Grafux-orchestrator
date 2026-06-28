"""Shared constants and enums that replace magic strings spread across modules.

Centralizing these keeps the canvas block vocabulary and the workflow tool-routing
rules in one auditable place instead of duplicated string literals in routers and
the graph engine.
"""

from __future__ import annotations

from enum import Enum


class BlockType(str, Enum):
    """Canvas block kinds. ``str`` subclass so values compare/serialize as plain strings."""

    TOPICS = "topics"
    COMPONENTS = "components"
    COMMANDS = "commands"
    TOOLS = "tools"
    PROCEDURES = "procedures"
    CODE = "code"
    IMAGE = "image"


# block_type → Msg_config section name used for LLM prompt selection.
BLOCK_TYPE_SECTION: dict[str, str] = {
    BlockType.TOPICS.value: "create_topic",
    BlockType.COMPONENTS.value: "create_component",
    BlockType.COMMANDS.value: "create_cmd",
    BlockType.TOOLS.value: "create_tool",
    BlockType.PROCEDURES.value: "create_procedure",
    BlockType.CODE.value: "create_code",
    BlockType.IMAGE.value: "create_image",
}

# Search-block types whose Run/Regenerate output is grounded in live web data.
GROUNDABLE_BLOCK_TYPES: frozenset[str] = frozenset(
    {BlockType.TOPICS.value, BlockType.COMPONENTS.value, BlockType.PROCEDURES.value}
)


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
