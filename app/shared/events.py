from __future__ import annotations

# ── Execution lifecycle events ─────────────────────────────────────────────────
EXECUTION_STARTED = "execution.started"
EXECUTION_STEP = "execution.step"
EXECUTION_LOG = "execution.log"
EXECUTION_COMPLETE = "execution.complete"
EXECUTION_FAILED = "execution.failed"
EXECUTION_CANCELLED = "execution.cancelled"
EXECUTION_RETRYING = "execution.retrying"
EXECUTION_CHECKPOINT = "execution.checkpoint"

# ── Agent events ──────────────────────────────────────────────────────────────
AGENT_THOUGHT = "agent.thought"
AGENT_TOOL_CALL = "agent.tool_call"
AGENT_TOOL_RESULT = "agent.tool_result"
AGENT_MESSAGE = "agent.message"
AGENT_MEMORY_UPDATED = "agent.memory_updated"

# ── MCP events ────────────────────────────────────────────────────────────────
MCP_INVOCATION = "mcp.invocation"
MCP_RESULT = "mcp.result"
MCP_ERROR = "mcp.error"

# ── Research events ───────────────────────────────────────────────────────────
RESEARCH_STARTED = "research.started"
RESEARCH_RESULT = "research.result"
RESEARCH_COMPLETE = "research.complete"

# ── Sandbox events ────────────────────────────────────────────────────────────
SANDBOX_STARTED = "sandbox.started"
SANDBOX_STDOUT = "sandbox.stdout"
SANDBOX_STDERR = "sandbox.stderr"
SANDBOX_COMPLETE = "sandbox.complete"
SANDBOX_ERROR = "sandbox.error"

# ── Device events ─────────────────────────────────────────────────────────────
DEVICE_COMMAND_SENT = "device.command_sent"
DEVICE_EVENT_RECEIVED = "device.event"
DEVICE_ERROR = "device.error"

# ── Redis channel helpers ─────────────────────────────────────────────────────

def execution_channel(execution_id: str) -> str:
    return f"execution:{execution_id}"


def org_channel(org_id: str) -> str:
    return f"org:{org_id}"
