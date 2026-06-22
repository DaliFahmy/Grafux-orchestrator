from __future__ import annotations

import functools
import operator
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.core.constants import route_tool_to_node
from app.core.logging import get_logger
from app.modules.streaming.event_bus import emit, emit_agent_thought, emit_log
from app.shared import events as ev

log = get_logger("workflow.engine")


# ── Graph State ───────────────────────────────────────────────────────────────

class ExecutionState(TypedDict):
    execution_id: str
    org_id: str
    project_id: str
    workflow_id: str | None
    messages: Annotated[list[BaseMessage], operator.add]
    current_node: str
    tool_results: dict[str, Any]
    agent_memory: dict[str, Any]
    iteration: int
    metadata: dict[str, Any]
    error: str | None
    cancelled: bool


# ── Node implementations ───────────────────────────────────────────────────────

async def _check_cancelled(execution_id: str) -> bool:
    """Check the Redis cancellation flag for this execution."""
    from app.core.redis import get_redis_client
    redis = get_redis_client()
    return bool(await redis.exists(f"cancel:{execution_id}"))


def cancellable_node(
    fn: Callable[[ExecutionState], Awaitable[dict[str, Any]]],
) -> Callable[[ExecutionState], Awaitable[dict[str, Any]]]:
    """Short-circuit a tool node when the execution's cancel flag is set.

    DRYs the identical guard every tool node opened with. The agent node keeps its
    own check because it returns a distinct error message on cancellation.
    """

    @functools.wraps(fn)
    async def wrapper(state: ExecutionState) -> dict[str, Any]:
        if await _check_cancelled(state["execution_id"]):
            return {"cancelled": True}
        return await fn(state)

    return wrapper


async def agent_node(state: ExecutionState) -> dict[str, Any]:
    """Core AI agent node — calls LLM and returns updated messages."""
    execution_id = state["execution_id"]

    if await _check_cancelled(execution_id):
        return {"cancelled": True, "error": "Execution cancelled by user"}

    settings = get_settings()
    if not settings.openai_api_key and not settings.anthropic_api_key:
        error = "No LLM API key configured"
        await emit_log(execution_id, error, level="error")
        return {"error": error}


    from app.core.llm import make_chat_model

    await emit_log(execution_id, "Agent node: calling LLM", level="info", node="agent")
    await emit_agent_thought(execution_id, "agent", "Thinking...")

    llm = make_chat_model(state["metadata"].get("llm_model"), streaming=False)

    # Bind tools if the workflow metadata defines them
    tool_definitions = state["metadata"].get("tools", [])
    bound_tools = _build_langchain_tools(tool_definitions)
    if bound_tools:
        llm = llm.bind_tools(bound_tools)

    try:
        response: AIMessage = await llm.ainvoke(state["messages"])
        await emit_agent_thought(
            execution_id, "agent", response.content[:500] if isinstance(response.content, str) else "..."
        )
        return {"messages": [response], "iteration": state["iteration"] + 1}
    except Exception as exc:
        error = f"LLM call failed: {exc}"
        await emit_log(execution_id, error, level="error")
        return {"error": error}


@cancellable_node
async def mcp_node(state: ExecutionState) -> dict[str, Any]:
    """MCP tool invocation node."""
    execution_id = state["execution_id"]
    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    from langchain_core.messages import ToolMessage

    from app.modules.mcp.invoker import invoke_tool

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_call_id = tool_call.get("id", str(uuid.uuid4()))

        await emit(execution_id, ev.MCP_INVOCATION, {
            "tool_name": tool_name,
            "input": tool_input,
        })

        try:
            result = await invoke_tool(
                execution_id=execution_id,
                tool_name=tool_name,
                tool_input=tool_input,
                org_id=state["org_id"],
            )
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )
            await emit(execution_id, ev.MCP_RESULT, {"tool_name": tool_name, "result": result})
        except Exception as exc:
            err_msg = f"MCP tool {tool_name!r} failed: {exc}"
            tool_messages.append(
                ToolMessage(content=err_msg, tool_call_id=tool_call_id)
            )
            await emit(execution_id, ev.MCP_ERROR, {"tool_name": tool_name, "error": str(exc)})

    return {"messages": tool_messages}


@cancellable_node
async def research_node(state: ExecutionState) -> dict[str, Any]:
    """Internet research node — Tavily + Firecrawl pipeline."""
    execution_id = state["execution_id"]
    metadata = state["metadata"]
    query = metadata.get("research_query") or _extract_last_user_query(state["messages"])
    if not query:
        return {}

    await emit(execution_id, ev.RESEARCH_STARTED, {"query": query})

    from langchain_core.messages import ToolMessage

    from app.modules.research.pipeline import ResearchPipeline

    pipeline = ResearchPipeline()
    try:
        result = await pipeline.run(query=query, execution_id=execution_id)
        summary = result.get("summary", "No summary available")
        await emit(execution_id, ev.RESEARCH_COMPLETE, {"query": query, "sources_count": len(result.get("sources", []))})
        return {
            "messages": [ToolMessage(content=summary, tool_call_id="research")],
            "tool_results": {**state["tool_results"], "research": result},
        }
    except Exception as exc:
        await emit_log(execution_id, f"Research failed: {exc}", level="error")
        return {"error": str(exc)}


@cancellable_node
async def sandbox_node(state: ExecutionState) -> dict[str, Any]:
    """Code execution node — runs code in an E2B sandbox."""
    execution_id = state["execution_id"]
    code = state["metadata"].get("code_to_execute") or _extract_code_from_messages(state["messages"])
    if not code:
        return {}

    await emit(execution_id, ev.SANDBOX_STARTED, {"code_length": len(code)})

    from langchain_core.messages import ToolMessage

    from app.modules.sandbox.e2b_client import E2BClient

    client = E2BClient()
    try:
        result = await client.execute_code(
            code=code,
            execution_id=execution_id,
        )
        await emit(execution_id, ev.SANDBOX_COMPLETE, {"exit_code": result.get("exit_code")})
        output = result.get("stdout", "") or result.get("stderr", "")
        return {
            "messages": [ToolMessage(content=output[:4000], tool_call_id="sandbox")],
            "tool_results": {**state["tool_results"], "sandbox": result},
        }
    except Exception as exc:
        await emit_log(execution_id, f"Sandbox failed: {exc}", level="error")
        return {"error": str(exc)}


@cancellable_node
async def device_node(state: ExecutionState) -> dict[str, Any]:
    """Device command node — dispatches commands to Grafux-devices."""
    execution_id = state["execution_id"]
    metadata = state["metadata"]
    device_id = metadata.get("device_id")
    command = metadata.get("device_command")
    if not device_id or not command:
        return {}

    await emit(execution_id, ev.DEVICE_COMMAND_SENT, {
        "device_id": device_id,
        "command": command,
    })

    from langchain_core.messages import ToolMessage

    from app.modules.devices.client import DevicesClient

    client = DevicesClient()
    try:
        result = await client.send_command(
            device_id=device_id,
            command=command,
            params=metadata.get("device_params", {}),
        )
        return {
            "messages": [ToolMessage(content=str(result), tool_call_id="device")],
            "tool_results": {**state["tool_results"], "device": result},
        }
    except Exception as exc:
        await emit_log(execution_id, f"Device command failed: {exc}", level="error")
        return {"error": str(exc)}


# ── Routing logic ─────────────────────────────────────────────────────────────

def should_continue(state: ExecutionState) -> str:
    """Conditional edge: decide what node to visit after agent_node."""
    from langgraph.graph import END

    if state.get("cancelled") or state.get("error"):
        return END

    if state["iteration"] >= get_settings().max_execution_steps:
        log.warning("max_steps_reached", execution_id=state["execution_id"])
        return END

    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    tool_name = last_message.tool_calls[0]["name"] if last_message.tool_calls else ""
    return route_tool_to_node(tool_name)


# ── WorkflowEngine ────────────────────────────────────────────────────────────

class WorkflowEngine:
    """Builds and executes LangGraph StateGraphs for workflow runs."""

    def build_graph(self, graph_definition: dict[str, Any] | None = None):
        """Compile a LangGraph StateGraph.

        When ``graph_definition`` is None, builds the default agentic graph
        (agent → conditional routing → tool nodes → back to agent).
        """
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(ExecutionState)

        # Always include the core nodes
        g.add_node("agent", agent_node)
        g.add_node("mcp", mcp_node)
        g.add_node("research", research_node)
        g.add_node("sandbox", sandbox_node)
        g.add_node("device", device_node)

        # LangGraph 1.x: set_entry_point is removed; use add_edge(START, ...)
        g.add_edge(START, "agent")

        g.add_conditional_edges(
            "agent",
            should_continue,
            {
                "mcp": "mcp",
                "research": "research",
                "sandbox": "sandbox",
                "device": "device",
                END: END,
            },
        )

        # All tool nodes route back to the agent
        for node in ("mcp", "research", "sandbox", "device"):
            g.add_edge(node, "agent")

        return g.compile()

    async def validate_graph(self, graph_definition: dict[str, Any]) -> None:
        """Validate a graph definition. Raises ValueError if invalid."""
        required_fields = ["entry_node", "nodes"]
        for field in required_fields:
            if field not in graph_definition:
                raise ValueError(f"Graph definition missing required field: {field!r}")

    async def run(
        self,
        execution_id: str,
        input_data: dict[str, Any],
        org_id: str,
        project_id: str,
        workflow_id: str | None = None,
        graph_definition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a workflow graph with a PostgreSQL checkpointer.

        Returns the final execution state.
        """
        from app.modules.workflow.checkpoints import get_postgres_checkpointer
        from app.prompts import get_system_prompt as _get_prompt
        initial_user_message = input_data.get("message") or input_data.get("prompt", "")
        messages: list[BaseMessage] = []
        system_prompt_text = input_data.get("system_prompt") or _get_prompt("build_app")
        if system_prompt_text:
            messages.append(SystemMessage(content=system_prompt_text))
        if initial_user_message:
            messages.append(HumanMessage(content=initial_user_message))

        initial_state = ExecutionState(
            execution_id=execution_id,
            org_id=org_id,
            project_id=project_id,
            workflow_id=workflow_id,
            messages=messages,
            current_node="agent",
            tool_results={},
            agent_memory={},
            iteration=0,
            metadata=input_data.get("metadata", {}),
            error=None,
            cancelled=False,
        )

        config = {"configurable": {"thread_id": execution_id}}

        await emit(execution_id, ev.EXECUTION_STARTED, {
            "org_id": org_id,
            "project_id": project_id,
            "workflow_id": workflow_id,
        })

        async with get_postgres_checkpointer() as checkpointer:
            graph = self.build_graph(graph_definition)
            graph_with_memory = graph.with_config({"checkpointer": checkpointer})

            final_state: ExecutionState = await graph_with_memory.ainvoke(
                initial_state, config=config
            )

        error = final_state.get("error")
        cancelled = final_state.get("cancelled", False)

        last_ai_message = next(
            (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        output_text = ""
        if last_ai_message and isinstance(last_ai_message.content, str):
            output_text = last_ai_message.content

        result = {
            "output": output_text,
            "tool_results": final_state.get("tool_results", {}),
            "iterations": final_state.get("iteration", 0),
            "error": error,
            "cancelled": cancelled,
        }

        if cancelled:
            await emit(execution_id, ev.EXECUTION_CANCELLED, result)
        elif error:
            await emit(execution_id, ev.EXECUTION_FAILED, result)
        else:
            await emit(execution_id, ev.EXECUTION_COMPLETE, result)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_last_user_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _extract_code_from_messages(messages: list[BaseMessage]) -> str:
    """Extract the most recent code block from the message history."""
    for msg in reversed(messages):
        content = msg.content if isinstance(msg.content, str) else ""
        if "```python" in content:
            start = content.find("```python") + 9
            end = content.find("```", start)
            if end > start:
                return content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                return content[start:end].strip()
    return ""


def _build_langchain_tools(tool_definitions: list[dict]) -> list:
    """Build LangChain-compatible tool objects from a list of tool definitions."""
    tools = []
    for td in tool_definitions:
        tool_type = td.get("type", "mcp")
        if tool_type == "mcp":
            from langchain_core.tools import StructuredTool
            def make_mcp_tool(name: str, description: str):
                async def _run(**kwargs: Any) -> str:
                    from app.modules.mcp.invoker import invoke_tool
                    result = await invoke_tool(
                        execution_id="",
                        tool_name=name,
                        tool_input=kwargs,
                        org_id="",
                    )
                    return str(result)
                return StructuredTool.from_function(
                    coroutine=_run,
                    name=name,
                    description=description,
                )
            tools.append(make_mcp_tool(td["name"], td.get("description", "")))
    return tools
