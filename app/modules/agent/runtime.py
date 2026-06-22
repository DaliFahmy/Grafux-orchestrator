from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.core.logging import get_logger
from app.modules.agent.memory import AgentMemory
from app.modules.agent.schemas import AgentConfig
from app.modules.streaming.event_bus import emit_agent_thought

log = get_logger("agent.runtime")


class AgentRuntime:
    """Creates and runs LangChain ReAct agents with memory and tool support."""

    def __init__(self) -> None:
        self._memory = AgentMemory()

    async def run(
        self,
        config: AgentConfig,
        messages: list[dict[str, Any]],
        execution_id: str,
        org_id: str,
    ) -> dict[str, Any]:
        settings = get_settings()
        if not settings.openai_api_key and not settings.anthropic_api_key:
            return {"error": "No LLM API key configured", "output": ""}

        # langgraph-prebuilt is a required dep of langgraph>=1.0.0
        from langgraph.prebuilt import create_react_agent

        from app.core.llm import make_chat_model

        # Load agent memory
        memory_data = await self._memory.load(execution_id, config.agent_id)

        # Build messages
        lc_messages: list[BaseMessage] = []
        if config.system_prompt:
            lc_messages.append(SystemMessage(content=config.system_prompt))
        # Inject memory context into system prompt if available
        if memory_data.get("context"):
            lc_messages.append(
                SystemMessage(content=f"Previous context:\n{memory_data['context']}")
            )
        for msg in messages:
            if msg.get("role") == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        # Build tools
        tools = self._build_tools(config.tools, execution_id, org_id)

        llm = make_chat_model(config.model)

        agent = create_react_agent(llm, tools)

        await emit_agent_thought(execution_id, config.agent_id, "Agent starting...")

        try:
            result = await agent.ainvoke({"messages": lc_messages})
            last_message = result["messages"][-1]
            output = last_message.content if isinstance(last_message.content, str) else str(last_message.content)

            # Update memory with this interaction
            await self._memory.update(execution_id, config.agent_id, {
                "context": output[:1000],
                "last_run": __import__("datetime").datetime.utcnow().isoformat(),
            })

            return {"output": output, "messages": result["messages"]}
        except Exception as exc:
            log.error("agent_runtime_error", error=str(exc), execution_id=execution_id)
            return {"error": str(exc), "output": ""}

    def _build_tools(
        self,
        tool_definitions: list[dict[str, Any]],
        execution_id: str,
        org_id: str,
    ) -> list:
        """Build LangChain tool objects from definitions."""
        from langchain_core.tools import StructuredTool

        tools = []
        for td in tool_definitions:
            tool_type = td.get("type", "mcp")
            name = td.get("name", "unknown_tool")
            description = td.get("description", "")

            if tool_type == "mcp":
                def make_mcp(tname: str, tdesc: str) -> StructuredTool:
                    async def _run(**kwargs: Any) -> str:
                        from app.modules.mcp.invoker import invoke_tool
                        result = await invoke_tool(
                            execution_id=execution_id,
                            tool_name=tname,
                            tool_input=kwargs,
                            org_id=org_id,
                        )
                        return str(result)
                    return StructuredTool.from_function(
                        coroutine=_run,
                        name=tname,
                        description=tdesc,
                    )
                tools.append(make_mcp(name, description))

            elif tool_type == "research":
                async def _research(query: str) -> str:
                    from app.modules.research.pipeline import ResearchPipeline
                    result = await ResearchPipeline().run(query=query, execution_id=execution_id)
                    return result.get("summary", "No results")
                tools.append(StructuredTool.from_function(
                    coroutine=_research,
                    name="web_search",
                    description="Search the web for information",
                ))

            elif tool_type == "sandbox":
                async def _sandbox(code: str) -> str:
                    from app.modules.sandbox.e2b_client import E2BClient
                    result = await E2BClient().execute_code(code=code, execution_id=execution_id)
                    return result.get("stdout", "") or result.get("stderr", "Error")
                tools.append(StructuredTool.from_function(
                    coroutine=_sandbox,
                    name="execute_code",
                    description="Execute Python code in a sandbox",
                ))

        return tools
