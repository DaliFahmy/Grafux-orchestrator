from __future__ import annotations

from typing import Any

from app.modules.agent.coordinator import MultiAgentCoordinator
from app.modules.agent.memory import AgentMemory
from app.modules.agent.runtime import AgentRuntime
from app.modules.agent.schemas import AgentRunRequest, MultiAgentConfig


class AgentService:
    def __init__(self) -> None:
        self._runtime = AgentRuntime()
        self._memory = AgentMemory()
        self._coordinator = MultiAgentCoordinator()

    async def run_agent(self, request: AgentRunRequest) -> dict[str, Any]:
        return await self._runtime.run(
            config=request.agent_config,
            messages=request.messages,
            execution_id=request.execution_id,
            org_id=request.org_id,
        )

    async def run_multi_agent(self, config: MultiAgentConfig) -> dict[str, Any]:
        return await self._coordinator.run(config)

    async def get_memory(self, execution_id: str, agent_id: str) -> dict[str, Any]:
        return await self._memory.load(execution_id, agent_id)

    async def update_memory(
        self,
        execution_id: str,
        agent_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._memory.update(execution_id, agent_id, updates)

    async def clear_memory(self, execution_id: str, agent_id: str) -> None:
        await self._memory.clear(execution_id, agent_id)
