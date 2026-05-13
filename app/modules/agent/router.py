from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import CurrentUser
from app.modules.agent.schemas import AgentRunRequest, MultiAgentConfig
from app.modules.agent.service import AgentService

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run")
async def run_agent(body: AgentRunRequest, user: CurrentUser) -> dict:
    service = AgentService()
    result = await service.run_agent(body)
    return {"result": result.get("output", ""), "error": result.get("error")}


@router.post("/multi")
async def run_multi_agent(body: MultiAgentConfig, user: CurrentUser) -> dict:
    service = AgentService()
    return await service.run_multi_agent(body)


@router.get("/{execution_id}/{agent_id}/memory")
async def get_agent_memory(
    execution_id: str,
    agent_id: str,
    user: CurrentUser,
) -> dict:
    service = AgentService()
    memory = await service.get_memory(execution_id, agent_id)
    return {"execution_id": execution_id, "agent_id": agent_id, "memory": memory}


@router.delete("/{execution_id}/{agent_id}/memory")
async def clear_agent_memory(
    execution_id: str,
    agent_id: str,
    user: CurrentUser,
) -> dict:
    service = AgentService()
    await service.clear_memory(execution_id, agent_id)
    return {"cleared": True}
