from __future__ import annotations

from typing import Any

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class AgentConfig(OrchestratorBaseModel):
    agent_id: str
    name: str = "Grafux Agent"
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    memory_enabled: bool = True
    max_iterations: int = 20


class AgentRunRequest(OrchestratorBaseModel):
    execution_id: str
    agent_config: AgentConfig
    messages: list[dict[str, Any]] = Field(default_factory=list)
    org_id: str
    project_id: str


class AgentMemoryUpdate(OrchestratorBaseModel):
    execution_id: str
    agent_id: str
    memory: dict[str, Any]


class MultiAgentConfig(OrchestratorBaseModel):
    agents: list[AgentConfig]
    coordinator_model: str = "gpt-4o"
    execution_id: str
    task: str
    org_id: str
    project_id: str
