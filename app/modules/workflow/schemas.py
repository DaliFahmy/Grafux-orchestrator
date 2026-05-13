from __future__ import annotations

from typing import Any

from pydantic import Field

from app.shared.models import IDModel, OrchestratorBaseModel, TimestampedModel


class NodeDefinition(OrchestratorBaseModel):
    name: str
    type: str  # agent | mcp | research | sandbox | device | decision | custom
    config: dict[str, Any] = Field(default_factory=dict)


class EdgeDefinition(OrchestratorBaseModel):
    from_node: str
    to_node: str
    condition: str | None = None  # Python expression for conditional edges


class GraphDefinition(OrchestratorBaseModel):
    entry_node: str
    nodes: list[NodeDefinition]
    edges: list[EdgeDefinition]
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowCreate(OrchestratorBaseModel):
    name: str
    description: str | None = None
    graph_definition: GraphDefinition
    org_id: str
    project_id: str


class WorkflowResponse(IDModel, TimestampedModel):
    name: str
    description: str | None
    org_id: str
    project_id: str
    graph_definition: dict[str, Any]
    is_active: bool


class WorkflowTemplateResponse(IDModel, TimestampedModel):
    name: str
    category: str
    description: str | None
    graph_definition: dict[str, Any]


class ExecutionResponse(IDModel):
    workflow_id: str | None
    org_id: str
    project_id: str
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None
    error_message: str | None
    retry_count: int
    started_at: str | None
    finished_at: str | None
    created_at: str


class ExecutionLogResponse(OrchestratorBaseModel):
    id: str
    level: str
    message: str
    step_id: str | None
    metadata: dict[str, Any]
    timestamp: str
