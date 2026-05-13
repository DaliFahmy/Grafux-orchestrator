from __future__ import annotations

from typing import Any

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class TriggerExecutionRequest(OrchestratorBaseModel):
    workflow_id: str | None = None
    org_id: str
    project_id: str
    triggered_by: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    graph_definition: dict[str, Any] | None = None  # inline graph, no workflow_id required


class TriggerExecutionResponse(OrchestratorBaseModel):
    execution_id: str
    status: str = "pending"


class CancelExecutionResponse(OrchestratorBaseModel):
    execution_id: str
    status: str = "cancelled"


class ExecutionStatusResponse(OrchestratorBaseModel):
    execution_id: str
    status: str
    workflow_id: str | None
    org_id: str
    project_id: str
    retry_count: int
    started_at: str | None
    finished_at: str | None
    error_message: str | None


class HealthResponse(OrchestratorBaseModel):
    status: str
    service: str
    version: str
    database: str
    redis: str
    celery: str
    environment: str
