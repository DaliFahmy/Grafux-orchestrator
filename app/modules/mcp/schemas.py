from __future__ import annotations

from typing import Any

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class MCPTool(OrchestratorBaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    server: str = ""


class MCPInvokeRequest(OrchestratorBaseModel):
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    execution_id: str | None = None
    timeout: float = 30.0


class MCPInvokeResponse(OrchestratorBaseModel):
    tool_name: str
    output: Any
    duration_ms: int
    error: str | None = None
