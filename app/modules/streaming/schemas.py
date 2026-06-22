from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class ExecutionEvent(OrchestratorBaseModel):
    """A single structured event emitted during workflow execution."""

    execution_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_json(self) -> str:
        import json
        return json.dumps(self.model_dump())

    @classmethod
    def from_json(cls, raw: str) -> ExecutionEvent:
        import json
        return cls(**json.loads(raw))


class LogEvent(OrchestratorBaseModel):
    execution_id: str
    level: str = "info"
    message: str
    step_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class AgentThoughtEvent(OrchestratorBaseModel):
    execution_id: str
    agent_id: str
    thought: str
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class StepEvent(OrchestratorBaseModel):
    execution_id: str
    step_id: str
    node_name: str
    status: str
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
