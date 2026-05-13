from __future__ import annotations

from typing import Any

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class DeviceCommand(OrchestratorBaseModel):
    device_id: str
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
    execution_id: str | None = None
    timeout: float = 30.0


class DeviceCommandResponse(OrchestratorBaseModel):
    device_id: str
    command: str
    result: Any
    status: str
    duration_ms: int
    error: str | None = None


class DeviceEvent(OrchestratorBaseModel):
    device_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
