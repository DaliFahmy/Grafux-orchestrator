from __future__ import annotations

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class SandboxExecuteRequest(OrchestratorBaseModel):
    code: str
    language: str = "python"
    execution_id: str | None = None
    timeout: int = Field(default=60, ge=1, le=300)
    packages: list[str] = Field(default_factory=list)


class SandboxExecuteResponse(OrchestratorBaseModel):
    session_id: str
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None = None
    duration_ms: int
