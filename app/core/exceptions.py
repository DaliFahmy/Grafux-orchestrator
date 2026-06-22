from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class OrchestratorError(Exception):
    """Base exception for the orchestrator service."""

    def __init__(self, message: str, code: str = "orchestrator_error") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class ExecutionNotFoundError(OrchestratorError):
    def __init__(self, execution_id: str) -> None:
        super().__init__(f"Execution {execution_id!r} not found", "execution_not_found")


class WorkflowNotFoundError(OrchestratorError):
    def __init__(self, workflow_id: str) -> None:
        super().__init__(f"Workflow {workflow_id!r} not found", "workflow_not_found")


class ExecutionAlreadyRunningError(OrchestratorError):
    def __init__(self, execution_id: str) -> None:
        super().__init__(
            f"Execution {execution_id!r} is already running",
            "execution_already_running",
        )


class ExecutionCancelledError(OrchestratorError):
    def __init__(self, execution_id: str) -> None:
        super().__init__(
            f"Execution {execution_id!r} has been cancelled",
            "execution_cancelled",
        )


class SandboxError(OrchestratorError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "sandbox_error")


class MCPError(OrchestratorError):
    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"MCP tool {tool!r} failed: {message}", "mcp_error")


class ResearchError(OrchestratorError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "research_error")


class AuthorizationError(OrchestratorError):
    def __init__(self, message: str = "Not authorized") -> None:
        super().__init__(message, "authorization_error")


class RateLimitError(OrchestratorError):
    def __init__(self) -> None:
        super().__init__("Rate limit exceeded", "rate_limit_exceeded")


# ── FastAPI Exception Handlers ────────────────────────────────────────────────

async def orchestrator_error_handler(
    request: Request, exc: OrchestratorError
) -> JSONResponse:
    status_map = {
        "execution_not_found": status.HTTP_404_NOT_FOUND,
        "workflow_not_found": status.HTTP_404_NOT_FOUND,
        "execution_already_running": status.HTTP_409_CONFLICT,
        "authorization_error": status.HTTP_403_FORBIDDEN,
        "rate_limit_exceeded": status.HTTP_429_TOO_MANY_REQUESTS,
    }
    http_status = status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    return JSONResponse(
        status_code=http_status,
        content={"error": exc.code, "message": exc.message},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return request-validation (422) errors in the house ``{error, message}`` schema.

    ``exc`` is a Starlette/FastAPI ``RequestValidationError``; its ``errors()`` list is
    attached under ``details`` so clients keep the per-field information.
    """
    details = exc.errors() if hasattr(exc, "errors") else None
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "message": "Request validation failed", "details": details},
    )
