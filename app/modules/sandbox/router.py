from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import CurrentUser
from app.modules.sandbox.schemas import SandboxExecuteRequest, SandboxExecuteResponse
from app.modules.sandbox.session import SandboxSessionManager

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


@router.post("/execute", response_model=SandboxExecuteResponse)
async def execute_code(
    body: SandboxExecuteRequest,
    user: CurrentUser,
) -> SandboxExecuteResponse:
    """Execute code in an isolated E2B sandbox."""
    manager = SandboxSessionManager()
    return await manager.execute(body)
