from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.modules.sandbox.e2b_client import E2BClient
from app.modules.sandbox.schemas import SandboxExecuteRequest, SandboxExecuteResponse

log = get_logger("sandbox.session")


class SandboxSessionManager:
    """High-level sandbox session orchestration."""

    def __init__(self) -> None:
        self._client = E2BClient()

    async def execute(self, request: SandboxExecuteRequest) -> SandboxExecuteResponse:
        import time
        start = time.monotonic()

        result = await self._client.execute_code(
            code=request.code,
            execution_id=request.execution_id or "",
            timeout=request.timeout,
            packages=request.packages,
        )

        return SandboxExecuteResponse(
            session_id=result["session_id"],
            exit_code=result.get("exit_code"),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            error=result.get("error"),
            duration_ms=result.get("duration_ms", int((time.monotonic() - start) * 1000)),
        )
