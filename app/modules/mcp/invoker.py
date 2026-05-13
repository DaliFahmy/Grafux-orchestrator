from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.logging import get_logger
from app.modules.mcp.client import MCPClient

log = get_logger("mcp.invoker")


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def invoke_tool(
    execution_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    org_id: str,
    timeout: float = 30.0,
) -> Any:
    """Invoke an MCP tool with retry on transient network errors.

    Records the invocation in the database for audit purposes.
    """
    client = MCPClient()
    client._timeout = timeout
    start = time.monotonic()

    try:
        result = await client.invoke_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            execution_id=execution_id,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "mcp_tool_invoked",
            execution_id=execution_id,
            tool_name=tool_name,
            duration_ms=duration_ms,
        )

        # Fire-and-forget DB audit record
        if execution_id:
            try:
                from app.core.database import get_db_session
                from app.modules.persistence.models import MCPInvocation
                async with get_db_session() as db:
                    db.add(MCPInvocation(
                        execution_id=execution_id,
                        tool_name=tool_name,
                        input=tool_input,
                        output=result if isinstance(result, dict) else {"value": str(result)},
                        duration_ms=duration_ms,
                    ))
                    await db.commit()
            except Exception:
                pass  # audit failure must not block execution

        return result

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        log.error(
            "mcp_tool_failed",
            execution_id=execution_id,
            tool_name=tool_name,
            error=str(exc),
            duration_ms=duration_ms,
        )
        raise
