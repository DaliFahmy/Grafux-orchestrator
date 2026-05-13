from __future__ import annotations

from typing import Any

from app.modules.mcp.discovery import MCPDiscovery
from app.modules.mcp.invoker import invoke_tool
from app.modules.mcp.schemas import MCPInvokeRequest, MCPInvokeResponse, MCPTool


class MCPService:
    def __init__(self) -> None:
        self._discovery = MCPDiscovery()

    async def list_tools(self) -> list[MCPTool]:
        return await self._discovery.get_tools()

    async def invoke(
        self, request: MCPInvokeRequest, org_id: str
    ) -> MCPInvokeResponse:
        import time
        start = time.monotonic()
        try:
            output = await invoke_tool(
                execution_id=request.execution_id or "",
                tool_name=request.tool_name,
                tool_input=request.input,
                org_id=org_id,
                timeout=request.timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            return MCPInvokeResponse(
                tool_name=request.tool_name,
                output=output,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return MCPInvokeResponse(
                tool_name=request.tool_name,
                output=None,
                duration_ms=duration_ms,
                error=str(exc),
            )
