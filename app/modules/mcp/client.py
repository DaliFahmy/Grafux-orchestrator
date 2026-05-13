from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("mcp.client")

_INTERNAL_HEADER = "X-Internal-Service-Secret"


class MCPClient:
    """HTTP client for communicating with the Grafux-mcp service."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.mcp_service_url
        self._secret = settings.internal_service_secret
        self._timeout = 30.0

    def _headers(self) -> dict[str, str]:
        return {_INTERNAL_HEADER: self._secret, "Content-Type": "application/json"}

    async def list_tools(self) -> list[dict[str, Any]]:
        """Retrieve all available MCP tools from Grafux-mcp."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}/tools",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("tools", [])

    async def invoke_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        execution_id: str = "",
    ) -> Any:
        """Invoke a single MCP tool and return its output."""
        payload = {
            "tool": tool_name,
            "input": tool_input,
            "execution_id": execution_id,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/invoke",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("output", result)
