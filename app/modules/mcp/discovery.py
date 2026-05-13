from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.core.redis import get_redis_client
from app.modules.mcp.client import MCPClient
from app.modules.mcp.schemas import MCPTool

log = get_logger("mcp.discovery")

_TOOLS_CACHE_KEY = "mcp:tools_cache"
_CACHE_TTL = 300  # 5 minutes


class MCPDiscovery:
    """Discovers and caches available MCP tools."""

    async def get_tools(self, force_refresh: bool = False) -> list[MCPTool]:
        redis = get_redis_client()

        if not force_refresh:
            cached = await redis.get(_TOOLS_CACHE_KEY)
            if cached:
                try:
                    tools_data = json.loads(cached)
                    return [MCPTool(**t) for t in tools_data]
                except Exception:
                    pass

        client = MCPClient()
        try:
            raw_tools = await client.list_tools()
            tools = [MCPTool(**t) for t in raw_tools]
            await redis.setex(_TOOLS_CACHE_KEY, _CACHE_TTL, json.dumps([t.model_dump() for t in tools]))
            log.info("mcp_tools_discovered", count=len(tools))
            return tools
        except Exception as exc:
            log.warning("mcp_discovery_failed", error=str(exc))
            return []

    async def get_tool(self, name: str) -> MCPTool | None:
        tools = await self.get_tools()
        return next((t for t in tools if t.name == name), None)

    async def invalidate_cache(self) -> None:
        redis = get_redis_client()
        await redis.delete(_TOOLS_CACHE_KEY)
