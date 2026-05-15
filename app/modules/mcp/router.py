from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import CurrentUser, require_internal_service

_internal = Depends(require_internal_service)
from app.modules.mcp.schemas import MCPInvokeRequest, MCPInvokeResponse, MCPTool
from app.modules.mcp.service import MCPService

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/tools", response_model=list[MCPTool])
async def list_tools(user: CurrentUser) -> list[MCPTool]:
    service = MCPService()
    return await service.list_tools()


@router.post("/invoke", response_model=MCPInvokeResponse)
async def invoke_tool(
    body: MCPInvokeRequest,
    user: CurrentUser,
) -> MCPInvokeResponse:
    service = MCPService()
    org_id = user.get("org_id", "")
    return await service.invoke(body, org_id=org_id)


@router.post(
    "/tools/refresh",
    dependencies=[_internal],
)
async def refresh_tool_cache() -> dict:
    from app.modules.mcp.discovery import MCPDiscovery
    discovery = MCPDiscovery()
    await discovery.invalidate_cache()
    tools = await discovery.get_tools(force_refresh=True)
    return {"refreshed": True, "tool_count": len(tools)}
