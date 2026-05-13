from __future__ import annotations

import time

from fastapi import APIRouter

from app.dependencies import CurrentUser
from app.modules.devices.client import DevicesClient
from app.modules.devices.schemas import DeviceCommand, DeviceCommandResponse

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/command", response_model=DeviceCommandResponse)
async def send_device_command(
    body: DeviceCommand,
    user: CurrentUser,
) -> DeviceCommandResponse:
    client = DevicesClient()
    start = time.monotonic()
    try:
        result = await client.send_command(
            device_id=body.device_id,
            command=body.command,
            params=body.params,
            timeout=body.timeout,
        )
        return DeviceCommandResponse(
            device_id=body.device_id,
            command=body.command,
            result=result,
            status="success",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:
        return DeviceCommandResponse(
            device_id=body.device_id,
            command=body.command,
            result=None,
            status="error",
            duration_ms=int((time.monotonic() - start) * 1000),
            error=str(exc),
        )


@router.get("/{device_id}/status")
async def get_device_status(device_id: str, user: CurrentUser) -> dict:
    client = DevicesClient()
    return await client.get_device_status(device_id)


@router.get("")
async def list_devices(user: CurrentUser) -> dict:
    client = DevicesClient()
    org_id = user.get("org_id", "")
    devices = await client.list_devices(org_id)
    return {"devices": devices}
