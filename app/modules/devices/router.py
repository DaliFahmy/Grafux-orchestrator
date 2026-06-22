from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import CurrentUser
from app.modules.devices.schemas import DeviceCommand, DeviceCommandResponse
from app.modules.devices.service import get_device_service

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/command", response_model=DeviceCommandResponse)
async def send_device_command(
    body: DeviceCommand,
    user: CurrentUser,
) -> DeviceCommandResponse:
    return await get_device_service().send_command(body)


@router.get("/{device_id}/status")
async def get_device_status(device_id: str, user: CurrentUser) -> dict:
    return await get_device_service().get_status(device_id)


@router.get("")
async def list_devices(user: CurrentUser) -> dict:
    org_id = user.get("org_id", "")
    return await get_device_service().list_devices(org_id)
