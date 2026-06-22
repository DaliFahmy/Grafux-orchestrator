"""Business logic for the devices module.

Wraps ``DevicesClient`` (raw HTTP to Grafux-devices) with the command timing,
response shaping, and error handling the router used to inline, so handlers stay
thin (parse request → call service → return).
"""

from __future__ import annotations

import time

from app.modules.devices.client import DevicesClient
from app.modules.devices.schemas import DeviceCommand, DeviceCommandResponse


class DeviceService:
    """Dispatches device commands and status queries via Grafux-devices."""

    def __init__(self, client: DevicesClient | None = None) -> None:
        self._client = client or DevicesClient()

    async def send_command(self, body: DeviceCommand) -> DeviceCommandResponse:
        """Send a command and wrap the outcome (timed) in a response model.

        A failed dispatch becomes a ``status="error"`` response rather than an
        exception, matching the previous endpoint contract.
        """
        start = time.monotonic()
        try:
            result = await self._client.send_command(
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

    async def get_status(self, device_id: str) -> dict:
        """Return the current status of a device."""
        return await self._client.get_device_status(device_id)

    async def list_devices(self, org_id: str) -> dict:
        """List the organisation's devices, wrapped in the response envelope."""
        devices = await self._client.list_devices(org_id)
        return {"devices": devices}


# Module-level singleton — the underlying client holds only config and shares the
# pooled HTTP client, so one instance is safe to reuse across requests.
_service = DeviceService()


def get_device_service() -> DeviceService:
    """FastAPI-friendly accessor for the shared ``DeviceService``."""
    return _service
