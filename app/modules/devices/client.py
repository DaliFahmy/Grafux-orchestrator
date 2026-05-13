from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("devices.client")

_INTERNAL_HEADER = "X-Internal-Service-Secret"


class DevicesClient:
    """HTTP client for communicating with the Grafux-devices service."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.devices_service_url
        self._secret = settings.internal_service_secret

    def _headers(self) -> dict[str, str]:
        return {
            _INTERNAL_HEADER: self._secret,
            "Content-Type": "application/json",
        }

    async def send_command(
        self,
        device_id: str,
        command: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a command to a device via Grafux-devices."""
        start = time.monotonic()
        payload = {
            "device_id": device_id,
            "command": command,
            "params": params or {},
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/commands",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info(
                    "device_command_sent",
                    device_id=device_id,
                    command=command,
                    duration_ms=duration_ms,
                )
                return result
        except Exception as exc:
            log.error(
                "device_command_failed",
                device_id=device_id,
                command=command,
                error=str(exc),
            )
            raise

    async def get_device_status(self, device_id: str) -> dict[str, Any]:
        """Fetch the current status of a device."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/devices/{device_id}/status",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            log.error("device_status_failed", device_id=device_id, error=str(exc))
            raise

    async def list_devices(self, org_id: str) -> list[dict[str, Any]]:
        """List all devices for an organisation."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/devices",
                    headers=self._headers(),
                    params={"org_id": org_id},
                )
                resp.raise_for_status()
                return resp.json().get("devices", [])
        except Exception as exc:
            log.error("device_list_failed", org_id=org_id, error=str(exc))
            return []
