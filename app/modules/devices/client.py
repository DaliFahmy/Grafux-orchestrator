from __future__ import annotations

import time
from typing import Any

from app.config import get_settings
from app.core.http_client import get_http_client
from app.core.logging import get_logger
from app.core.resilience import retry_transient

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
            resp = await get_http_client().post(
                f"{self._base_url}/commands",
                headers=self._headers(),
                json=payload,
                timeout=timeout,
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

    @retry_transient()
    async def get_device_status(self, device_id: str) -> dict[str, Any]:
        """Fetch the current status of a device (idempotent — safe to retry)."""
        try:
            resp = await get_http_client().get(
                f"{self._base_url}/devices/{device_id}/status",
                headers=self._headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("device_status_failed", device_id=device_id, error=str(exc))
            raise

    async def scaffold_claw(
        self,
        description: str,
        name: str = "",
        connections: list[str] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Draft a claw block's design ports (soul/task/connections/…) from a description.

        Wraps the devices ``POST /claw/scaffold`` endpoint (which AI-drafts the ports and
        normalizes ``connections`` to valid Composio toolkit slugs). Best-effort: returns ``{}``
        on any failure so the caller can fall back to the empty scaffold.
        """
        payload: dict[str, Any] = {"description": description, "name": name}
        if connections:
            payload["connections"] = connections
        try:
            resp = await get_http_client().post(
                f"{self._base_url}/claw/scaffold",
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("claw_scaffold_failed", name=name, error=str(exc))
            return {}

    async def list_claw_toolkits(self, timeout: float = 10.0) -> list[str]:
        """List the Composio toolkit slugs a claw can connect to (idempotent — safe to retry).

        Wraps the devices ``GET /claw/toolkits`` endpoint. Returns ``[]`` on any failure.
        """
        try:
            resp = await get_http_client().get(
                f"{self._base_url}/claw/toolkits",
                headers=self._headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("toolkits", [])
        except Exception as exc:
            log.warning("claw_toolkits_failed", error=str(exc))
            return []

    async def list_devices(self, org_id: str) -> list[dict[str, Any]]:
        """List all devices for an organisation."""
        try:
            resp = await get_http_client().get(
                f"{self._base_url}/devices",
                headers=self._headers(),
                params={"org_id": org_id},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get("devices", [])
        except Exception as exc:
            log.error("device_list_failed", org_id=org_id, error=str(exc))
            return []
