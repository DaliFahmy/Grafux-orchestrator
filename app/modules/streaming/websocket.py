from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket

from app.core.logging import get_logger

log = get_logger("streaming.websocket")


class ConnectionManager:
    """Manages active WebSocket connections, keyed by execution_id."""

    def __init__(self) -> None:
        # execution_id → set of WebSocket connections
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, execution_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[execution_id].add(websocket)
        log.info(
            "ws_connected",
            execution_id=execution_id,
            total=len(self._connections[execution_id]),
        )

    def disconnect(self, execution_id: str, websocket: WebSocket) -> None:
        self._connections[execution_id].discard(websocket)
        if not self._connections[execution_id]:
            del self._connections[execution_id]
        log.info("ws_disconnected", execution_id=execution_id)

    async def send_to_execution(self, execution_id: str, message: str) -> None:
        """Fan-out a text message to all subscribers for the given execution."""
        dead: list[WebSocket] = []
        for ws in set(self._connections.get(execution_id, [])):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(execution_id, ws)

    async def broadcast(self, message: str) -> None:
        """Send to all connected clients across all executions."""
        tasks = []
        for execution_id in list(self._connections):
            tasks.append(self.send_to_execution(execution_id, message))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def active_executions(self) -> list[str]:
        return list(self._connections)

    def connection_count(self, execution_id: str) -> int:
        return len(self._connections.get(execution_id, set()))


# Singleton shared across the FastAPI process
connection_manager = ConnectionManager()
