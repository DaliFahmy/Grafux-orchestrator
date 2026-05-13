from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.core.redis import get_redis_client

log = get_logger("agent.memory")

_MEMORY_TTL = 3600 * 24  # 24 hours


def _memory_key(execution_id: str, agent_id: str) -> str:
    return f"agent_memory:{execution_id}:{agent_id}"


class AgentMemory:
    """Short-term agent memory backed by Redis + long-term in PostgreSQL."""

    async def load(self, execution_id: str, agent_id: str) -> dict[str, Any]:
        redis = get_redis_client()
        raw = await redis.get(_memory_key(execution_id, agent_id))
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        # Fall back to DB if not in Redis
        return await self._load_from_db(execution_id, agent_id)

    async def save(
        self, execution_id: str, agent_id: str, memory: dict[str, Any]
    ) -> None:
        redis = get_redis_client()
        await redis.setex(
            _memory_key(execution_id, agent_id),
            _MEMORY_TTL,
            json.dumps(memory),
        )
        await self._save_to_db(execution_id, agent_id, memory)

    async def update(
        self,
        execution_id: str,
        agent_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        current = await self.load(execution_id, agent_id)
        current.update(updates)
        await self.save(execution_id, agent_id, current)
        return current

    async def clear(self, execution_id: str, agent_id: str) -> None:
        redis = get_redis_client()
        await redis.delete(_memory_key(execution_id, agent_id))

    async def _load_from_db(
        self, execution_id: str, agent_id: str
    ) -> dict[str, Any]:
        try:
            from sqlalchemy import select
            from app.core.database import get_db_session
            from app.modules.persistence.models import AgentSession
            async with get_db_session() as db:
                result = await db.execute(
                    select(AgentSession).where(
                        AgentSession.execution_id == execution_id,
                        AgentSession.agent_id == agent_id,
                    )
                )
                session = result.scalar_one_or_none()
                return session.memory if session else {}
        except Exception as exc:
            log.warning("agent_memory_db_load_failed", error=str(exc))
            return {}

    async def _save_to_db(
        self, execution_id: str, agent_id: str, memory: dict[str, Any]
    ) -> None:
        try:
            from sqlalchemy import select
            from app.core.database import get_db_session
            from app.modules.persistence.models import AgentSession
            async with get_db_session() as db:
                result = await db.execute(
                    select(AgentSession).where(
                        AgentSession.execution_id == execution_id,
                        AgentSession.agent_id == agent_id,
                    )
                )
                session = result.scalar_one_or_none()
                if session:
                    session.memory = memory
                else:
                    db.add(AgentSession(
                        execution_id=execution_id,
                        agent_id=agent_id,
                        memory=memory,
                    ))
                await db.commit()
        except Exception as exc:
            log.warning("agent_memory_db_save_failed", error=str(exc))
