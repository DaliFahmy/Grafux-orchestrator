from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.redis import get_redis
from app.core.security import verify_internal_secret, verify_jwt

# ── Type aliases for cleaner signatures ───────────────────────────────────────

DBSession = Annotated[AsyncSession, Depends(get_session)]
RedisClient = Annotated[Redis, Depends(get_redis)]


# ── Auth dependencies ─────────────────────────────────────────────────────────

async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    """Validate Bearer JWT via Grafux-backend and return the user payload."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = authorization.removeprefix("Bearer ").strip()
    payload = await verify_jwt(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload


async def require_internal_service(
    x_internal_service_secret: Annotated[str | None, Header()] = None,
) -> None:
    """Ensure a request carries the internal service secret."""
    if not x_internal_service_secret or not verify_internal_secret(
        x_internal_service_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Internal service secret required",
        )


CurrentUser = Annotated[dict, Depends(get_current_user)]
InternalService = Annotated[None, Depends(require_internal_service)]
