from __future__ import annotations

import hashlib

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis_client

log = get_logger("core.security")

_AUTH_CACHE_PREFIX = "auth:token:"
_INTERNAL_HEADER = "X-Internal-Service-Secret"


def _token_cache_key(token: str) -> str:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return f"{_AUTH_CACHE_PREFIX}{token_hash}"


async def verify_jwt(token: str) -> dict | None:
    """Validate a JWT by delegating to Grafux-backend.

    Returns the decoded payload dict on success, None on failure.
    Results are cached in Redis for ``auth_cache_ttl`` seconds.
    """
    if not token:
        return None

    settings = get_settings()
    redis = get_redis_client()
    cache_key = _token_cache_key(token)

    # Check Redis cache first
    cached = await redis.get(cache_key)
    if cached:
        import json
        try:
            return json.loads(cached)
        except Exception:
            pass

    # Delegate validation to Grafux-backend
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.backend_url}/internal/auth/verify",
                headers={
                    "Authorization": f"Bearer {token}",
                    _INTERNAL_HEADER: settings.internal_service_secret,
                },
            )
            if resp.status_code != 200:
                log.warning(
                    "jwt_verification_failed",
                    status=resp.status_code,
                )
                return None

            payload: dict = resp.json()

    except httpx.RequestError as exc:
        log.error("backend_auth_unreachable", error=str(exc))
        return None

    # Cache the result
    import json
    await redis.setex(cache_key, settings.auth_cache_ttl, json.dumps(payload))
    return payload


def verify_internal_secret(secret: str) -> bool:
    """Verify an internal service-to-service request."""
    settings = get_settings()
    if not settings.internal_service_secret:
        log.warning("internal_service_secret_not_configured")
        return False
    return secret == settings.internal_service_secret


async def invalidate_token_cache(token: str) -> None:
    """Remove a token from the auth cache (e.g., on logout)."""
    redis = get_redis_client()
    await redis.delete(_token_cache_key(token))
