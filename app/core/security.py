from __future__ import annotations

import hashlib

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis_client

log = get_logger("core.security")

_AUTH_CACHE_PREFIX = "auth:token:"
_INTERNAL_KEY_HEADER = "X-Internal-Service-Key"
_INTERNAL_NAME_HEADER = "X-Service-Name"
_SERVICE_NAME = "grafux-orchestrator"


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

    # #region agent log
    import json as _json_mod, time as _time_mod
    def _dbg_write(msg, data, hyp):
        log.info(f"[DBG-0f2551] {msg}", hypothesis=hyp, **data)
        try:
            with open("debug-0f2551.log", "a") as _f:
                _f.write(_json_mod.dumps({
                    "sessionId": "0f2551",
                    "id": f"log_{int(_time_mod.time()*1000)}_sec",
                    "timestamp": int(_time_mod.time() * 1000),
                    "location": "core/security.py:verify_jwt",
                    "message": msg,
                    "data": data,
                    "runId": "post-fix",
                    "hypothesisId": hyp,
                }) + "\n")
        except Exception:
            pass
    _dbg_write("verify_jwt_called", {
        "backend_url": settings.backend_url,
        "target_url": f"{settings.backend_url}/api/internal/auth/validate",
        "token_prefix": token[:20] + "...",
    }, "H-B")
    # #endregion

    redis = get_redis_client()
    cache_key = _token_cache_key(token)

    # Check Redis cache first
    # #region agent log
    try:
        cached = await redis.get(cache_key)
    except Exception as _redis_exc:
        _dbg_write("redis_cache_error", {"error": str(_redis_exc)}, "H-D")
        cached = None
    else:
        _dbg_write("redis_cache_result", {"hit": cached is not None}, "H-D")
    # #endregion
    if cached:
        import json
        try:
            return json.loads(cached)
        except Exception:
            pass

    # Delegate validation to Grafux-backend
    # Correct endpoint: POST /api/internal/auth/validate (token in body)
    target_url = f"{settings.backend_url}/api/internal/auth/validate"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                target_url,
                headers={
                    _INTERNAL_KEY_HEADER: settings.internal_service_secret,
                    _INTERNAL_NAME_HEADER: _SERVICE_NAME,
                },
                json={"token": token},
            )
            # #region agent log
            _dbg_write("backend_auth_response", {
                "url": target_url,
                "http_status": resp.status_code,
                "body_snippet": resp.text[:200],
            }, "H-B")
            # #endregion
            if resp.status_code != 200:
                log.warning("jwt_verification_failed", status=resp.status_code, url=target_url)
                return None

            body = resp.json()
            data = body.get("data", {})
            if not data.get("valid"):
                log.warning("jwt_invalid", url=target_url)
                return None

            payload: dict = {
                "user_id": data.get("user_id"),
                "email": data.get("email"),
                "valid": True,
            }

    except httpx.RequestError as exc:
        # #region agent log
        _dbg_write("backend_auth_request_error", {"error": str(exc), "url": target_url}, "H-C")
        # #endregion
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
