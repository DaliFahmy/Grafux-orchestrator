from __future__ import annotations

import hashlib
import json

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis_client

log = get_logger("core.security")

_AUTH_CACHE_PREFIX = "auth:token:"
_INTERNAL_KEY_HEADER = "X-Internal-Service-Key"
_INTERNAL_NAME_HEADER = "X-Service-Name"
_SERVICE_NAME = "grafux-orchestrator"

# In-process JWKS cache: issuer → list of JWK dicts
_jwks_cache: dict[str, list[dict]] = {}


def _token_cache_key(token: str) -> str:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return f"{_AUTH_CACHE_PREFIX}{token_hash}"


async def _fetch_jwks(issuer: str) -> list[dict]:
    """Fetch Supabase JWKS keys from the issuer's well-known endpoint."""
    if issuer in _jwks_cache:
        return _jwks_cache[issuer]
    # Supabase JWKS: strip trailing /auth/v1 if present, then append the path
    base = issuer.rstrip("/")
    jwks_url = f"{base}/.well-known/jwks.json"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(jwks_url)
        resp.raise_for_status()
    keys: list[dict] = resp.json().get("keys", [])
    _jwks_cache[issuer] = keys
    log.info("jwks_fetched", issuer=issuer, key_count=len(keys))
    return keys


async def _verify_supabase_jwt(token: str) -> dict:
    """Verify a Supabase JWT (HS256 or ES256) and return the payload.

    Raises on invalid/expired token.
    """
    import jwt as pyjwt

    header = pyjwt.get_unverified_header(token)
    alg = header.get("alg", "HS256")

    settings = get_settings()

    if alg == "HS256":
        return pyjwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )

    if alg == "ES256":
        # Extract issuer from unverified claims to find the JWKS URL
        unverified = pyjwt.decode(
            token,
            options={"verify_signature": False, "verify_aud": False},
            algorithms=["ES256"],
        )
        issuer: str = unverified.get("iss", "")
        if not issuer:
            raise ValueError("Token missing iss claim")

        keys = await _fetch_jwks(issuer)
        kid = header.get("kid")
        key = next((k for k in keys if k.get("kid") == kid), keys[0] if keys else None)
        if not key:
            raise ValueError("No matching JWK found for token kid")

        from jwt.algorithms import ECAlgorithm
        public_key = ECAlgorithm.from_jwk(key)
        return pyjwt.decode(
            token,
            public_key,
            algorithms=["ES256"],
            options={"verify_aud": False},
        )

    raise ValueError(f"Unsupported JWT algorithm: {alg}")


async def verify_jwt(token: str) -> dict | None:
    """Validate a Supabase JWT directly (ES256/HS256 via JWKS).

    Returns the decoded payload dict on success, None on failure.
    Results are cached in Redis for ``auth_cache_ttl`` seconds.
    """
    if not token:
        return None

    settings = get_settings()
    redis = get_redis_client()
    cache_key = _token_cache_key(token)

    # Check Redis cache first
    try:
        cached = await redis.get(cache_key)
    except Exception as _redis_exc:
        log.warning("redis_cache_error", error=str(_redis_exc))
        cached = None

    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    # Verify the JWT directly — no backend roundtrip
    try:
        payload = await _verify_supabase_jwt(token)
    except Exception as exc:
        log.warning("jwt_verification_failed", error=str(exc))
        return None

    result: dict = {
        "user_id": payload.get("sub"),
        "email": payload.get("email", ""),
        "valid": True,
    }

    # Cache the result
    try:
        await redis.setex(cache_key, settings.auth_cache_ttl, json.dumps(result))
    except Exception:
        pass

    return result


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
