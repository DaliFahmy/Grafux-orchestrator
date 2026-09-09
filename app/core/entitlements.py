"""What a user's plan allows — resolved from Grafux-backend, cached in Redis.

Block agents are the most expensive thing a user can start (an LLM tool-call
loop that runs blocks, and for EDA types can occupy a machine for an hour), so
they are a paid capability. ``verify_jwt`` deliberately does no backend round
trip and returns only ``{user_id, email, valid}`` — there is no plan claim in
the token to read — so the answer has to be fetched.

THE FAILURE POLICY IS THE WHOLE POINT OF THIS FILE. Four different things can
happen, and collapsing them into "did the call succeed" gets one of them wrong:

    backend says entitled          -> allow, refresh both caches
    backend says NOT entitled      -> deny, and OVERWRITE the long-lived cache
                                      (a stale yes must never outlive a real no)
    backend unreachable (5xx/timeout) -> hot cache, then last-known-good, then
                                      the configured fallback
    backend rejects us (4xx)        -> treated as unreachable, logged loudly:
                                      this is our misconfiguration, and making
                                      the customer pay for it is not honest

The fallback defaults to "allow" because the two errors are not symmetric. An
outage that briefly grants free users an agent costs a little money. An outage
that blocks the customers paying $100/month looks exactly like the product being
broken, and they cannot tell the difference.
"""
from __future__ import annotations

import json

from app.config import get_settings
from app.core.http_client import get_http_client
from app.core.logging import get_logger
from app.core.redis import get_redis_client

log = get_logger(__name__)

# Keep this string stable: it crosses a process boundary into the backend's plan
# catalog and a version boundary into shipped desktop builds.
BLOCK_AGENT = "block_agent"

_HOT_KEY = "ent:user:{user_id}"
_LKG_KEY = "ent:lkg:user:{user_id}"


async def _cache_read(key: str) -> set[str] | None:
    try:
        raw = await get_redis_client().get(key)
    except Exception as exc:  # noqa: BLE001 — a cache outage is not an auth failure
        log.debug("entitlement_cache_read_failed", error=str(exc))
        return None
    if not raw:
        return None
    try:
        return set(json.loads(raw))
    except Exception:  # noqa: BLE001 — corrupt entry, re-resolve
        return None


async def _cache_write(user_id: str, entitlements: set[str]) -> None:
    settings = get_settings()
    payload = json.dumps(sorted(entitlements))
    redis = get_redis_client()
    for key, ttl in (
        (_HOT_KEY.format(user_id=user_id), settings.entitlement_cache_ttl),
        (_LKG_KEY.format(user_id=user_id), settings.entitlement_lkg_ttl),
    ):
        try:
            await redis.setex(key, ttl, payload)
        except Exception as exc:  # noqa: BLE001 — best effort
            log.debug("entitlement_cache_write_failed", error=str(exc))


async def _fetch(token: str, user_id: str) -> set[str] | None:
    """Ask the backend. None means "could not get an answer", not "no rights"."""
    settings = get_settings()
    url = f"{settings.backend_url.rstrip('/')}/api/billing/entitlements"
    try:
        resp = await get_http_client().get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("entitlement_fetch_failed", user_id=user_id, error=str(exc))
        return None

    if resp.status_code >= 500:
        log.warning("entitlement_backend_error", status=resp.status_code)
        return None
    if resp.status_code >= 400:
        # Our token handling or the route is wrong — a deployment problem, not a
        # statement about this user's plan. Never turn it into a refusal.
        log.error(
            "entitlement_request_rejected",
            status=resp.status_code,
            hint="orchestrator could not authenticate to the backend",
        )
        return None

    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("entitlement_bad_json", error=str(exc))
        return None

    # The backend wraps every payload in a {"data": ...} envelope.
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        log.warning("entitlement_unexpected_shape")
        return None

    return set(data.get("entitlements") or [])


async def get_entitlements(
    token: str, user_id: str, *, refresh: bool = False
) -> set[str]:
    """Resolve this user's entitlements, applying the policy in the module docstring.

    ``refresh=True`` skips the hot cache. Used on a denial so that a user who has
    just upgraded in the browser is not told to upgrade again for the rest of the
    cache TTL.
    """
    settings = get_settings()

    if not refresh:
        cached = await _cache_read(_HOT_KEY.format(user_id=user_id))
        if cached is not None:
            return cached

    fetched = await _fetch(token, user_id)
    if fetched is not None:
        # This branch covers the explicit "not entitled" answer too, which is why
        # it writes through to the long-lived cache: a downgrade must invalidate
        # the last-known-good, or a cancelled subscription keeps working for a
        # week.
        await _cache_write(user_id, fetched)
        return fetched

    lkg = await _cache_read(_LKG_KEY.format(user_id=user_id))
    if lkg is not None:
        log.info("entitlement_using_last_known_good", user_id=user_id)
        return lkg

    if settings.entitlement_fallback == "allow":
        log.warning("entitlement_unresolved_allowing", user_id=user_id)
        return {BLOCK_AGENT}

    log.warning("entitlement_unresolved_denying", user_id=user_id)
    return set()


async def may_run_block_agents(token: str, user_id: str) -> bool:
    """The gate. Honours the kill switch.

    With ``enforce_agent_entitlement`` off this always allows, and does not even
    make the HTTP call — but it still logs what it *would* have refused, which is
    how the rollout finds out that everyone currently resolves to free before
    that costs anyone their Agent button.
    """
    settings = get_settings()
    if not settings.enforce_agent_entitlement:
        return True
    entitlements = await get_entitlements(token, user_id)
    if BLOCK_AGENT in entitlements:
        return True
    # One more look, straight past the cache: the common case for a denial is a
    # user who has just paid.
    entitlements = await get_entitlements(token, user_id, refresh=True)
    return BLOCK_AGENT in entitlements


async def audit_block_agent_entitlement(token: str, user_id: str) -> None:
    """Log what the gate *would* do, without gating. For the enforcement-off phase."""
    settings = get_settings()
    if settings.enforce_agent_entitlement:
        return
    try:
        entitlements = await get_entitlements(token, user_id)
    except Exception as exc:  # noqa: BLE001 — auditing must never break a session
        log.debug("entitlement_audit_failed", error=str(exc))
        return
    if BLOCK_AGENT not in entitlements:
        log.info(
            "entitlement_would_refuse",
            user_id=user_id,
            hint="enforcement is off; this block agent would be refused if it were on",
        )
