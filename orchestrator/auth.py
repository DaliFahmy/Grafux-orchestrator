from __future__ import annotations

import logging

import jwt

from .config import Config

log = logging.getLogger("orchestrator.auth")


class AuthService:
    """Validates JWT tokens for incoming WebSocket connections."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def decode_token(self, token: str) -> dict | None:
        """Decode and validate a JWT token.

        Returns the payload dict on success, or None on failure.
        When JWT_SECRET is not configured, allows anonymous connections.
        """
        if not self._config.jwt_secret:
            log.warning("JWT_SECRET not set – allowing unauthenticated connections")
            return {"sub": "anonymous", "username": "anonymous"}

        if not token:
            return None

        try:
            return jwt.decode(token, self._config.jwt_secret, algorithms=["HS256"])
        except jwt.PyJWTError as exc:
            log.warning("JWT decode failed: %s", exc)
            return None
