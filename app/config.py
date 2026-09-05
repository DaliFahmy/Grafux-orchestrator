from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service Identity ──────────────────────────────────────────────────────
    service_name: str = "grafux-orchestrator"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    port: int = 8000

    # ── Grafux-backend ────────────────────────────────────────────────────────
    backend_url: str = "http://localhost:8001"
    internal_service_secret: str = "change_me"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/grafux_orchestrator"
    )
    # Used by LangGraph checkpointer (psycopg v3) and Alembic
    database_sync_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/grafux_orchestrator"
    )
    # Sized so concurrent workflow executions don't starve on connection checkout.
    database_pool_size: int = 20
    database_max_overflow: int = 20
    database_pool_timeout: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    # REDIS_URL can be set directly, or constructed from REDIS_HOST + REDIS_PORT
    # (Render Key Value exposes host/port as the most reliable fromService props).
    redis_url: str = ""
    redis_host: str = "localhost"
    redis_port: int = 6379
    # Sized for many concurrent WebSocket sessions each publishing execution events.
    redis_max_connections: int = 50

    # ── Outbound HTTP (shared pooled client) ──────────────────────────────────
    http_default_timeout: float = 30.0
    http_max_connections: int = 100
    http_max_keepalive_connections: int = 20

    # ── Celery ────────────────────────────────────────────────────────────────
    # Derived from redis_url in the validator below; can be overridden via env.
    celery_broker_url: str = ""
    celery_result_backend: str = ""
    celery_task_serializer: str = "json"
    celery_result_expires: int = 3600

    @model_validator(mode="after")
    def _derive_redis_and_celery_urls(self) -> Settings:
        # Build redis_url from host/port when not provided as a full URL
        if not self.redis_url:
            self.redis_url = f"redis://{self.redis_host}:{self.redis_port}/0"
        # Celery broker and backend default to the same Redis instance
        if not self.celery_broker_url:
            self.celery_broker_url = self.redis_url
        if not self.celery_result_backend:
            self.celery_result_backend = self.redis_url
        return self

    # ── AI ────────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str = ""
    gemini_live_model: str = "gemini-3.1-flash-live-preview"
    anthropic_api_key: str = ""               # env: ANTHROPIC_API_KEY
    # Default Claude model. Used when a caller asks for Anthropic without naming a
    # version -- notably a block agent whose model dropdown is unset. NOT the
    # default for an empty model id: that stays on openai_model, which is the
    # non-regression guarantee for every caller that sends no model at all.
    anthropic_model: str = "claude-opus-5"

    # ── External Services ─────────────────────────────────────────────────────
    e2b_api_key: str = ""
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""

    # ── Internal Service URLs ─────────────────────────────────────────────────
    mcp_service_url: str = "http://localhost:8002"
    devices_service_url: str = "http://localhost:8003"
    interaction_service_url: str = "http://localhost:8004"

    # ── Security ──────────────────────────────────────────────────────────────
    jwt_algorithm: str = "HS256"
    # Supabase JWT secret — used as HS256 fallback; ES256 tokens use JWKS auto-discovery
    supabase_jwt_secret: str = ""
    auth_cache_ttl: int = 60  # seconds to cache validated tokens in Redis

    # ── Execution Limits ──────────────────────────────────────────────────────
    max_execution_steps: int = 50
    max_execution_timeout: int = 3600
    sandbox_timeout: int = 300

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def asyncpg_url(self) -> str:
        """SQLAlchemy async URL using asyncpg driver.

        Normalises whatever form DATABASE_URL arrives in (including the
        ``postgres://`` scheme that Render provides by default).
        """
        import re
        url = self.database_url
        url = re.sub(r"^postgres://", "postgresql://", url)
        url = re.sub(r"^postgresql\+psycopg://", "postgresql://", url)
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url

    @property
    def psycopg_url(self) -> str:
        """psycopg v3 URL for LangGraph checkpointer and Alembic.

        Normalises whatever form DATABASE_URL or DATABASE_SYNC_URL arrives in.
        The checkpointer expects the raw ``postgresql://`` scheme without a
        driver prefix.
        """
        import re
        # Prefer the explicit sync URL if set to something other than default
        url = self.database_sync_url or self.database_url
        url = re.sub(r"^postgres://", "postgresql://", url)
        url = re.sub(r"^postgresql\+asyncpg://", "postgresql://", url)
        url = re.sub(r"^postgresql\+psycopg://", "postgresql://", url)
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
