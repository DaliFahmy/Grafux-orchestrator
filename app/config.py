from __future__ import annotations

from functools import lru_cache
from typing import Literal

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
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_serializer: str = "json"
    celery_result_expires: int = 3600

    # ── AI ────────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str = ""

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
    def psycopg_url(self) -> str:
        """Raw psycopg v3 URL for LangGraph checkpointer (no driver prefix)."""
        url = self.database_sync_url
        if url.startswith("postgresql+psycopg://"):
            return "postgresql://" + url[len("postgresql+psycopg://"):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
