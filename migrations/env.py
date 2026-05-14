from __future__ import annotations

import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import all models so Alembic can detect them
from app.core.database import Base
import app.modules.persistence.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _to_sync_psycopg_url(url: str) -> str:
    """Normalize any PostgreSQL URL to use the psycopg v3 sync driver.

    Handles the three URL forms that commonly arrive via env vars:
      postgres://...               → postgresql+psycopg://...  (Render default)
      postgresql://...             → postgresql+psycopg://...
      postgresql+asyncpg://...     → postgresql+psycopg://...  (our async app URL)
      postgresql+psycopg://...     → unchanged (already correct)
    """
    url = re.sub(r"^postgres://", "postgresql://", url)
    url = re.sub(r"^postgresql\+asyncpg://", "postgresql://", url)
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


# URL resolution priority:
#   1. DATABASE_SYNC_URL  — explicit psycopg URL if set
#   2. DATABASE_URL       — any PostgreSQL URL (Render sets this automatically
#                           when a PostgreSQL database is linked to the service)
#   3. alembic.ini        — local development fallback
raw_url = (
    os.environ.get("DATABASE_SYNC_URL")
    or os.environ.get("DATABASE_URL")
    or config.get_main_option("sqlalchemy.url")
)

db_url = _to_sync_psycopg_url(raw_url)
config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
