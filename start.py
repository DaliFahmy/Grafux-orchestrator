#!/usr/bin/env python3
"""
Grafux Orchestrator startup script.

Validates required env vars, runs Alembic migrations, then execs uvicorn.
Using a Python script (instead of shell &&) gives us clear error messages
and cross-platform behaviour on Render.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(
            f"\nFATAL: Required environment variable {name!r} is not set.\n"
            f"  → Set it in the Render dashboard under Environment → Environment Variables.\n"
            f"  → For DATABASE_URL use the Internal Database URL from your PostgreSQL service.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


def main() -> None:
    # ── Validate required env vars before touching the database ───────────────
    db_url = _require_env("DATABASE_URL")

    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        print(
            "WARNING: REDIS_URL is not set — event streaming and Celery workers "
            "will not function.",
            file=sys.stderr,
        )

    # Log enough of the URL to confirm it's non-local without leaking secrets
    host_hint = db_url.split("@")[-1].split("/")[0] if "@" in db_url else "unknown"
    print(f"[start] DATABASE_URL host: {host_hint}", flush=True)

    # ── Run Alembic migrations ─────────────────────────────────────────────────
    print("[start] Running database migrations ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env=os.environ,
    )
    if result.returncode != 0:
        print("\nFATAL: Alembic migrations failed. Check the output above.\n", file=sys.stderr)
        sys.exit(result.returncode)

    print("[start] Migrations complete.", flush=True)

    # ── Start uvicorn (os.execvp replaces this process — no zombie left) ──────
    port = os.environ.get("PORT", "8000")
    workers = os.environ.get("WEB_CONCURRENCY", "2")
    cmd = [
        "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", workers,
    ]
    print(f"[start] Launching: {' '.join(cmd)}", flush=True)
    os.execvp("uvicorn", cmd)


if __name__ == "__main__":
    main()
