"""Default-value factories shared by ORM models and Pydantic schemas.

A single home for id/timestamp generation so the persistence models and the
shared base models don't each carry their own private copies.
"""

from __future__ import annotations

import uuid
from datetime import datetime


def generate_uuid() -> str:
    """Return a new random UUID4 as a string."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Return the current naive UTC timestamp (matches existing column defaults)."""
    return datetime.utcnow()
