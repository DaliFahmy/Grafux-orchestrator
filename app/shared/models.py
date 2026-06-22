from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.shared.factories import generate_uuid  # re-exported for existing imports

__all__ = [
    "generate_uuid",
    "OrchestratorBaseModel",
    "TimestampedModel",
    "IDModel",
    "BaseResponse",
    "ErrorResponse",
]


class OrchestratorBaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class TimestampedModel(OrchestratorBaseModel):
    created_at: datetime | None = None
    updated_at: datetime | None = None


class IDModel(OrchestratorBaseModel):
    id: str = Field(default_factory=generate_uuid)


class BaseResponse(OrchestratorBaseModel):
    success: bool = True
    message: str = "OK"


class ErrorResponse(OrchestratorBaseModel):
    success: bool = False
    error: str
    message: str
