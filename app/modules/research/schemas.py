from __future__ import annotations

from typing import Any

from pydantic import Field

from app.shared.models import OrchestratorBaseModel


class SearchResult(OrchestratorBaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: str | None = None


class ResearchRequest(OrchestratorBaseModel):
    query: str
    execution_id: str | None = None
    max_results: int = Field(default=5, ge=1, le=20)
    include_raw_content: bool = False
    crawl_top_results: int = Field(default=2, ge=0, le=5)


class ResearchResponse(OrchestratorBaseModel):
    query: str
    sources: list[SearchResult]
    summary: str
    citations: list[str] = Field(default_factory=list)
    execution_id: str | None = None
