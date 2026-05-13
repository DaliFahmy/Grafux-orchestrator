from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import CurrentUser
from app.modules.research.pipeline import ResearchPipeline
from app.modules.research.schemas import ResearchRequest, ResearchResponse

router = APIRouter(prefix="/research", tags=["research"])


@router.post("", response_model=ResearchResponse)
async def run_research(
    body: ResearchRequest,
    user: CurrentUser,
) -> ResearchResponse:
    pipeline = ResearchPipeline()
    result = await pipeline.run(
        query=body.query,
        execution_id=body.execution_id or "",
        max_results=body.max_results,
        crawl_top=body.crawl_top_results,
    )
    from app.modules.research.schemas import SearchResult
    return ResearchResponse(
        query=result["query"],
        sources=[SearchResult(**s) for s in result["sources"]],
        summary=result["summary"],
        citations=result["citations"],
        execution_id=body.execution_id,
    )
