from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import get_logger
from app.modules.research.citations import extract_citations, format_sources_for_context
from app.modules.research.firecrawl_client import FirecrawlClient
from app.modules.research.schemas import ResearchResponse, SearchResult
from app.modules.research.tavily_client import TavilyClient

log = get_logger("research.pipeline")


class ResearchPipeline:
    """Orchestrates: search → crawl top results → summarize with LLM."""

    def __init__(self) -> None:
        self._tavily = TavilyClient()
        self._firecrawl = FirecrawlClient()

    async def run(
        self,
        query: str,
        execution_id: str = "",
        max_results: int = 5,
        crawl_top: int = 2,
    ) -> dict[str, Any]:
        settings = get_settings()
        log.info("research_pipeline_started", query=query, execution_id=execution_id)

        # Step 1: Web search
        sources = await self._tavily.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
        )

        if not sources:
            log.warning("no_search_results", query=query)
            return {"query": query, "sources": [], "summary": "No results found.", "citations": []}

        # Step 2: Deep crawl top N results for richer content
        if crawl_top > 0:
            top_urls = [s.url for s in sources[:crawl_top]]
            for url in top_urls:
                scraped = await self._firecrawl.scrape(url)
                md = scraped.get("markdown", "")
                if md:
                    for source in sources:
                        if source.url == url:
                            source.content = md[:3000]
                            break

        # Step 3: Summarize with LLM
        context = format_sources_for_context(sources)
        summary = await self._summarize(query, context, settings)
        citations = extract_citations(sources)

        # Persist to DB
        if execution_id:
            await self._persist(execution_id, query, sources, summary, citations)

        log.info("research_pipeline_complete", query=query, sources=len(sources))
        return {
            "query": query,
            "sources": [s.model_dump() for s in sources],
            "summary": summary,
            "citations": citations,
        }

    async def _summarize(self, query: str, context: str, settings: Any) -> str:
        if not settings.openai_api_key:
            return context[:2000]

        prompt = (
            f"Research query: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Based on the sources above, provide a comprehensive, accurate, "
            "and well-structured answer to the research query. "
            "Include key findings, cite source numbers where relevant."
        )
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    json={
                        "model": settings.openai_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
                        "temperature": 0.3,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.error("summarization_failed", error=str(exc))
        return context[:2000]

    async def _persist(
        self,
        execution_id: str,
        query: str,
        sources: list[SearchResult],
        summary: str,
        citations: list[str],
    ) -> None:
        try:
            from app.core.database import get_db_session
            from app.modules.persistence.models import ResearchResult
            async with get_db_session() as db:
                db.add(ResearchResult(
                    execution_id=execution_id,
                    query=query,
                    sources=[s.model_dump() for s in sources],
                    summary=summary,
                    citations=citations,
                ))
                await db.commit()
        except Exception as exc:
            log.warning("research_persist_failed", error=str(exc))
