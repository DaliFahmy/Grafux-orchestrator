from __future__ import annotations

from app.config import get_settings
from app.core.logging import get_logger
from app.modules.research.schemas import SearchResult

log = get_logger("research.tavily")


class TavilyClient:
    """Async wrapper around the Tavily search API (tavily-python >= 0.5)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.tavily_api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_raw_content: bool = False,
        search_depth: str = "basic",
    ) -> list[SearchResult]:
        if not self._api_key:
            log.warning("tavily_api_key_not_configured")
            return []

        try:
            # tavily-python 0.5+ provides AsyncTavilyClient
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=self._api_key)
            response = await client.search(
                query=query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                search_depth=search_depth,
            )
        except ImportError:
            # Fallback: try synchronous client in executor (older SDK)
            import asyncio
            from tavily import TavilyClient as SyncClient
            sync_client = SyncClient(api_key=self._api_key)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sync_client.search(
                    query=query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    search_depth=search_depth,
                ),
            )
        except Exception as exc:
            log.error("tavily_search_failed", query=query, error=str(exc))
            return []

        results = []
        for r in (response.get("results") if isinstance(response, dict) else []):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", "") or r.get("raw_content", ""),
                    score=float(r.get("score", 0.0)),
                    published_date=r.get("published_date"),
                )
            )

        log.info("tavily_search_complete", query=query, results=len(results))
        return results
