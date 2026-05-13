from __future__ import annotations

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("research.firecrawl")


class FirecrawlClient:
    """Wrapper around the Firecrawl web scraping API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.firecrawl_api_key

    async def scrape(self, url: str, formats: list[str] | None = None) -> dict:
        """Scrape a URL and return markdown + metadata."""
        if not self._api_key:
            log.warning("firecrawl_api_key_not_configured")
            return {"markdown": "", "metadata": {}}

        try:
            import asyncio
            from firecrawl import FirecrawlApp

            app = FirecrawlApp(api_key=self._api_key)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: app.scrape_url(
                    url,
                    params={"formats": formats or ["markdown"]},
                ),
            )
            log.info("firecrawl_scraped", url=url)
            return result or {}
        except Exception as exc:
            log.error("firecrawl_scrape_failed", url=url, error=str(exc))
            return {"markdown": "", "metadata": {}, "error": str(exc)}

    async def crawl(self, url: str, max_pages: int = 5) -> list[dict]:
        """Crawl a site and return pages as markdown."""
        if not self._api_key:
            return []
        try:
            import asyncio
            from firecrawl import FirecrawlApp

            app = FirecrawlApp(api_key=self._api_key)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: app.crawl_url(url, params={"limit": max_pages}),
            )
            data = result.get("data", []) if result else []
            log.info("firecrawl_crawled", url=url, pages=len(data))
            return data
        except Exception as exc:
            log.error("firecrawl_crawl_failed", url=url, error=str(exc))
            return []
