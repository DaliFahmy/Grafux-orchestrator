from __future__ import annotations

from app.modules.research.schemas import SearchResult


def extract_citations(sources: list[SearchResult]) -> list[str]:
    """Generate citation strings from search results."""
    citations = []
    for i, source in enumerate(sources, 1):
        title = source.title or "Untitled"
        url = source.url
        date = f" ({source.published_date})" if source.published_date else ""
        citations.append(f"[{i}] {title}{date} — {url}")
    return citations


def format_sources_for_context(sources: list[SearchResult], max_chars: int = 8000) -> str:
    """Build a context block from search results for LLM consumption."""
    parts = []
    total_chars = 0
    for i, source in enumerate(sources, 1):
        snippet = source.content[:1500] if source.content else ""
        entry = f"[{i}] {source.title}\nURL: {source.url}\n{snippet}\n"
        if total_chars + len(entry) > max_chars:
            break
        parts.append(entry)
        total_chars += len(entry)
    return "\n---\n".join(parts)
