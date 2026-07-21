"""Tavily web search integration for GovResearch-AI."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from tavily import TavilyClient

from config.settings import Settings


LOGGER = logging.getLogger(__name__)


class TavilySearchError(Exception):
    """Raised when Tavily search fails."""

    pass


def classify_source(url: str) -> str:
    """Classify source type based on domain.

    Args:
        url: URL to classify.

    Returns:
        Source type: government, research, news, or other.
    """
    domain = urlparse(url).netloc.lower()

    # Government domains
    if (
        domain.endswith(".go.kr")
        or domain.endswith(".gov")
        or domain.endswith("gov.sg")
        or domain.endswith("gov.uk")
    ):
        return "government"

    # Research domains
    if (
        domain.endswith(".ac.kr")
        or domain.endswith(".edu")
        or "arxiv.org" in domain
        or "oecd.org" in domain
        or "worldbank.org" in domain
    ):
        return "research"

    # News domains
    if (
        "reuters.com" in domain
        or "bloomberg.com" in domain
        or "nytimes.com" in domain
    ):
        return "news"

    return "other"


def search_web(query: str, settings: Settings) -> dict:
    """Perform web search using Tavily and return normalized results.

    Args:
        query: Search query string.
        settings: Application settings with Tavily configuration.

    Returns:
        Normalized search results with query and classified results.

    Raises:
        TavilySearchError: If Tavily API key is missing or request fails.
    """
    if not settings.tavily_api_key:
        raise TavilySearchError("Tavily API key is not configured.")

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        LOGGER.info("Tavily search started: query='%s', max_results=%d", query, settings.search_max_results)

        response = client.search(
            query=query,
            max_results=settings.search_max_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        LOGGER.info("Tavily search completed: %d results returned", len(response.get("results", [])))

        normalized_results = []
        for result in response.get("results", []):
            normalized_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "source_type": classify_source(result.get("url", "")),
            }
            normalized_results.append(normalized_result)

        return {
            "query": query,
            "results": normalized_results,
        }

    except Exception as e:
        LOGGER.error("Tavily search failed: %s", str(e), exc_info=True)
        raise TavilySearchError(f"Tavily search failed: {str(e)}") from e
