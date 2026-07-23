"""Search result normalization for consistent data format."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from config.settings import Settings


LOGGER = logging.getLogger(__name__)


def normalize_search_result(result: dict) -> dict | None:
    """Normalize a single Tavily search result into a common format.
    
    Args:
        result: Raw Tavily search result.
        
    Returns:
        Normalized result or None if malformed.
    """
    try:
        # Extract required fields
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        
        # Validate required fields
        if not title or not url or not content:
            LOGGER.debug(f"Skipping malformed result: missing required fields")
            return None
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                LOGGER.debug(f"Skipping result with invalid URL: {url}")
                return None
        except Exception:
            LOGGER.debug(f"Skipping result with invalid URL: {url}")
            return None
        
        # Extract domain
        domain = parsed.netloc.lower()
        
        # Determine source type
        source_type = classify_source_type(domain)
        
        # Extract published date if available
        published_date = result.get("published_date", None)
        
        normalized = {
            "title": title,
            "url": url,
            "content": content,
            "domain": domain,
            "source_type": source_type,
            "published_date": published_date,
        }
        
        return normalized
        
    except Exception as e:
        LOGGER.warning(f"Failed to normalize result: {e}")
        return None


def classify_source_type(domain: str) -> str:
    """Classify source type based on domain.
    
    Args:
        domain: Domain name.
        
    Returns:
        Source type: government, research, news, corporate, or other.
    """
    # Government domains
    if (
        domain.endswith(".gov.kr")
        or domain.endswith(".go.kr")
        or domain.endswith(".or.kr")
        or domain.endswith(".gov")
        or "gov.sg" in domain
        or "gov.uk" in domain
        or "usa.gov" in domain
    ):
        return "government"
    
    # Research/academic domains
    if (
        domain.endswith(".ac.kr")
        or domain.endswith(".edu")
        or "arxiv.org" in domain
        or "oecd.org" in domain
        or "worldbank.org" in domain
        or "nature.com" in domain
        or "science.org" in domain
        or "ieee.org" in domain
        or "springer.com" in domain
    ):
        return "research"
    
    # News domains
    if (
        "reuters.com" in domain
        or "bloomberg.com" in domain
        or "nytimes.com" in domain
        or "ft.com" in domain
        or "economist.com" in domain
        or "wsj.com" in domain
        or "forbes.com" in domain
        or "techcrunch.com" in domain
    ):
        return "news"
    
    # Corporate domains
    if (
        "blogspot.com" in domain
        or "wordpress.com" in domain
        or "medium.com" in domain
        or "dev.to" in domain
    ):
        return "corporate"
    
    return "other"


def normalize_search_results(results: list[dict]) -> list[dict]:
    """Normalize all search results.
    
    Args:
        results: List of raw Tavily search results.
        
    Returns:
        List of normalized results.
    """
    normalized = []
    rejected = 0
    
    for result in results:
        norm = normalize_search_result(result)
        if norm:
            normalized.append(norm)
        else:
            rejected += 1
    
    LOGGER.info(f"Normalized {len(normalized)} results, rejected {rejected} malformed results")
    
    # Stage 2 Debug: Normalized results
    print("\n========================")
    print("Stage 2")
    print("========================")
    print(f"Number of normalized results: {len(normalized)}")
    if normalized:
        print("\nFirst normalized object:")
        print(f"  Title: {normalized[0].get('title', 'N/A')}")
        print(f"  URL: {normalized[0].get('url', 'N/A')}")
        print(f"  Domain: {normalized[0].get('domain', 'N/A')}")
        print(f"  Source Type: {normalized[0].get('source_type', 'N/A')}")
    print("========================\n")
    
    return normalized
