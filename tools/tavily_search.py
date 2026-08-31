"""Tavily web search integration for GovResearch-AI."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import chardet
import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from config.settings import Settings


LOGGER = logging.getLogger(__name__)


class TavilySearchError(Exception):
    """Raised when Tavily search fails."""

    pass


def fetch_clean_content(url: str) -> dict:
    """Fetch content and title from URL with proper encoding detection.
    
    Args:
        url: URL to fetch content from.
        
    Returns:
        Dict with status and content information.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
    
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)
    
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True, verify=False)
        raw = resp.content
        
        # Check for Cloudflare protection
        if b"Just a moment" in raw[:2000]:
            return {
                "url": url,
                "status": "cloudflare_blocked",
                "title": "",
                "content": ""
            }
        
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        
        LOGGER.info(f"[Content Fetch] Detected encoding: {encoding} for URL: {url[:50]}...")
        
        # Try multiple encodings for Korean content
        encodings_to_try = [encoding, "utf-8", "cp949", "euc-kr"]
        text = ""
        
        for enc in encodings_to_try:
            try:
                text = raw.decode(enc, errors="strict")
                LOGGER.info(f"[Content Fetch] Successfully decoded with {enc}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if not text:
            LOGGER.error(f"[Content Fetch] Failed to decode with any encoding")
            return {
                "url": url,
                "status": "decode_failed",
                "title": "",
                "content": ""
            }
        
        # Extract title using BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        title = soup.title.string if soup.title else ""
        
        # Extract main content (remove script, style, etc.)
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        content = soup.get_text(separator=' ', strip=True)
        
        return {
            "url": url,
            "status": "success",
            "title": title,
            "content": content
        }
        
    except httpx.ConnectTimeout:
        return {
            "url": url,
            "status": "connect_timeout",
            "title": "",
            "content": ""
        }
    except httpx.ReadTimeout:
        return {
            "url": url,
            "status": "read_timeout",
            "title": "",
            "content": ""
        }
    except httpx.SSLError as e:
        return {
            "url": url,
            "status": f"ssl_error: {e}",
            "title": "",
            "content": ""
        }
    except Exception as e:
        return {
            "url": url,
            "status": f"error: {type(e).__name__}: {e}",
            "title": "",
            "content": ""
        }


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


def search_web(query: str, settings: Settings, max_results: int = None) -> dict:
    """Perform web search using Tavily and return normalized results.

    Args:
        query: Search query string.
        settings: Application settings with Tavily configuration.
        max_results: Optional override for max_results from settings.

    Returns:
        Normalized search results with query and classified results.

    Raises:
        TavilySearchError: If Tavily API key is missing or request fails.
    """
    if not settings.tavily_api_key:
        raise TavilySearchError("Tavily API key is not configured.")

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        actual_max_results = max_results if max_results is not None else settings.search_max_results
        LOGGER.info("Tavily search started: query='%s', max_results=%d", query, actual_max_results)

        # Domain preferences for high-quality sources (prioritize Korean domains for Korean content)
        include_domains = [
            # Korean government domains
            "gov.kr", "go.kr", "or.kr", "ac.kr", "kr",
            # Specific Korean government agencies
            "msit.go.kr", "mois.go.kr", "mofe.go.kr", "mohw.go.kr", "molit.go.kr",
            "me.go.kr", "moe.go.kr", "mke.go.kr", "mcst.go.kr", "mafra.go.kr",
            "korea.kr", "open.go.kr",
            # Korean research institutions
            "spr.or.kr", "nia.or.kr", "kisa.or.kr", "etri.re.kr", "kaist.ac.kr",
            "kisti.re.kr", "krii.re.kr",
            # International organizations (for comparison)
            "oecd.org", "gov.uk", "govtech.sg",
            "usa.gov", "gov.sg", "gouv.fr"
        ]
        
        # Exclude low-quality domains
        exclude_domains = [
            "wikipedia.org",
            "blogspot.com",
            "wordpress.com",
            "reddit.com",
            "quora.com",
            "youtube.com"
        ]

        response = client.search(
            query=query,
            max_results=actual_max_results,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,  # Don't use Tavily's content fetching
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            # Add search context for better Korean results
            search_context="Korean government public institutions AI services artificial intelligence",
        )

        results = response.get("results", [])
        LOGGER.info("Tavily search completed: %d URLs discovered", len(results))

        # Fetch clean content from each URL using proper encoding detection
        normalized_results = []
        status_counts = {}
        
        for result in results:
            url = result.get("url", "")
            
            # Fetch clean title and content from the URL
            fetch_result = fetch_clean_content(url)
            status = fetch_result["status"]
            
            # Count status types
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Handle different fetch statuses
            if status == "success":
                normalized_result = {
                    "title": fetch_result["title"],
                    "url": url,
                    "content": fetch_result["content"],
                    "score": result.get("score", 0.0),
                    "source_type": classify_source(url),
                }
                normalized_results.append(normalized_result)
            elif status in ["cloudflare_blocked", "ssl_error"]:
                # Use Tavily's data as fallback for these cases
                LOGGER.info(f"[Content Fetch] Using Tavily fallback for {status}: {url[:50]}...")
                normalized_result = {
                    "title": result.get("title", ""),
                    "url": url,
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "source_type": classify_source(url),
                }
                normalized_results.append(normalized_result)
            else:
                LOGGER.warning(f"[Content Fetch] Skipping URL due to {status}: {url[:50]}...")
        
        # Log status summary
        LOGGER.info(f"[Content Fetch] Status summary: {status_counts}")
        LOGGER.info(f"[Content Fetch] Successfully fetched {len(normalized_results)} out of {len(results)} URLs")

        return {
            "query": query,
            "results": normalized_results,
        }

    except Exception as e:
        LOGGER.error("Tavily search failed: %s", str(e), exc_info=True)
        raise TavilySearchError(f"Tavily search failed: {str(e)}") from e
