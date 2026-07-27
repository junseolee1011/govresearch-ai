"""Webpage content extraction for enriching search results."""

import logging
import re
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def extract_article_text(url: str, timeout: int = 10) -> tuple[str, str]:
    """Extract main article text from a webpage.
    
    Args:
        url: URL to fetch and extract content from.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (extracted_text, error_message). 
        If successful, error_message is empty string.
    """
    try:
        # Fetch the webpage with encoding detection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Try to detect encoding from response
        if response.encoding == 'ISO-8859-1':
            # Common fallback for misconfigured servers
            response.encoding = response.apparent_encoding
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
            element.decompose()
        
        # Remove common navigation/menu classes
        for element in soup.find_all(class_=re.compile(r"(nav|menu|sidebar|footer|header)", re.I)):
            element.decompose()
        
        # Get text from main content areas
        # Try to find main content containers
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=re.compile(r"(content|article|post|body)", re.I)) or
            soup.body
        )
        
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text, ""
        
    except requests.RequestException as e:
        error_msg = f"Failed to fetch {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg
    except Exception as e:
        error_msg = f"Failed to parse {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg


def is_download_url(url: str) -> bool:
    """Check if URL is a file download link.
    
    Args:
        url: URL to check.
        
    Returns:
        True if URL appears to be a download link.
    """
    url_lower = url.lower()
    download_indicators = [
        "filedown.do",
        "download.do",
        ".pdf",
        ".hwp",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".zip",
        ".rar",
        "atchfileid",
        "fn=",
        "rs=",
    ]
    
    for indicator in download_indicators:
        if indicator in url_lower:
            return True
    
    return False


def enrich_search_results(search_results: list[dict], min_length: int = 300) -> list[dict]:
    """Enrich search results with full webpage content.
    
    Args:
        search_results: List of search result dictionaries.
        min_length: Minimum character length for extracted body text.
        
    Returns:
        List of enriched search results with extracted content.
        Documents with insufficient content are filtered out.
    """
    enriched_results = []
    
    for result in search_results:
        url = result.get("url", "")
        original_content = result.get("content", "")
        
        # Skip file download URLs
        if is_download_url(url):
            LOGGER.warning(f"Skipping download URL: {url}")
            continue
        
        # Skip if we already have sufficient content
        if len(original_content) >= min_length:
            LOGGER.info(f"Skipping {url}: already has sufficient content")
            enriched_results.append(result)
            continue
        
        LOGGER.info(f"Processing URL: {url}")
        LOGGER.info(f"Original snippet length: {len(original_content)} characters")
        
        # Extract full article text
        extracted_text, error = extract_article_text(url)
        
        if error:
            LOGGER.warning(f"Skipping {url} due to extraction error: {error}")
            continue
        
        LOGGER.info(f"Extracted body length: {len(extracted_text)} characters")
        
        # Discard if too short
        if len(extracted_text) < min_length:
            LOGGER.warning(f"Skipping {url}: extracted content too short ({len(extracted_text)} < {min_length})")
            continue
        
        # Enrich the result
        enriched_result = result.copy()
        enriched_result["content"] = extracted_text
        enriched_result["original_snippet"] = original_content
        enriched_results.append(enriched_result)
    
    LOGGER.info(f"Enriched {len(enriched_results)} out of {len(search_results)} documents")
    
    return enriched_results
