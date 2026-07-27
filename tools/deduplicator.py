"""Duplicate removal for research results."""

import logging
from typing import TYPE_CHECKING

from rapidfuzz import fuzz, process

if TYPE_CHECKING:
    from config.settings import Settings


LOGGER = logging.getLogger(__name__)


def remove_duplicates_by_url(sources: list[dict]) -> list[dict]:
    """Remove duplicate sources by URL.
    
    Args:
        sources: List of source dictionaries with 'url' field.
        
    Returns:
        Deduplicated list of sources.
    """
    seen_urls = set()
    unique = []
    
    for source in sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(source)
        else:
            LOGGER.debug(f"Removed duplicate URL: {url}")
    
    removed = len(sources) - len(unique)
    if removed > 0:
        LOGGER.info(f"Removed {removed} duplicate URLs")
    
    return unique


def remove_duplicates_by_title(sources: list[dict], threshold: int = 90) -> list[dict]:
    """Remove duplicate sources by title similarity.
    
    Args:
        sources: List of source dictionaries with 'title' field.
        threshold: Similarity threshold (0-100).
        
    Returns:
        Deduplicated list of sources.
    """
    if len(sources) <= 1:
        return sources
    
    titles = [s.get("title", "") for s in sources]
    unique = []
    seen_titles = set()
    
    for i, source in enumerate(sources):
        title = source.get("title", "")
        
        # Check if title is too similar to any seen title
        is_duplicate = False
        for seen_title in seen_titles:
            similarity = fuzz.ratio(title, seen_title)
            if similarity >= threshold:
                LOGGER.debug(
                    f"Duplicate title found: '{title}' vs '{seen_title}' (similarity: {similarity})"
                )
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(source)
            seen_titles.add(title)
    
    removed = len(sources) - len(unique)
    if removed > 0:
        LOGGER.info(f"Removed {removed} duplicate titles (threshold: {threshold})")
    
    return unique


def remove_duplicates_by_content(sources: list[dict], threshold: int = 90) -> list[dict]:
    """Remove duplicate sources by content similarity.
    
    Args:
        sources: List of source dictionaries with 'content' field.
        threshold: Similarity threshold (0-100).
        
    Returns:
        Deduplicated list of sources.
    """
    if len(sources) <= 1:
        return sources
    
    contents = [s.get("content", "") for s in sources]
    unique = []
    seen_contents = []
    
    for i, source in enumerate(sources):
        content = source.get("content", "")
        
        # Skip if content is too short
        if len(content) < 50:
            unique.append(source)
            continue
        
        # Check if content is too similar to any seen content
        is_duplicate = False
        for seen_content in seen_contents:
            similarity = fuzz.ratio(content, seen_content)
            if similarity >= threshold:
                LOGGER.debug(
                    f"Duplicate content found (similarity: {similarity})"
                )
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(source)
            seen_contents.append(content)
    
    removed = len(sources) - len(unique)
    if removed > 0:
        LOGGER.info(f"Removed {removed} duplicate contents (threshold: {threshold})")
    
    return unique


def deduplicate_sources(sources: list[dict], threshold: int = 90) -> list[dict]:
    """Remove duplicates using all methods.
    
    Args:
        sources: List of source dictionaries.
        threshold: Similarity threshold for content/title comparison.
        
    Returns:
        Deduplicated list of sources.
    """
    LOGGER.info(f"Deduplicating {len(sources)} sources")
    
    # Remove by URL first (most reliable)
    sources = remove_duplicates_by_url(sources)
    
    # Remove by title
    sources = remove_duplicates_by_title(sources, threshold)
    
    # Remove by content
    sources = remove_duplicates_by_content(sources, threshold)
    
    LOGGER.info(f"Deduplication complete: {len(sources)} unique sources")
    return sources


def calculate_duplicate_ratio(original_count: int, final_count: int) -> float:
    """Calculate the ratio of duplicates removed.
    
    Args:
        original_count: Original number of sources.
        final_count: Final number of unique sources.
        
    Returns:
        Duplicate ratio (0-1).
    """
    if original_count == 0:
        return 0.0
    return (original_count - final_count) / original_count
