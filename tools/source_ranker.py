"""Source ranking for research quality assessment."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


LOGGER = logging.getLogger(__name__)


# Source quality rankings
SOURCE_RANKINGS = {
    # 5 stars - Highest quality
    "gov.kr": 5,
    "go.kr": 5,
    "or.kr": 5,
    "ac.kr": 5,
    "edu": 5,
    "spr.or.kr": 5,
    "nia.or.kr": 5,
    "kisa.or.kr": 5,
    "msit.go.kr": 5,
    "mois.go.kr": 5,
    "oecd.org": 5,
    "gov.uk": 5,
    "govtech.sg": 5,
    "usa.gov": 5,
    "gouv.fr": 5,
    "bundesregierung.de": 5,
    "europa.eu": 5,
    "nasa.gov": 5,
    "nih.gov": 5,
    "nature.com": 5,
    "science.org": 5,
    "ieee.org": 5,
    "acm.org": 5,
    "springer.com": 5,
    "elsevier.com": 5,
    
    # 4 stars - High quality
    "university": 4,
    "consulting": 4,
    "mckinsey.com": 4,
    "bcg.com": 4,
    "deloitte.com": 4,
    "pwc.com": 4,
    "accenture.com": 4,
    "worldbank.org": 4,
    "imf.org": 4,
    "weforum.org": 4,
    "un.org": 4,
    
    # 3 stars - Medium quality
    "news": 3,
    "reuters.com": 3,
    "bloomberg.com": 3,
    "ft.com": 3,
    "economist.com": 3,
    "wsj.com": 3,
    "forbes.com": 3,
    "techcrunch.com": 3,
    "venturebeat.com": 3,
    "zdnet.com": 3,
    "gartner.com": 3,
    "idc.com": 3,
    
    # 2 stars - Lower quality
    "medium.com": 2,
    "dev.to": 2,
    "hashnode.com": 2,
    "towardsdatascience.com": 2,
    
    # 1 star - Lowest quality
    "wikipedia.org": 1,
    "blogspot.com": 1,
    "wordpress.com": 1,
    "reddit.com": 1,
    "quora.com": 1,
    "youtube.com": 1,
}


def rank_source(url: str, title: str = "") -> int:
    """Rank a source based on domain and content.
    
    Args:
        url: Source URL.
        title: Source title (optional).
        
    Returns:
        Quality score from 1-5.
    """
    url_lower = url.lower()
    
    # Check domain-based ranking
    for domain, score in SOURCE_RANKINGS.items():
        if domain in url_lower:
            return score
    
    # Check for government patterns
    if ".gov." in url_lower or "gov." in url_lower:
        return 5
    
    # Check for academic patterns
    if ".edu" in url_lower or ".ac." in url_lower:
        return 5
    
    # Check for research institutions
    research_keywords = ["research", "institute", "lab", "laboratory"]
    if any(keyword in url_lower for keyword in research_keywords):
        return 4
    
    # Check for news
    news_keywords = ["news", "press", "media"]
    if any(keyword in url_lower for keyword in news_keywords):
        return 3
    
    # Default to medium quality
    return 3


def filter_by_quality(sources: list[dict], min_quality: int = 3) -> list[dict]:
    """Filter sources by minimum quality score.
    
    Args:
        sources: List of source dictionaries with 'url' field.
        min_quality: Minimum quality score to keep.
        
    Returns:
        Filtered list of sources.
    """
    filtered = []
    for source in sources:
        url = source.get("url", "")
        quality = rank_source(url)
        if quality >= min_quality:
            source["source_quality"] = quality
            filtered.append(source)
        else:
            LOGGER.debug(f"Filtered low-quality source: {url} (score: {quality})")
    
    LOGGER.info(f"Filtered {len(sources)} sources to {len(filtered)} with quality >= {min_quality}")
    return filtered


def calculate_average_quality(sources: list[dict]) -> float:
    """Calculate average source quality score.
    
    Args:
        sources: List of source dictionaries with 'source_quality' field.
        
    Returns:
        Average quality score.
    """
    if not sources:
        return 0.0
    
    qualities = [s.get("source_quality", 3) for s in sources]
    return sum(qualities) / len(qualities)


def categorize_sources(sources: list[dict]) -> dict[str, int]:
    """Categorize sources by type.
    
    Args:
        sources: List of source dictionaries with 'url' field.
        
    Returns:
        Dictionary with source type counts.
    """
    categories = {
        "government": 0,
        "public_institution": 0,
        "research_institute": 0,
        "paper": 0,
        "news": 0,
        "blog": 0,
        "unknown": 0,
    }
    
    for source in sources:
        url = source.get("url", "").lower()
        quality = source.get("source_quality", 3)
        
        if quality == 5:
            if ".gov" in url or "gov.kr" in url or "go.kr" in url:
                categories["government"] += 1
            elif ".ac.kr" in url or ".edu" in url:
                categories["research_institute"] += 1
            elif "nature.com" in url or "science.org" in url or "ieee.org" in url:
                categories["paper"] += 1
            else:
                categories["public_institution"] += 1
        elif quality == 3:
            categories["news"] += 1
        elif quality <= 2:
            categories["blog"] += 1
        else:
            categories["unknown"] += 1
    
    return categories
