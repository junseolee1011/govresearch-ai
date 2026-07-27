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
    """Extract main article text from a webpage using trafilatura.
    
    Args:
        url: URL to fetch and extract content from.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (extracted_text, error_message). 
        If successful, error_message is empty string.
    """
    try:
        # Try trafilatura first for better content extraction
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(url, timeout=timeout)
            if downloaded:
                # Use trafilatura with more aggressive extraction
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
                if text and len(text) > 100:
                    LOGGER.info(f"Trafilatura extraction successful for {url}: {len(text)} characters")
                    return text, ""
        except ImportError:
            LOGGER.debug("Trafilatura not available, falling back to BeautifulSoup")
        except Exception as e:
            LOGGER.debug(f"Trafilatura extraction failed for {url}: {e}, falling back to BeautifulSoup")
        
        # Fallback to BeautifulSoup
        return extract_article_text_beautifulsoup(url, timeout)
        
    except Exception as e:
        error_msg = f"Failed to extract from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg


def extract_article_text_beautifulsoup(url: str, timeout: int = 10) -> tuple[str, str]:
    """Extract main article text from a webpage using BeautifulSoup.
    
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
        
        # Remove unwanted elements - more aggressive filtering
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript", "form", "button"]):
            element.decompose()
        
        # Remove common navigation/menu classes - more patterns
        for element in soup.find_all(class_=re.compile(r"(nav|menu|sidebar|footer|header|breadcrumb|pagination|wrapper|container|toolbar|utility|skip|accessibility)", re.I)):
            element.decompose()
        
        # Remove elements by id patterns
        for element in soup.find_all(id=re.compile(r"(nav|menu|sidebar|footer|header|breadcrumb|pagination|wrapper|toolbar|utility|skip|accessibility)", re.I)):
            element.decompose()
        
        # Remove elements with specific text patterns (navigation text)
        for element in soup.find_all(text=re.compile(r"(본문 바로가기|전체메뉴|바로가기|팝업|알림|닫기|건너뛰기)", re.I)):
            if element.parent:
                element.parent.decompose()
        
        # Get text from main content areas
        # Try to find main content containers
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", id=re.compile(r"(content|article|post|body|main)", re.I)) or
            soup.find("div", class_=re.compile(r"(content|article|post|body|main)", re.I)) or
            soup.body
        )
        
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is too short, return error to trigger fallback
        if len(text) < 50:
            error_msg = f"Extracted content too short ({len(text)} chars)"
            LOGGER.warning(error_msg)
            return "", error_msg
        
        return text, ""
        
    except requests.RequestException as e:
        error_msg = f"Failed to fetch {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg
    except Exception as e:
        error_msg = f"Failed to parse {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg


def extract_pdf_text(url: str, timeout: int = 10) -> tuple[str, str]:
    """Extract text from PDF using PyMuPDF.
    
    Args:
        url: URL to fetch and extract content from.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (extracted_text, error_message). 
        If successful, error_message is empty string.
    """
    try:
        import fitz  # PyMuPDF
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Open PDF from bytes with error handling
        try:
            doc = fitz.open(stream=response.content, filetype="pdf")
        except Exception as pdf_error:
            # Try opening without filetype specification
            try:
                doc = fitz.open(stream=response.content)
            except Exception as e:
                error_msg = f"Failed to open PDF: {e}"
                LOGGER.warning(error_msg)
                return "", error_msg
        
        # Extract text from all pages with error suppression
        text_parts = []
        for page in doc:
            try:
                # Suppress MuPDF font errors by using get_text with flags
                page_text = page.get_text(flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
                text_parts.append(page_text)
            except Exception as page_error:
                # Try alternative extraction method if page extraction fails
                try:
                    page_text = page.get_text()
                    text_parts.append(page_text)
                except Exception:
                    # Skip this page if both methods fail
                    continue
        
        doc.close()
        
        text = " ".join(text_parts)
        text = " ".join(text.split())  # Clean up whitespace
        
        if len(text) < 50:
            error_msg = f"Extracted PDF content too short ({len(text)} chars)"
            LOGGER.warning(error_msg)
            return "", error_msg
        
        LOGGER.info(f"PDF extraction successful for {url}: {len(text)} characters")
        return text, ""
        
    except ImportError:
        error_msg = "PyMuPDF (fitz) not installed"
        LOGGER.warning(error_msg)
        return "", error_msg
    except requests.RequestException as e:
        error_msg = f"Failed to fetch PDF from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg
    except Exception as e:
        error_msg = f"Failed to parse PDF from {url}: {e}"
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
        min_length: Minimum character length for extracted body text (ignored for fallback).
        
    Returns:
        List of enriched search results with extracted content.
        Always returns the same number of documents as input.
        If extraction fails, uses original content as fallback.
    """
    enriched_results = []
    original_count = len(search_results)
    
    for result in search_results:
        url = result.get("url", "")
        original_content = result.get("content", "")
        
        # Branch based on URL type
        if is_download_url(url):
            # Try PDF extraction for download URLs
            extracted_text, error = extract_pdf_text(url)
            if error:
                LOGGER.warning(f"PDF extraction failed for {url}: {error}, using fallback content")
                enriched_result = result.copy()
                enriched_result["content"] = original_content
                enriched_result["original_snippet"] = original_content
                enriched_results.append(enriched_result)
                continue
        else:
            # Try HTML extraction for regular URLs
            extracted_text, error = extract_article_text(url)
            if error:
                LOGGER.warning(f"HTML extraction failed for {url}: {error}, using fallback content")
                enriched_result = result.copy()
                enriched_result["content"] = original_content
                enriched_result["original_snippet"] = original_content
                enriched_results.append(enriched_result)
                continue
        
        # If extraction succeeded but content is too short, use extracted anyway
        # (don't filter, just use what we got)
        if len(extracted_text) < min_length:
            LOGGER.info(f"Extracted content for {url} is short ({len(extracted_text)} chars), using it anyway")
        
        # Enrich the result with extracted content
        enriched_result = result.copy()
        enriched_result["content"] = extracted_text
        enriched_result["original_snippet"] = original_content
        enriched_results.append(enriched_result)
    
    LOGGER.info(f"Content enrichment: processed {len(enriched_results)} documents (input: {original_count}, output: {len(enriched_results)})")
    
    return enriched_results
