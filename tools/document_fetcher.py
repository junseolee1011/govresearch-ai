"""Document content fetcher with multi-format support."""

import logging
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def fetch_html_content(url: str, timeout: int = 15) -> tuple[str, str]:
    """Fetch and extract main content from HTML page.
    
    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (extracted_text, error_message).
    """
    print(f"\n=== Fetch Stage ===")
    print(f"URL: {url}")
    print(f"Extract Method: html")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        # Handle encoding
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding
        
        html_length = len(response.text)
        print(f"HTML Length (before extraction): {html_length}")
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
            element.decompose()
        
        # Remove common navigation/menu classes
        import re
        for element in soup.find_all(class_=re.compile(r"(nav|menu|sidebar|footer|header|breadcrumb|pagination)", re.I)):
            element.decompose()
        
        # Try to find main content areas
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", id=re.compile(r"(content|article|post|body)", re.I)) or
            soup.find("div", class_=re.compile(r"(content|article|post|body)", re.I)) or
            soup.body
        )
        
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        print(f"Extracted Text Length: {len(text)}")
        print(f"Success: True")
        print("====================\n")
        
        return text, ""
        
    except requests.RequestException as e:
        print(f"HTTP Status: Error")
        print(f"Success: False")
        print(f"Exception: {type(e).__name__}: {e}")
        print("====================\n")
        error_msg = f"Failed to fetch HTML from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg
    except Exception as e:
        print(f"Success: False")
        print(f"Exception: {type(e).__name__}: {e}")
        print("====================\n")
        error_msg = f"Failed to parse HTML from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg


def fetch_pdf_content(url: str, timeout: int = 15) -> tuple[str, str]:
    """Fetch and extract text from PDF.
    
    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (extracted_text, error_message).
    """
    print(f"\n=== Fetch Stage ===")
    print(f"URL: {url}")
    print(f"Extract Method: pdf")
    
    try:
        import fitz  # PyMuPDF
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        # Open PDF from bytes
        doc = fitz.open(stream=response.content, filetype="pdf")
        
        page_count = len(doc)
        print(f"PDF Page Count: {page_count}")
        
        # Extract text from all pages
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        
        text = " ".join(text_parts)
        text = " ".join(text.split())  # Clean up whitespace
        
        print(f"Extracted Text Length: {len(text)}")
        print(f"Success: True")
        print("====================\n")
        
        return text, ""
        
    except ImportError as e:
        print(f"Success: False")
        print(f"Exception: {type(e).__name__}: {e}")
        print("====================\n")
        error_msg = "PyMuPDF (fitz) not installed, skipping PDF extraction"
        LOGGER.warning(error_msg)
        return "", error_msg
    except requests.RequestException as e:
        print(f"HTTP Status: Error")
        print(f"Success: False")
        print(f"Exception: {type(e).__name__}: {e}")
        print("====================\n")
        error_msg = f"Failed to fetch PDF from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg
    except Exception as e:
        print(f"Success: False")
        print(f"Exception: {type(e).__name__}: {e}")
        print("====================\n")
        error_msg = f"Failed to parse PDF from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg


def fetch_hwp_content(url: str, timeout: int = 15) -> tuple[str, str]:
    """Fetch and extract text from HWP/HWPX file.
    
    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (extracted_text, error_message).
    """
    print(f"\n=== Fetch Stage ===")
    print(f"URL: {url}")
    print(f"Extract Method: hwp")
    
    try:
        # HWP extraction requires additional libraries and is complex
        # For now, we'll skip HWP extraction as it requires specialized tools
        print(f"Success: False")
        print(f"Exception: HWP extraction not yet implemented, requires specialized libraries")
        print("====================\n")
        error_msg = "HWP extraction not yet implemented, requires specialized libraries"
        LOGGER.warning(error_msg)
        return "", error_msg
        
    except Exception as e:
        print(f"Success: False")
        print(f"Exception: {type(e).__name__}: {e}")
        print("====================\n")
        error_msg = f"Failed to parse HWP from {url}: {e}"
        LOGGER.warning(error_msg)
        return "", error_msg


def fetch_document(url: str, fallback_content: str = "", timeout: int = 15) -> str:
    """Fetch document content with multi-format fallback.
    
    Priority:
    1. HTML (trafilatura/BeautifulSoup)
    2. PDF (PyMuPDF)
    3. HWP/HWPX (if available)
    4. Fallback content (Tavily raw_content or snippet)
    
    Args:
        url: URL to fetch.
        fallback_content: Fallback content if all methods fail.
        timeout: Request timeout in seconds.
        
    Returns:
        Extracted text content.
    """
    url_lower = url.lower()
    
    # Try HTML first
    if not any(ext in url_lower for ext in [".pdf", ".hwp", ".hwpx"]):
        LOGGER.info(f"Attempting HTML extraction for: {url}")
        text, error = fetch_html_content(url, timeout)
        if text and len(text) > 100:
            LOGGER.info(f"HTML extraction successful: {len(text)} characters")
            return text
        if error:
            LOGGER.debug(f"HTML extraction failed: {error}")
    
    # Try PDF
    if ".pdf" in url_lower:
        LOGGER.info(f"Attempting PDF extraction for: {url}")
        text, error = fetch_pdf_content(url, timeout)
        if text and len(text) > 100:
            LOGGER.info(f"PDF extraction successful: {len(text)} characters")
            return text
        if error:
            LOGGER.debug(f"PDF extraction failed: {error}")
    
    # Try HWP
    if ".hwp" in url_lower or ".hwpx" in url_lower:
        LOGGER.info(f"Attempting HWP extraction for: {url}")
        text, error = fetch_hwp_content(url, timeout)
        if text and len(text) > 100:
            LOGGER.info(f"HWP extraction successful: {len(text)} characters")
            return text
        if error:
            LOGGER.debug(f"HWP extraction failed: {error}")
    
    # Fallback to provided content
    print(f"\n=== Fetch Stage ===")
    print(f"URL: {url}")
    print(f"Extract Method: fallback")
    print(f"Success: True (using fallback)")
    print(f"Fallback Content Length: {len(fallback_content)}")
    print("====================\n")
    LOGGER.warning(f"All extraction methods failed for {url}, using fallback content")
    return fallback_content
