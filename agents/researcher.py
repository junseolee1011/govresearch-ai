"""Research agent for PoC with MCP-based tool invocation and document chunking."""

from __future__ import annotations

import logging
import re

from app.state import ResearchState
from config.settings import Settings


LOGGER = logging.getLogger(__name__)


def conduct_research(state: ResearchState, settings: Settings, mcp_client=None) -> dict[str, object]:
    """Perform research using MCP Client for tool invocation with rule-based tool selection.

    Args:
        state: Workflow state containing topic and search queries.
        settings: Application settings.
        mcp_client: MCP Client instance for tool invocation.

    Returns:
        State update containing search results and local documents.
    """
    search_queries = state.get("search_queries", [])
    topic = state["topic"]
    
    if not search_queries:
        LOGGER.warning("No search queries provided, using topic as query")
        search_queries = [topic]
    
    # Initialize results containers
    web_results = []
    local_documents = []
    document_chunks = []
    
    # Use MCP Client if available
    if mcp_client:
        LOGGER.info("Using MCP Client for research")
        
        # Rule-based tool selection
        use_filesystem = should_use_filesystem(topic)
        use_tavily = should_use_tavily(topic)
        
        LOGGER.info(f"Tool selection: Filesystem={use_filesystem}, Tavily={use_tavily}")
        
        # Call Filesystem MCP if needed
        if use_filesystem:
            local_documents, document_chunks = fetch_local_documents(mcp_client, topic)
        
        # Call Tavily MCP if needed
        if use_tavily:
            web_results = fetch_web_results(mcp_client, search_queries)
    
    else:
        LOGGER.warning("MCP Client not provided, skipping MCP-based research")
    
    # Remove duplicate URLs from web results
    seen_urls = set()
    unique_web_results = []
    for result in web_results:
        url = result.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_web_results.append(result)

    duplicates_removed = len(web_results) - len(unique_web_results)
    LOGGER.info("Removed %d duplicate URLs, %d unique web results remain", duplicates_removed, len(unique_web_results))

    LOGGER.info(
        "Researcher completed: %d search queries, %d web results, %d local documents, %d chunks",
        len(search_queries),
        len(unique_web_results),
        len(local_documents),
        len(document_chunks),
    )

    return {
        "search_results": unique_web_results,
        "local_documents": local_documents,
        "document_chunks": document_chunks,
    }


def should_use_filesystem(topic: str) -> bool:
    """Determine if Filesystem MCP should be used based on topic.

    Args:
        topic: Research topic.

    Returns:
        True if Filesystem MCP should be used.
    """
    # Keywords that suggest local document search
    filesystem_keywords = [
        "보고서", "요약", "사례", "현황", "정책", "가이드",
        "report", "summary", "case", "status", "policy", "guide"
    ]
    
    topic_lower = topic.lower()
    for keyword in filesystem_keywords:
        if keyword in topic_lower:
            return True
    
    return False


def should_use_tavily(topic: str) -> bool:
    """Determine if Tavily MCP should be used based on topic.

    Args:
        topic: Research topic.

    Returns:
        True if Tavily MCP should be used.
    """
    # Keywords that suggest web search
    tavily_keywords = [
        "뉴스", "최신", "기사", "동향", "트렌드",
        "news", "latest", "article", "trend"
    ]
    
    topic_lower = topic.lower()
    for keyword in tavily_keywords:
        if keyword in topic_lower:
            return True
    
    # Default: use Tavily for general queries
    return True


def fetch_local_documents(mcp_client, topic: str):
    """Fetch local documents using Filesystem MCP and chunk them.

    Args:
        mcp_client: MCP Client instance.
        topic: Research topic (used to determine which files to read).

    Returns:
        Tuple of (local_documents, document_chunks).
    """
    local_documents = []
    document_chunks = []
    
    try:
        LOGGER.info("[MCP] Calling Filesystem MCP to list files in data/")
        list_result = mcp_client.call_tool("filesystem", "list_files", directory="data")
        
        if list_result.get("success"):
            files = list_result.get("files", [])
            LOGGER.info(f"[MCP] Found {len(files)} files in data/")
            
            # Read all files in data/ directory
            for file_info in files:
                file_name = file_info["name"]
                file_path = f"data/{file_name}"
                
                # Determine which tool to use based on file extension
                if file_name.endswith(".pdf"):
                    result = mcp_client.call_tool("filesystem", "read_pdf", file_path=file_path)
                elif file_name.endswith(".md"):
                    result = mcp_client.call_tool("filesystem", "read_markdown", file_path=file_path)
                else:
                    result = mcp_client.call_tool("filesystem", "read_text_file", file_path=file_path)
                
                if result.get("success"):
                    content = result["content"]
                    doc_type = result.get("type", "unknown")
                    
                    # Log PDF processing details
                    if doc_type == "pdf":
                        pages = result.get("pages", 0)
                        LOGGER.info(f"[PDF] Pages: {pages}, Extracted characters: {len(content)}")
                    
                    # Chunk the content for RAG
                    chunks = chunk_text(content, chunk_size=1500, overlap=200)
                    
                    if not chunks:
                        LOGGER.warning(f"[PDF] No chunks generated for {file_name} - content may be empty or too short")
                        # Fallback: store raw content as a single chunk
                        if content.strip():
                            chunks = [content]
                            LOGGER.info(f"[PDF] Fallback: using raw content as single chunk for {file_name}")
                    
                    LOGGER.info(f"[PDF] Generated chunks: {len(chunks)}")
                    
                    # Store metadata in local_documents
                    local_documents.append({
                        "path": result["path"],
                        "type": doc_type,
                        "source": "local"
                    })
                    
                    # Add chunks with metadata
                    for chunk_idx, chunk_content in enumerate(chunks):
                        document_chunks.append({
                            "content": chunk_content,
                            "source": result["path"],
                            "type": doc_type,
                            "chunk_id": chunk_idx + 1
                        })
                    
                    LOGGER.info(f"[MCP] Successfully read and chunked local document: {file_name}")
                else:
                    LOGGER.warning(f"[MCP] Failed to read {file_name}: {result.get('error')}")
        else:
            LOGGER.warning(f"[MCP] Failed to list files: {list_result.get('error')}")
            
    except Exception as e:
        LOGGER.warning(f"[MCP] Error calling Filesystem MCP: {e}")
    
    return local_documents, document_chunks


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If this is the last chunk, take all remaining text
        if end >= text_length:
            chunks.append(text[start:])
            break
        
        # Find a good break point (newline or space)
        chunk = text[start:end]
        
        # Try to break at newline
        last_newline = chunk.rfind('\n')
        if last_newline > chunk_size // 2:  # Only break if newline is in second half
            chunk = chunk[:last_newline]
            end = start + last_newline
        else:
            # Try to break at space
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap if end < text_length else text_length
    
    # Filter out empty chunks
    return [c for c in chunks if c]


def fetch_web_results(mcp_client, search_queries: list[str]) -> list[dict]:
    """Fetch web results using Tavily MCP.

    Args:
        mcp_client: MCP Client instance.
        search_queries: List of search queries.

    Returns:
        List of web search results.
    """
    web_results = []
    
    for query in search_queries:
        try:
            LOGGER.info(f"[MCP] Calling Tavily MCP for query: {query}")
            search_result = mcp_client.call_tool("tavily", "search", query=query, max_results=5)
            
            if search_result.get("success"):
                web_results.extend(search_result["results"])
                LOGGER.info(f"[MCP] Tavily MCP completed for query: {query}, results: {len(search_result['results'])}")
            else:
                LOGGER.error(f"[MCP] Tavily MCP failed for query '{query}': {search_result.get('error')}")
        except Exception as e:
            LOGGER.error(f"[MCP] Error calling Tavily MCP: {e}")
            continue
    
    return web_results
