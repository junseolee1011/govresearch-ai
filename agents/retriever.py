"""Retriever agent for RAG system."""

import logging
from typing import TYPE_CHECKING

from app.state import ResearchState
from tools.vector_store import VectorStore

if TYPE_CHECKING:
    from config.settings import Settings


LOGGER = logging.getLogger(__name__)


def retrieve_documents(state: ResearchState, settings: "Settings") -> dict[str, object]:
    """Retrieve relevant documents from vector store.
    
    Args:
        state: Workflow state containing topic and research plan.
        settings: Application settings with vector store configuration.
        
    Returns:
        State update containing retrieved documents.
    """
    topic = state["topic"]
    
    # Initialize vector store
    vector_store = VectorStore(settings)
    
    # Log vector store status
    doc_count = vector_store.get_document_count()
    LOGGER.info(f"Vector store contains {doc_count} documents")
    
    # Perform similarity search
    retrieved_docs = vector_store.similarity_search(
        query=topic,
        k=settings.retrieval_k,
        score_threshold=settings.similarity_threshold,
    )
    
    # Log retrieval results
    if retrieved_docs:
        LOGGER.info(
            f"Retrieved {len(retrieved_docs)} documents from vector store (cache hit)"
        )
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Log top 3
            LOGGER.debug(
                f"  Doc {i}: {doc.title[:50]}... (score: {doc.score:.3f})"
            )
    else:
        LOGGER.info("No relevant documents found in vector store (cache miss)")
    
    return {
        "retrieved_documents": retrieved_docs,
        "cache_hit": len(retrieved_docs) > 0,
    }


def should_search_web(state: ResearchState, settings: "Settings") -> str:
    """Decide whether to perform web search based on retrieval results.
    
    Args:
        state: Workflow state containing retrieved documents.
        settings: Application settings.
        
    Returns:
        "search" if web search is needed, "skip" if enough context retrieved.
    """
    retrieved_docs = state.get("retrieved_documents", [])
    min_docs = settings.min_retrieved_docs
    
    if len(retrieved_docs) >= min_docs:
        LOGGER.info(
            f"Retrieved {len(retrieved_docs)} documents >= {min_docs}, skipping web search"
        )
        return "skip"
    
    LOGGER.info(
        f"Retrieved {len(retrieved_docs)} documents < {min_docs}, performing web search"
    )
    return "search"
