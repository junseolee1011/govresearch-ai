"""LangGraph definition for the GovResearch-AI workflow."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.fact_extractor import make_extract_facts
from agents.planner import plan_research
from agents.reflection import make_reflect, should_research_again
from agents.report_writer import make_write_report
from agents.researcher import conduct_research
from agents.retriever import retrieve_documents, should_search_web
from agents.summarizer import make_summarize_findings
from agents.validator import make_validate_facts
from app.state import ResearchState
from config.settings import Settings
from tools.content_enricher import enrich_search_results
from tools.deduplicator import deduplicate_sources
from tools.search_normalizer import normalize_search_results
from tools.source_ranker import filter_by_quality


LOGGER = logging.getLogger(__name__)


def build_research_graph(settings: Settings) -> object:
    """Build the research quality workflow graph.

    Args:
        settings: Application settings with Ollama and RAG configuration.

    Returns:
        Compiled LangGraph application ready for invocation.
    """

    def planner_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for planner to inject settings."""
        return plan_research(state)

    def retriever_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for retriever to inject settings."""
        return retrieve_documents(state, settings)

    def decision_node(state: ResearchState) -> str:
        """Wrapper for decision to inject settings."""
        return should_search_web(state, settings)

    def researcher_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for researcher to inject settings."""
        return conduct_research(state, settings)

    def store_node(state: ResearchState) -> dict[str, object]:
        """Store new research results in vector store."""
        from tools.vector_store import VectorStore
        from models.document import Document
        
        vector_store = VectorStore(settings)
        
        # Convert search results to documents
        search_results = state.get("search_results", [])
        documents = []
        
        for result in search_results:
            # Use raw_content if available, otherwise fall back to content
            content = result.get("raw_content", "") or result.get("content", "")
            doc = Document(
                title=result["title"],
                url=result["url"],
                content=content,
                source_type=result.get("source_type", "unknown"),
            )
            documents.append(doc)
        
        # Add to vector store
        added_count = vector_store.add_documents(documents)
        vector_store.persist()
        
        LOGGER.info(f"Stored {added_count} new documents in vector store")
        
        return {"stored_documents_count": added_count}

    def normalizer_node(state: ResearchState) -> dict[str, object]:
        """Normalize search results."""
        search_results = state.get("search_results", [])
        normalized = normalize_search_results(search_results)
        return {"normalized_results": normalized, "search_results": normalized}

    def source_ranker_node(state: ResearchState) -> dict[str, object]:
        """Filter and rank sources by quality."""
        search_results = state.get("search_results", [])
        filtered = filter_by_quality(search_results, min_quality=3)
        LOGGER.info(f"Source ranking: {len(search_results)} -> {len(filtered)} high-quality sources")
        return {"search_results": filtered}

    def deduplicator_node(state: ResearchState) -> dict[str, object]:
        """Remove duplicate sources."""
        search_results = state.get("search_results", [])
        original_count = len(search_results)
        deduplicated = deduplicate_sources(search_results, threshold=90)
        LOGGER.info(f"Deduplication: {original_count} -> {len(deduplicated)} unique sources")
        return {
            "deduplicated_results": deduplicated,
            "deduplicated_count": len(deduplicated),
            "search_results": deduplicated,
        }

    def content_enricher_node(state: ResearchState) -> dict[str, object]:
        """Enrich search results with full webpage content."""
        search_results = state.get("search_results", [])
        original_count = len(search_results)
        enriched = enrich_search_results(search_results, min_length=300)
        LOGGER.info(f"Content enrichment: {original_count} -> {len(enriched)} documents with sufficient content")
        return {"search_results": enriched}

    def fact_extractor_node(state: ResearchState) -> dict[str, object]:
        """Extract structured facts from documents."""
        extract_facts = make_extract_facts(settings)
        return extract_facts(state)

    def validator_node(state: ResearchState) -> dict[str, object]:
        """Validate extracted facts."""
        validate_facts = make_validate_facts(settings)
        return validate_facts(state)

    def reflection_node(state: ResearchState) -> dict[str, object]:
        """Evaluate research quality."""
        reflect = make_reflect(settings)
        return reflect(state)

    def reflection_decision_node(state: ResearchState) -> str:
        """Decide whether to research again based on reflection."""
        return should_research_again(state)

    def summarizer_node(state: ResearchState) -> dict[str, str]:
        """Wrapper for summarizer to inject settings."""
        summarize_findings = make_summarize_findings(settings)
        return summarize_findings(state)

    def report_writer_node(state: ResearchState) -> dict[str, str]:
        """Wrapper for report writer to inject settings."""
        write_report = make_write_report(settings)
        return write_report(state)

    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("store", store_node)
    workflow.add_node("normalizer", normalizer_node)
    workflow.add_node("source_ranker", source_ranker_node)
    workflow.add_node("deduplicator", deduplicator_node)
    workflow.add_node("content_enricher", content_enricher_node)
    workflow.add_node("fact_extractor", fact_extractor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("report_writer", report_writer_node)
    
    # Define edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    
    # Conditional edge: decide whether to search web or skip to normalizer
    # Cache hits skip only web search, but still go through full processing pipeline
    workflow.add_conditional_edges(
        "retriever",
        decision_node,
        {
            "search": "researcher",
            "skip": "normalizer",
        },
    )
    
    workflow.add_edge("researcher", "store")
    workflow.add_edge("store", "normalizer")
    workflow.add_edge("normalizer", "source_ranker")
    workflow.add_edge("source_ranker", "deduplicator")
    workflow.add_edge("deduplicator", "content_enricher")
    workflow.add_edge("content_enricher", "fact_extractor")
    workflow.add_edge("fact_extractor", "validator")
    workflow.add_edge("validator", "reflection")
    
    # Conditional edge: decide whether to research again based on reflection
    workflow.add_conditional_edges(
        "reflection",
        reflection_decision_node,
        {
            "research_again": "researcher",
            "continue": "summarizer",
        },
    )
    
    workflow.add_edge("summarizer", "report_writer")
    workflow.add_edge("report_writer", END)
    
    return workflow.compile()
