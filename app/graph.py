"""LangGraph definition for the GovResearch-AI workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.planner import plan_research
from agents.report_writer import make_write_report
from agents.researcher import conduct_research
from agents.retriever import retrieve_documents, should_search_web
from agents.summarizer import make_summarize_findings
from app.state import ResearchState
from config.settings import Settings


def build_research_graph(settings: Settings) -> object:
    """Build the RAG-enabled research workflow graph.

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

    def should_search_web_node(state: ResearchState) -> str:
        """Wrapper for should_search_web to inject settings."""
        return should_search_web(state, settings)

    def researcher_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for researcher to inject settings."""
        return conduct_research(state, settings)

    def store_node(state: ResearchState) -> dict[str, object]:
        """Store new research results in vector store."""
        from tools.vector_store import VectorStore
        from models.document import Document
        import logging
        
        LOGGER = logging.getLogger(__name__)
        
        vector_store = VectorStore(settings)
        
        # Convert search results to documents
        search_results = state.get("search_results", [])
        documents = []
        
        for result in search_results:
            doc = Document(
                title=result["title"],
                url=result["url"],
                content=result.get("content", ""),
                source_type=result.get("source_type", "unknown"),
            )
            documents.append(doc)
        
        # Add to vector store
        added_count = vector_store.add_documents(documents)
        vector_store.persist()
        
        LOGGER.info(f"Stored {added_count} new documents in vector store")
        
        return {"stored_documents_count": added_count}

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
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("report_writer", report_writer_node)
    
    # Define edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    
    # Conditional edge: decide whether to search web or skip to summarizer
    workflow.add_conditional_edges(
        "retriever",
        should_search_web_node,
        {
            "search": "researcher",
            "skip": "summarizer",
        },
    )
    
    workflow.add_edge("researcher", "store")
    workflow.add_edge("store", "retriever")
    workflow.add_edge("summarizer", "report_writer")
    workflow.add_edge("report_writer", END)
    
    return workflow.compile()
