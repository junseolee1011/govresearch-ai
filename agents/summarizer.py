"""Summarizer agent for research findings using Ollama."""

from __future__ import annotations

import logging

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.prompt_loader import render_prompt


LOGGER = logging.getLogger(__name__)


def make_summarize_findings(settings: Settings):
    """Factory function to create a summarizer node with settings injected.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        A node function that accepts ResearchState.
    """

    def summarize_findings(state: ResearchState) -> dict[str, str]:
        """Generate a concise research summary using Ollama with RAG.

        Args:
            state: Workflow state with search results and retrieved documents.

        Returns:
            State update containing the generated summary.
        """
        with log_node_execution("summarizer") as metrics:
            try:
                llm = get_llm(settings)

                # Extract retrieved documents
                retrieved_docs = state.get("retrieved_documents", [])
                
                # Build retrieved context
                retrieved_context = ""
                if retrieved_docs:
                    context_snippets = []
                    for doc in retrieved_docs[:5]:  # Limit to top 5 retrieved docs
                        context_snippets.append(f"- {doc['title']}: {doc['content'][:300]}")
                    retrieved_context = "\n".join(context_snippets)
                    LOGGER.info(f"Using {len(retrieved_docs)} retrieved documents in summary")
                else:
                    retrieved_context = "검색된 지식이 없습니다."
                    LOGGER.info("No retrieved documents available")

                # Extract content from search results
                search_results = state.get("search_results", [])
                if not search_results:
                    summary = "검색 결과가 없습니다."
                    LOGGER.warning("No search results to summarize")
                    return {"search_summary": summary}

                # Build context from search results
                content_snippets = []
                for result in search_results[:10]:  # Limit to top 10 results
                    title = result.get("title", "")
                    content = result.get("content", "")
                    source_type = result.get("source_type", "")
                    snippet = f"[{source_type}] {title}: {content[:200]}"
                    content_snippets.append(snippet)

                search_results_text = "\n".join(content_snippets)
                topic = state.get("topic", "연구 주제")

                # Use prompt template for summarization
                prompt = render_prompt(
                    "summarizer.txt",
                    topic=topic,
                    retrieved_context=retrieved_context,
                    search_results=search_results_text,
                )

                LOGGER.debug("Summarizer prompt length: %d characters", len(prompt))

                summary = llm.invoke(prompt)
                metrics["response_length"] = len(summary)

                LOGGER.info(
                    "Summarizer generated summary (%d characters) from %d search results and %d retrieved documents",
                    len(summary),
                    len(search_results),
                    len(retrieved_docs),
                )
                return {"search_summary": summary}

            except Exception as e:
                LOGGER.error("Summarizer failed: %s", str(e), exc_info=True)
                raise

    return summarize_findings
