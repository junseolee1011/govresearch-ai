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
        """Generate a concise research summary using validated facts only.

        Args:
            state: Workflow state with validated facts.

        Returns:
            State update containing the generated summary.
        """
        with log_node_execution("summarizer") as metrics:
            try:
                # Extract structured facts - use validated_facts only
                facts = state.get("validated_facts", [])
                if not facts:
                    summary = "검증된 팩트가 없습니다."
                    LOGGER.warning("No validated facts to summarize")
                    return {"search_summary": summary}

                # Build facts summary
                facts_text = ""
                for fact in facts[:10]:  # Limit to top 10 facts
                    org = fact.get("organization", "알 수 없음")
                    service = fact.get("service_name", "알 수 없음")
                    purpose = fact.get("purpose", "") or ""
                    facts_text += f"- {org}: {service} - {purpose[:100] if purpose else '목적 없음'}\n"

                topic = state.get("topic", "연구 주제")

                # Generate simple summary from validated facts only
                summary = f"연구 주제: {topic}\n\n"
                summary += f"검증된 팩트 수: {len(facts)}\n\n"
                summary += "주요 AI 서비스 사례:\n"
                summary += facts_text

                metrics["response_length"] = len(summary)
                metrics["fact_count"] = len(facts)

                LOGGER.info(
                    "Summarizer generated summary (%d characters) from %d facts",
                    len(summary),
                    len(facts),
                )
                return {"search_summary": summary}

            except Exception as e:
                LOGGER.error("Summarizer failed: %s", str(e), exc_info=True)
                raise

    return summarize_findings
