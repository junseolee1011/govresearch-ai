"""Report writer agent using Ollama to generate Markdown reports."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.prompt_loader import render_prompt


LOGGER = logging.getLogger(__name__)


def make_write_report(settings: Settings):
    """Factory function to create a report writer node with settings injected.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        A node function that accepts ResearchState.
    """

    def write_report(state: ResearchState) -> dict[str, str]:
        """Generate a structured Markdown report using Ollama.

        Args:
            state: Workflow state with research plan, sources, and findings.

        Returns:
            State update containing the generated Markdown report.
        """
        with log_node_execution("report_writer") as metrics:
            try:
                llm = get_llm(settings)

                # Format context for the LLM
                plan_text = "\n".join(
                    f"{i}. {item}" for i, item in enumerate(state["plan"], 1)
                )
                findings_text = "\n".join(
                    f"- {finding}" for finding in state["findings"]
                )
                sources_text = "\n".join(
                    f"- {source['title']} ({source['institution']}): {source['use_case']}"
                    for source in state["sources"]
                )

                prompt = render_prompt(
                    "report_writer.txt",
                    query=state["topic"],
                    research_plan=plan_text,
                    documents=f"Findings:\n{findings_text}\n\nSources:\n{sources_text}",
                )

                LOGGER.debug("Report writer prompt: %s", prompt)

                report = llm.invoke(prompt)
                metrics["response_length"] = len(report)

                # Ensure report has proper Markdown structure
                if not report.strip().startswith("#"):
                    report = f"# GovResearch-AI Research Report\n\n{report}"

                # Add metadata footer
                generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
                report += f"\n\n---\n\nGenerated: {generated_at}"

                LOGGER.info(
                    "Report writer generated report (%d characters).", len(report)
                )
                return {"report": report}
            except Exception as e:
                LOGGER.error("Report writer failed: %s", str(e), exc_info=True)
                raise

    return write_report
