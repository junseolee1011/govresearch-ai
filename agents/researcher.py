"""Research agent with deterministic Sprint 1 evidence."""

from __future__ import annotations

import logging

from app.state import ResearchState
from tools.prompt_loader import load_prompt



LOGGER = logging.getLogger(__name__)


def conduct_research(state: ResearchState) -> dict[str, object]:
    """Collect deterministic source notes for the current research plan.

    This dummy implementation defines the state contract for a future
    Tavily-backed research agent.

    Args:
        state: Workflow state containing topic and research plan.

    Returns:
        State update containing source records and synthesized findings.
    """
    _ = load_prompt("researcher.txt")
    topic = state["topic"]
    sources = [
        {
            "title": "OECD AI Principles",
            "url": "https://oecd.ai/en/ai-principles",
            "note": "A reference framework for trustworthy, accountable AI.",
        },
        {
            "title": "NIST AI Risk Management Framework",
            "url": "https://www.nist.gov/itl/ai-risk-management-framework",
            "note": "A lifecycle approach for governing AI risks.",
        },
        {
            "title": "Dummy Sprint 1 Government Case Study",
            "url": "https://example.org/govresearch-ai-sprint-1",
            "note": "Placeholder evidence for the Sprint 1 research contract.",
        },
    ]
    findings = [
        f"{topic} should begin with a clearly defined public value and owner.",
        "Risk management and human accountability must be built into delivery.",
        "A reusable platform needs governance, integration, monitoring, "
        "and adoption support.",
        "Case-study evidence should be validated with primary government "
        "sources in later sprints.",
    ]
    LOGGER.info(
        "Researcher produced %d sources for %d plan items.",
        len(sources),
        len(state["plan"]),
    )
    return {"sources": sources, "findings": findings}
