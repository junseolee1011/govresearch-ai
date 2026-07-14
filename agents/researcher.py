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
    sources = [
        {
            "title": "Dummy Citizen Service Assistant",
            "url": "https://example.org/govresearch-ai/citizen-assistant",
            "institution": "Example City Hall",
            "service_type": "Citizen-facing conversational service",
            "use_case": "Answers common municipal-service questions",
            "maturity": "Pilot",
        },
        {
            "title": "Dummy Document Triage Service",
            "url": "https://example.org/govresearch-ai/document-triage",
            "institution": "Example National Agency",
            "service_type": "Internal workflow automation",
            "use_case": "Classifies incoming applications for staff review",
            "maturity": "Operational",
        },
        {
            "title": "Dummy Policy Insight Dashboard",
            "url": "https://example.org/govresearch-ai/policy-insight",
            "institution": "Example Provincial Government",
            "service_type": "Decision-support analytics",
            "use_case": "Summarizes service-demand trends for policy teams",
            "maturity": "Prototype",
        },
    ]
    findings = [
        "Public-sector AI services can be grouped into citizen-facing, "
        "internal workflow, and decision-support services.",
        "The classification must distinguish prototypes and pilots from "
        "operational services.",
        "Each service record needs a named institution, use case, and "
        "verifiable primary source.",
        "Sprint 1 entries are dummy data; future sprints must validate "
        "them with authoritative public-institution sources.",
    ]
    LOGGER.info(
        "Researcher produced %d sources for %d plan items.",
        len(sources),
        len(state["plan"]),
    )
    return {"sources": sources, "findings": findings}
