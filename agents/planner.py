"""Planner agent for research decomposition."""

from __future__ import annotations

import logging

from app.state import ResearchState
from tools.prompt_loader import load_prompt


LOGGER = logging.getLogger(__name__)


def plan_research(state: ResearchState) -> dict[str, list[str]]:
    """Create a focused research plan for a submitted topic.

    Args:
        state: Current workflow state containing a research topic.

    Returns:
        State update containing ordered research questions.
    """
    _ = load_prompt("planner.txt")
    topic = state["topic"]
    plan = [
        f"Which public institutions use AI services related to {topic}?",
        "How can the identified services be classified by service type?",
        "Which public-service workflows and user groups does each service support?",
        "What is each service's deployment maturity and evidence source?",
    ]
    LOGGER.info("Planner created %d research questions.", len(plan))
    return {"plan": plan}
