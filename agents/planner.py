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
        f"What public-sector problem does {topic} address?",
        f"Which AI methodology and governance controls support {topic}?",
        f"What platform architecture and implementation practices are relevant?",
        f"Which government case studies demonstrate lessons for {topic}?",
    ]
    LOGGER.info("Planner created %d research questions.", len(plan))
    return {"plan": plan}
