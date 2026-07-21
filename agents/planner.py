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
        f"{topic}와 관련된 AI 서비스를 사용하는 공공 기관은 어디인가요?",
        "식별된 서비스들을 서비스 유형별로 어떻게 분류할 수 있나요?",
        "각 서비스는 어떤 공공 서비스 워크플로우와 사용자 그룹을 지원하나요?",
        "각 서비스의 배포 성숙도와 증거 출처는 무엇인가요?",
    ]
    LOGGER.info("Planner created %d research questions.", len(plan))
    return {"plan": plan}
