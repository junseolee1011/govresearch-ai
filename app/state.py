"""Shared state definitions for the research workflow."""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    """State passed between GovResearch-AI workflow agents.

    Attributes:
        topic: Research topic submitted by the user.
        plan: Ordered research questions created by the planner.
        sources: Dummy source records gathered by the researcher.
        findings: Normalized findings derived from sources.
        report: Final Markdown report created by the report writer.
    """

    topic: str
    plan: list[str]
    sources: list[dict[str, str]]
    findings: list[str]
    report: str
