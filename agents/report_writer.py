"""Report writer agent for structured research output."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.state import ResearchState
from tools.prompt_loader import load_prompt


LOGGER = logging.getLogger(__name__)


def write_report(state: ResearchState) -> dict[str, str]:
    """Generate a structured Markdown report from workflow evidence.

    Args:
        state: Workflow state with research plan, sources, and findings.

    Returns:
        State update containing the rendered Markdown report.
    """
    _ = load_prompt("report_writer.txt")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    plan_section = "\n".join(
        f"{index}. {item}" for index, item in enumerate(state["plan"], 1)
    )
    findings_section = "\n".join(f"- {item}" for item in state["findings"])
    sources_section = "\n".join(
        f"- [{source['title']}]({source['url']}): {source['note']}"
        for source in state["sources"]
    )
    report = f"""# GovResearch-AI Research Report

## Topic

{state["topic"]}

## Executive Summary

This Sprint 1 report establishes an initial, structured view of the topic. Its
evidence set is deterministic placeholder data and must be validated with live
sources before it supports policy or implementation decisions.

## Research Plan

{plan_section}

## Key Findings

{findings_section}

## Source Notes

{sources_section}

## Recommended Next Steps

- Validate each finding with authoritative government and primary sources.
- Define accountable owners, risk controls, and measurable public-value outcomes.
- Extend the workflow with live retrieval and reflection capabilities in future sprints.

---

Generated: {generated_at}
"""
    LOGGER.info("Report writer generated a report for topic: %s", state["topic"])
    return {"report": report}
