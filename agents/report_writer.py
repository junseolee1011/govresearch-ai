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
    service_inventory = "\n".join(
        "| {title} | {institution} | {service_type} | {use_case} | {maturity} |".format(
            **source
        )
        for source in state["sources"]
    )
    sources_section = "\n".join(
        f"- [{source['title']}]({source['url']})" for source in state["sources"]
    )
    report = f"""# GovResearch-AI Research Report

## Topic

{state["topic"]}

## Executive Summary

This Sprint 1 report inventories and classifies AI services used by public
institutions. Its evidence set is deterministic placeholder data and must be
validated with primary public-institution sources before use in decisions.

## Research Plan

{plan_section}

## AI Service Inventory and Classification

| Service | Institution | Service type | Public-service use case | Maturity |
| --- | --- | --- | --- | --- |
{service_inventory}

## Classification Findings

{findings_section}

## Source Notes

{sources_section}

## Recommended Next Steps

- Validate each service with an authoritative public-institution source.
- Extend the taxonomy with sector, data sensitivity, and deployment model.
- Add live retrieval and reflection capabilities in future sprints.

---

Generated: {generated_at}
"""
    LOGGER.info("Report writer generated a report for topic: %s", state["topic"])
    return {"report": report}
