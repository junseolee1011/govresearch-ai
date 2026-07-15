"""High-level workflow execution and report persistence."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from app.graph import build_research_graph
from config.settings import Settings


def run_research(topic: str, settings: Settings) -> dict[str, object]:
    """Run the research graph and persist its report.

    Args:
        topic: Subject to research.
        settings: Runtime application settings.

    Returns:
        Completed workflow state.
    """
    graph = build_research_graph(settings)
    result = graph.invoke({"topic": topic})
    report = str(result["report"])
    settings.reports_directory.mkdir(parents=True, exist_ok=True)
    safe_topic = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{safe_topic or 'report'}.md"
    report_path = settings.reports_directory / filename
    report_path.write_text(report, encoding="utf-8")
    return dict(result)
