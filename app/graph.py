"""LangGraph definition for the GovResearch-AI workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.planner import plan_research
from agents.report_writer import write_report
from agents.researcher import conduct_research
from app.state import ResearchState
from config.settings import Settings


def build_research_graph(settings: Settings) -> object:
    """Build the ordered Planner → Research → Report state graph.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        Compiled LangGraph application ready for invocation.
    """

    def planner_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for planner to inject settings."""
        return plan_research(state, settings)

    def researcher_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for researcher to inject settings."""
        return conduct_research(state, settings)

    def report_writer_node(state: ResearchState) -> dict[str, str]:
        """Wrapper for report writer to inject settings."""
        return write_report(state, settings)

    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("report_writer", report_writer_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "report_writer")
    workflow.add_edge("report_writer", END)
    return workflow.compile()
