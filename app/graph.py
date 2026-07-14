"""LangGraph definition for the GovResearch-AI workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.planner import plan_research
from agents.report_writer import write_report
from agents.researcher import conduct_research
from app.state import ResearchState


def build_research_graph() -> object:
    """Build the ordered Planner → Research → Report state graph.

    Returns:
        Compiled LangGraph application ready for invocation.
    """
    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", plan_research)
    workflow.add_node("researcher", conduct_research)
    workflow.add_node("report_writer", write_report)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "report_writer")
    workflow.add_edge("report_writer", END)
    return workflow.compile()
