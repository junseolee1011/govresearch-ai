"""LangGraph definition for the GovResearch-AI PoC workflow with MCP."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.planner import plan_research
from agents.researcher import conduct_research
from agents.report_writer import make_write_report
from app.state import ResearchState
from config.settings import Settings
from mcp.client import MCPClient
from mcp.servers.filesystem import FilesystemMCPServer
from mcp.servers.tavily import TavilyMCPServer


LOGGER = logging.getLogger(__name__)


def build_research_graph(settings: Settings) -> object:
    """Build the simplified PoC research workflow graph with MCP integration.

    Args:
        settings: Application settings with Ollama and Tavily configuration.

    Returns:
        Compiled LangGraph application ready for invocation.
    """
    # Initialize MCP Client and register servers
    mcp_client = MCPClient()
    
    # Register Filesystem MCP Server
    filesystem_server = FilesystemMCPServer(base_path=".")
    mcp_client.register_server("filesystem", filesystem_server)
    
    # Register Tavily MCP Server
    tavily_server = TavilyMCPServer(settings)
    mcp_client.register_server("tavily", tavily_server)
    
    LOGGER.info("MCP Client initialized with Filesystem and Tavily servers")

    # Create wrapper functions that get settings from state
    def planner_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for planner to get settings from state."""
        settings_from_state = state.get("settings")
        return plan_research(state, settings_from_state)

    def researcher_node(state: ResearchState) -> dict[str, object]:
        """Wrapper for researcher to get settings from state."""
        settings_from_state = state.get("settings")
        return conduct_research(state, settings_from_state, mcp_client=mcp_client)

    report_writer_node = make_write_report(settings)

    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("report_writer", report_writer_node)
    
    # Define simple linear workflow
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "report_writer")
    workflow.add_edge("report_writer", END)
    
    return workflow.compile()
