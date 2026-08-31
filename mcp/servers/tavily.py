"""Tavily MCP Server for web search."""

import logging
from typing import Dict, List

from config.settings import Settings
from tools.tavily_search import TavilySearchError, search_web


LOGGER = logging.getLogger(__name__)


class TavilyMCPServer:
    """MCP Server for Tavily web search."""

    def __init__(self, settings: Settings):
        """Initialize Tavily MCP Server.

        Args:
            settings: Application settings with Tavily configuration.
        """
        self.settings = settings
        LOGGER.info("Tavily MCP Server initialized")

    def get_tools(self) -> list[str]:
        """Get list of available tools.

        Returns:
            List of tool names.
        """
        return ["search"]

    def search(self, query: str, max_results: int = 5) -> dict:
        """Perform web search using Tavily.

        Args:
            query: Search query.
            max_results: Maximum number of results to return.

        Returns:
            Dictionary with search results.
        """
        LOGGER.info(f"Tavily search: query='{query}', max_results={max_results}")

        try:
            search_response = search_web(query, self.settings, max_results=max_results)

            return {
                "success": True,
                "results": search_response["results"],
                "query": query,
                "count": len(search_response["results"])
            }
        except TavilySearchError as e:
            LOGGER.error(f"Tavily search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
        except Exception as e:
            LOGGER.error(f"Unexpected error in Tavily search: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
