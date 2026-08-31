"""MCP Client for tool invocation."""

import logging
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


class MCPClient:
    """Simple MCP Client for invoking tools on MCP Servers."""

    def __init__(self):
        """Initialize MCP Client with registered servers."""
        self.servers: Dict[str, Any] = {}

    def register_server(self, name: str, server: Any):
        """Register an MCP server.

        Args:
            name: Server name.
            server: Server instance.
        """
        self.servers[name] = server
        LOGGER.info(f"[MCP] Registered MCP server: {name}")

    def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:
        """Call a tool on a registered MCP server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            **kwargs: Tool arguments.

        Returns:
            Tool result.

        Raises:
            ValueError: If server or tool not found.
        """
        if server_name not in self.servers:
            raise ValueError(f"MCP server not found: {server_name}")

        server = self.servers[server_name]
        if not hasattr(server, tool_name):
            raise ValueError(f"Tool not found on server {server_name}: {tool_name}")

        tool = getattr(server, tool_name)
        LOGGER.info(f"[MCP] Calling tool: {server_name}.{tool_name} with args: {kwargs}")
        result = tool(**kwargs)
        LOGGER.info(f"[MCP] Tool completed: {server_name}.{tool_name}")
        return result

    def list_tools(self, server_name: str) -> list[str]:
        """List available tools on a server.

        Args:
            server_name: Name of the MCP server.

        Returns:
            List of tool names.
        """
        if server_name not in self.servers:
            raise ValueError(f"MCP server not found: {server_name}")

        server = self.servers[server_name]
        if hasattr(server, 'get_tools'):
            return server.get_tools()
        
        # Fallback: introspect callable methods
        tools = [attr for attr in dir(server) if not attr.startswith('_') and callable(getattr(server, attr))]
        return tools
