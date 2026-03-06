"""MCP bridge: connects to multiple MCP servers and discovers their tools."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPBridge:
    """Connects to multiple MCP servers and discovers their tools."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[dict] = []  # Ollama-format tool definitions
        self.tool_map: dict[str, tuple[ClientSession, str]] = {}  # tool_name -> (session, server_name)
        self._exit_stacks: list[AsyncExitStack] = []

    async def connect_all(self):
        """Connect to all configured MCP servers and discover tools (in parallel)."""
        servers = self.config.get("mcpServers", {})

        async def _safe_connect(name: str, cfg: dict):
            try:
                await self._connect_server(name, cfg)
            except Exception as e:
                logger.warning("Could not connect to %s: %s", name, e)

        await asyncio.gather(*[
            _safe_connect(name, cfg) for name, cfg in servers.items()
        ])

    async def _connect_server(self, name: str, cfg: dict):
        """Connect to a single MCP server."""
        env = cfg.get("env")
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=env,
        )

        stack = AsyncExitStack()
        self._exit_stacks.append(stack)

        stdio_transport = await stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self.sessions[name] = session

        # Discover tools from this server
        result = await session.list_tools()
        for tool in result.tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            self.tools.append(ollama_tool)
            self.tool_map[tool.name] = (session, name)

        logger.info("Connected to %s: %d tools", name, len(result.tools))

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call via MCP."""
        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        session, server_name = self.tool_map[tool_name]
        result = await session.call_tool(tool_name, arguments)

        # Extract text content from result
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts)

    async def close(self):
        """Close all MCP server connections."""
        for stack in self._exit_stacks:
            await stack.aclose()
