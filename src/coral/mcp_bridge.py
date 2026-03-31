"""MCP bridge: connects to multiple MCP servers and discovers their tools."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from coral.audit import record_audit_event, split_tool_audit_payload, summarize_arguments
from coral.policy import validate_configured_servers

logger = logging.getLogger(__name__)


def _load_plugins() -> dict:
    """Scan plugin directories for additional MCP server configs."""
    import os
    from pathlib import Path

    plugins: dict = {}
    plugin_dirs = [
        Path.home() / ".coral" / "servers",
        Path(os.environ.get("CORAL_PLUGINS_DIR", "")) if os.environ.get("CORAL_PLUGINS_DIR") else None,
    ]
    for plugin_dir in plugin_dirs:
        if plugin_dir is None or not plugin_dir.is_dir():
            continue
        for config_file in plugin_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    plugin_config = json.load(f)
                # Each plugin JSON has: {"command": "...", "args": [...]}
                server_name = config_file.stem
                plugins[server_name] = plugin_config
                logger.info("Loaded plugin: %s from %s", server_name, config_file)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load plugin %s: %s", config_file, e)
    return plugins


class MCPBridge:
    """Connects to multiple MCP servers and discovers their tools."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)

        # Load plugins from ~/.coral/servers/*.json
        self._plugin_names: set[str] = set()
        plugins = _load_plugins()
        if plugins:
            servers = self.config.setdefault("mcpServers", {})
            for name, cfg in plugins.items():
                if name not in servers:
                    servers[name] = cfg
                    self._plugin_names.add(name)

        # Validate only non-plugin servers against the policy manifest
        core_servers = [s for s in self.config.get("mcpServers", {}).keys() if s not in self._plugin_names]
        validate_configured_servers(core_servers)
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[dict] = []  # Ollama-format tool definitions
        self.tool_map: dict[str, tuple[ClientSession, str]] = {}  # tool_name -> (session, server_name)
        self.tool_server_map: dict[str, str] = {}  # tool_name -> server_name (for agent filtering)
        self._exit_stacks: list[AsyncExitStack] = []
        self._cache: dict[str, tuple[float, str]] = {}  # cache_key -> (timestamp, result)
        self._cache_ttl = 300  # 5 minutes default TTL

    async def connect_all(self):
        """Connect to all configured MCP servers and discover tools."""
        for name, cfg in self.config.get("mcpServers", {}).items():
            try:
                await self._connect_server(name, cfg)
            except Exception as e:
                logger.warning("Could not connect to %s: %s", name, e)
                print(f"Could not connect to {name}: {e}")

    def _register_tools(self, session: ClientSession, server_name: str, tools) -> None:
        """Validate and register tools discovered from a server."""
        pending_names: set[str] = set()
        pending_tools: list[dict] = []

        for tool in tools:
            if tool.name in self.tool_map:
                existing_server = self.tool_server_map[tool.name]
                raise ValueError(
                    f"Duplicate tool name '{tool.name}' from server '{server_name}' "
                    f"already provided by server '{existing_server}'"
                )
            if tool.name in pending_names:
                raise ValueError(
                    f"Duplicate tool name '{tool.name}' discovered multiple times from server '{server_name}'"
                )

            pending_names.add(tool.name)
            pending_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )

        self.tools.extend(pending_tools)
        for tool in tools:
            self.tool_map[tool.name] = (session, server_name)
            self.tool_server_map[tool.name] = server_name

    async def _connect_server(self, name: str, cfg: dict):
        """Connect to a single MCP server."""
        env = cfg.get("env")
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=env,
        )

        stack = AsyncExitStack()

        try:
            stdio_transport = await stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = stdio_transport
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except Exception:
            # Clean up the stack immediately on failure to avoid
            # cancel scope leaks from partially-entered contexts
            try:
                await stack.aclose()
            except Exception:
                pass
            raise

        try:
            result = await session.list_tools()
            self._register_tools(session, name, result.tools)
        except Exception:
            try:
                await stack.aclose()
            except Exception:
                pass
            raise

        # Only track the stack after successful connection and tool registration
        self._exit_stacks.append(stack)
        self.sessions[name] = session

        logger.info("Connected to %s: %d tools", name, len(result.tools))

    _TOOL_TIMEOUT = int(__import__("os").environ.get("CORAL_TOOL_TIMEOUT", "120"))

    # Tools whose results can be cached (read-only, stable data)
    _CACHEABLE_PREFIXES = ("hpc_", "nos_list", "nos_get_config", "nos_get_domain", "nos_get_ensemble")

    def _cache_key(self, tool_name: str, arguments: dict) -> str | None:
        """Return a cache key if this tool is cacheable, else None."""
        if any(tool_name.startswith(p) for p in self._CACHEABLE_PREFIXES):
            args_str = json.dumps(arguments, sort_keys=True)
            return f"{tool_name}:{args_str}"
        return None

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call via MCP with timeout, retry, and caching."""
        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check cache for read-only tools
        cache_key = self._cache_key(tool_name, arguments)
        if cache_key and cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.perf_counter() - cached_time) < self._cache_ttl:
                logger.debug("Cache hit: %s", tool_name)
                return cached_result

        session, server_name = self.tool_map[tool_name]
        started = time.perf_counter()
        args_summary = summarize_arguments(arguments)
        success = False
        sandbox_used = None
        error = None

        try:
            # Wrap in timeout to catch hung tools
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=self._TOOL_TIMEOUT,
            )

            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))

            text = "\n".join(parts)
            payload, cleaned = split_tool_audit_payload(text)
            sandbox_used = payload.get("sandbox_used")
            success = True

            # Store in cache if cacheable
            if cache_key:
                self._cache[cache_key] = (time.perf_counter(), cleaned)

            return cleaned
        except asyncio.TimeoutError:
            error = f"Tool '{tool_name}' timed out after {self._TOOL_TIMEOUT}s"
            raise TimeoutError(error)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            record_audit_event(
                "tool_call",
                server=server_name,
                tool=tool_name,
                args_summary=args_summary,
                duration_ms=duration_ms,
                success=success,
                error=error,
                sandbox_used=sandbox_used,
            )

    async def close(self):
        """Close all MCP server connections."""
        for stack in self._exit_stacks:
            try:
                await stack.aclose()
            except (Exception, asyncio.CancelledError):
                pass
