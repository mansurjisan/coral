"""Tests for the MCP bridge connection manager."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from coral.mcp_bridge import MCPBridge


class TestMCPBridgeInit:
    def test_loads_config(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {"coops": {"command": "echo", "args": ["hi"]}}}')
        bridge = MCPBridge(str(config))
        assert "coops" in bridge.config["mcpServers"]

    def test_empty_state_on_init(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))
        assert bridge.sessions == {}
        assert bridge.tools == []
        assert bridge.tool_map == {}

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            MCPBridge("/nonexistent/config.json")

    def test_unknown_server_in_config_raises(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {"filesystem": {"command": "echo", "args": ["hi"]}}}')

        with pytest.raises(ValueError, match="Unapproved MCP servers"):
            MCPBridge(str(config))


class TestCallTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        with pytest.raises(ValueError, match="Unknown tool"):
            await bridge.call_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_delegates_to_session(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        # Manually register a mock session + tool
        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "result data"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result

        bridge.tool_map["test_tool"] = (mock_session, "test_server")

        result = await bridge.call_tool("test_tool", {"arg": "value"})
        assert result == "result data"
        mock_session.call_tool.assert_called_once_with("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_joins_multi_content(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        mock_session = AsyncMock()
        c1 = MagicMock()
        c1.text = "line 1"
        c2 = MagicMock()
        c2.text = "line 2"
        mock_result = MagicMock()
        mock_result.content = [c1, c2]
        mock_session.call_tool.return_value = mock_result

        bridge.tool_map["multi_tool"] = (mock_session, "server")
        result = await bridge.call_tool("multi_tool", {})
        assert result == "line 1\nline 2"

    @pytest.mark.asyncio
    async def test_call_tool_handles_non_text_content(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        mock_session = AsyncMock()

        class FakeContent:
            """Content object without a .text attribute."""
            def __str__(self):
                return "binary content"

        mock_result = MagicMock()
        mock_result.content = [FakeContent()]
        mock_session.call_tool.return_value = mock_result

        bridge.tool_map["bin_tool"] = (mock_session, "server")
        result = await bridge.call_tool("bin_tool", {})
        assert "binary content" in result


class TestRegisterTools:
    def _make_tool(self, name: str, description: str = "", schema: dict | None = None):
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputSchema = schema or {"type": "object"}
        return tool

    def test_register_tools_populates_bridge(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        session = AsyncMock()
        tool = self._make_tool("test_tool", description="desc")

        bridge._register_tools(session, "server_a", [tool])

        assert bridge.tools == [{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "desc",
                "parameters": {"type": "object"},
            },
        }]
        assert bridge.tool_map["test_tool"] == (session, "server_a")
        assert bridge.tool_server_map["test_tool"] == "server_a"

    def test_register_tools_rejects_duplicate_across_servers(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        session_a = AsyncMock()
        session_b = AsyncMock()
        tool = self._make_tool("shared_tool")

        bridge._register_tools(session_a, "server_a", [tool])

        with pytest.raises(ValueError, match="Duplicate tool name 'shared_tool'"):
            bridge._register_tools(session_b, "server_b", [tool])

        assert bridge.tool_map["shared_tool"] == (session_a, "server_a")
        assert len(bridge.tools) == 1

    def test_register_tools_rejects_duplicate_within_server(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        session = AsyncMock()
        tool_a = self._make_tool("duplicate_tool")
        tool_b = self._make_tool("duplicate_tool")

        with pytest.raises(ValueError, match="Duplicate tool name 'duplicate_tool'"):
            bridge._register_tools(session, "server_a", [tool_a, tool_b])

        assert bridge.tools == []
        assert bridge.tool_map == {}
        assert bridge.tool_server_map == {}


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up_stacks(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        mock_stack = AsyncMock()
        bridge._exit_stacks = [mock_stack]
        await bridge.close()
        mock_stack.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_ignores_errors(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        mock_stack = AsyncMock()
        mock_stack.aclose.side_effect = RuntimeError("cleanup error")
        bridge._exit_stacks = [mock_stack]
        # Should not raise
        await bridge.close()
