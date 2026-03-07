"""Tests for the MCP bridge connection manager."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from coral.mcp_bridge import MCPBridge


class TestMCPBridgeInit:
    def test_loads_config(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {"test": {"command": "echo", "args": ["hi"]}}}')
        bridge = MCPBridge(str(config))
        assert "test" in bridge.config["mcpServers"]

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
