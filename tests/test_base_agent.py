"""Tests for the BaseAgent class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.agents.base import BaseAgent, _truncate_result


class TestTruncateResult:
    def test_short_result_unchanged(self):
        assert _truncate_result("short") == "short"

    def test_long_result_truncated(self):
        long = "x" * 10000
        result = _truncate_result(long)
        assert len(result) < 10000
        assert "truncated" in result

    def test_many_lines_keeps_head_and_tail(self):
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = _truncate_result(text)
        assert "line 0" in result
        assert "line 199" in result
        assert "truncated" in result


class TestBaseAgent:
    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = [
            {"type": "function", "function": {"name": "test_tool", "description": "", "parameters": {}}}
        ]
        bridge.tool_server_map = {"test_tool": "test_server"}
        bridge.call_tool = AsyncMock(return_value="tool result")
        return bridge

    @pytest.mark.asyncio
    async def test_simple_text_response(self, mock_bridge):
        agent = BaseAgent("test", "model", "system prompt", mock_bridge, tool_filter=["test_server"])

        mock_response = MagicMock()
        mock_response.message.tool_calls = None
        mock_response.message.content = "Hello!"

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await agent.chat("Hi")

        assert result == "Hello!"
        assert len(agent.history) == 2

    @pytest.mark.asyncio
    async def test_tool_call_flow(self, mock_bridge):
        agent = BaseAgent("test", "model", "prompt", mock_bridge, tool_filter=["test_server"])

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "test_tool"
        tc.function.arguments = {"arg": "val"}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Done."

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            result = await agent.chat("Do something")

        assert result == "Done."
        mock_bridge.call_tool.assert_called_once_with("test_tool", {"arg": "val"})

    @pytest.mark.asyncio
    async def test_on_tool_call_callback(self, mock_bridge):
        calls = []
        agent = BaseAgent(
            "test", "model", "prompt", mock_bridge,
            tool_filter=["test_server"],
            on_tool_call=lambda name, args, result: calls.append((name, args, result)),
        )

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "test_tool"
        tc.function.arguments = {}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Ok."

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            await agent.chat("test")

        assert len(calls) == 1
        assert calls[0][0] == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_error_handled(self, mock_bridge):
        mock_bridge.call_tool = AsyncMock(side_effect=Exception("broken"))
        agent = BaseAgent("test", "model", "prompt", mock_bridge, tool_filter=["test_server"])

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "test_tool"
        tc.function.arguments = {}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Error occurred."

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            result = await agent.chat("test")

        assert result == "Error occurred."
        # Error message should be in tool history
        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        assert "Error calling test_tool" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_clear_history(self, mock_bridge):
        agent = BaseAgent("test", "model", "prompt", mock_bridge)
        agent.history = [{"role": "user", "content": "hi"}]
        agent.clear_history()
        assert agent.history == []

    def test_tool_filter_restricts_tools(self, mock_bridge):
        agent = BaseAgent("test", "model", "prompt", mock_bridge, tool_filter=["other_server"])
        assert len(agent.tools) == 0

    def test_no_filter_returns_all(self, mock_bridge):
        agent = BaseAgent("test", "model", "prompt", mock_bridge, tool_filter=None)
        assert len(agent.tools) == 1
