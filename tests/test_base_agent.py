"""Tests for the BaseAgent class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.agents.base import BaseAgent, _estimate_history_chars, _prune_history, _truncate_result


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


class TestHistoryPruning:
    def test_estimate_chars(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert _estimate_history_chars(history) == 10

    def test_estimate_includes_tool_call_args(self):
        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "t", "arguments": {"key": "value"}}}],
            },
        ]
        assert _estimate_history_chars(history) > 0

    def test_prune_noop_when_small(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        original_len = len(history)
        _prune_history(history, max_chars=1000)
        assert len(history) == original_len

    def test_prune_removes_oldest_pair(self):
        history = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        _prune_history(history, max_chars=1000)
        # Oldest pair should be gone, recent pair kept
        assert len(history) == 2
        assert history[0]["content"] == "recent question"

    def test_prune_removes_trailing_tool_results(self):
        history = [
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "t" * 3000},
            {"role": "tool", "content": "t" * 3000},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "done"},
        ]
        _prune_history(history, max_chars=500)
        # After pruning oldest user+assistant+tool messages, recent pair remains
        assert history[-1]["content"] == "done"
        assert all(m["role"] != "tool" or m["content"] == "done" for m in history[:2])

    def test_prune_preserves_minimum_two_messages(self):
        history = [
            {"role": "user", "content": "x" * 100_000},
            {"role": "assistant", "content": "y" * 100_000},
        ]
        _prune_history(history, max_chars=100)
        # Should keep at least 2 messages (the guard prevents over-pruning)
        assert len(history) == 2


class TestBaseAgent:
    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = [{"type": "function", "function": {"name": "test_tool", "description": "", "parameters": {}}}]
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
            "test",
            "model",
            "prompt",
            mock_bridge,
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

    def test_peer_hint_lists_other_servers(self, mock_bridge):
        agent = BaseAgent(
            "test",
            "model",
            "prompt",
            mock_bridge,
            tool_filter=["test_server", "stofs", "ofs"],
        )
        hint = agent._peer_server_hint("test_tool")
        assert "stofs" in hint
        assert "ofs" in hint
        assert "test_server" not in hint  # the failing server is excluded

    def test_peer_hint_empty_without_filter(self, mock_bridge):
        agent = BaseAgent("test", "model", "prompt", mock_bridge, tool_filter=None)
        assert agent._peer_server_hint("test_tool") == ""

    def test_peer_hint_empty_when_only_failing_server(self, mock_bridge):
        agent = BaseAgent("test", "model", "prompt", mock_bridge, tool_filter=["test_server"])
        # Tool's server is the only one in the filter, so no peers to suggest.
        assert agent._peer_server_hint("test_tool") == ""

    @pytest.mark.asyncio
    async def test_tool_error_includes_peer_hint(self, mock_bridge):
        """When a tool call fails, the error message should suggest peer servers."""
        mock_bridge.call_tool = AsyncMock(side_effect=Exception("broken"))
        agent = BaseAgent(
            "test",
            "model",
            "prompt",
            mock_bridge,
            tool_filter=["test_server", "stofs", "ofs"],
        )

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "test_tool"
        tc.function.arguments = {}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "ack."

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            await agent.chat("test")

        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        assert "Error calling test_tool" in tool_msgs[0]["content"]
        assert "stofs" in tool_msgs[0]["content"]

    def test_ask_section_tool_exposed_when_wired(self, mock_bridge):
        agent = BaseAgent(
            "data",
            "model",
            "prompt",
            mock_bridge,
            tool_filter=["test_server"],
            delegate_fn=AsyncMock(),
            delegate_peers=["CODE", "WORKFLOW"],
        )
        tools = {t["function"]["name"]: t for t in agent.tools}
        assert "ask_section" in tools
        enum = tools["ask_section"]["function"]["parameters"]["properties"]["section"]["enum"]
        assert enum == ["CODE", "WORKFLOW"]

    def test_ask_section_not_exposed_without_delegate(self, mock_bridge):
        agent = BaseAgent("data", "model", "prompt", mock_bridge, tool_filter=["test_server"])
        names = {t["function"]["name"] for t in agent.tools}
        assert "ask_section" not in names

    @pytest.mark.asyncio
    async def test_delegation_tool_routes_to_delegate_fn(self, mock_bridge):
        """An ask_section call must invoke delegate_fn, not the MCP bridge."""
        delegate = AsyncMock(return_value="peer says hi")
        agent = BaseAgent(
            "data",
            "model",
            "prompt",
            mock_bridge,
            tool_filter=["test_server"],
            delegate_fn=delegate,
            delegate_peers=["CODE"],
        )

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "ask_section"
        tc.function.arguments = {"section": "CODE", "query": "explain the CFL error"}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "done"

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            result = await agent.chat("ask the code section")

        assert result == "done"
        delegate.assert_awaited_once_with("CODE", "explain the CFL error")
        mock_bridge.call_tool.assert_not_called()
        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        assert "peer says hi" in tool_msgs[0]["content"]
