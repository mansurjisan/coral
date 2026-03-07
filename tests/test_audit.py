"""Tests for request-context and server-boundary audit logging."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.agent import CoralAgent
from coral.agents.orchestrator import Orchestrator
from coral.audit import request_context, section_context, split_tool_audit_payload, with_tool_audit_payload
from coral.mcp_bridge import MCPBridge


class TestAuditHelpers:
    def test_split_tool_audit_payload(self):
        result = with_tool_audit_payload("hello", sandbox_used=True)
        payload, cleaned = split_tool_audit_payload(result)
        assert payload == {"sandbox_used": True}
        assert cleaned == "hello"


class TestBridgeAudit:
    @pytest.mark.asyncio
    async def test_call_tool_records_audit_event(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "result data"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result

        bridge.tool_map["coops_get_water_levels"] = (mock_session, "coops")
        bridge.tool_server_map["coops_get_water_levels"] = "coops"

        with request_context(query_id="query-123", route=["DATA"], mode="multi"), \
             section_context("data"), \
             patch("coral.mcp_bridge.record_audit_event") as mock_record:
            result = await bridge.call_tool("coops_get_water_levels", {"station": "8518750"})

        assert result == "result data"
        mock_record.assert_called_once()
        event_name = mock_record.call_args.args[0]
        fields = mock_record.call_args.kwargs
        assert event_name == "tool_call"
        assert fields["server"] == "coops"
        assert fields["tool"] == "coops_get_water_levels"
        assert fields["success"] is True
        assert fields["sandbox_used"] is None
        assert "8518750" in fields["args_summary"]

    @pytest.mark.asyncio
    async def test_call_tool_strips_viz_audit_payload(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = with_tool_audit_payload("plot ready", sandbox_used=True)
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result

        bridge.tool_map["execute_python"] = (mock_session, "viz")
        bridge.tool_server_map["execute_python"] = "viz"

        with request_context(query_id="query-456", route=["CODE"], mode="multi"), \
             section_context("code"), \
             patch("coral.mcp_bridge.record_audit_event") as mock_record:
            result = await bridge.call_tool("execute_python", {"code": "print(1)"})

        assert result == "plot ready"
        fields = mock_record.call_args.kwargs
        assert fields["server"] == "viz"
        assert fields["sandbox_used"] is True


class TestAgentAuditIntegration:
    @pytest.mark.asyncio
    async def test_orchestrator_sets_route_and_section_for_tool_call(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        bridge.tools = [
            {"type": "function", "function": {"name": "coops_get_water_levels", "description": "", "parameters": {}}}
        ]
        bridge.tool_server_map = {"coops_get_water_levels": "coops"}

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "Water level: 0.5m"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result
        bridge.tool_map["coops_get_water_levels"] = (mock_session, "coops")

        orch = Orchestrator(model="test", mcp_bridge=bridge)

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "coops_get_water_levels"
        tc.function.arguments = {"station": "8518750"}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Water level is 0.5m."

        with patch("coral.agents.base.ollama") as mock_ollama, \
             patch("coral.mcp_bridge.record_audit_event") as mock_record:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            result = await orch.chat("What is the water level at Newport?")

        assert "0.5m" in result

        tool_events = [
            call for call in mock_record.call_args_list
            if call.args and call.args[0] == "tool_call"
        ]
        assert len(tool_events) == 1

        fields = tool_events[0].kwargs
        assert fields["server"] == "coops"
        assert fields["tool"] == "coops_get_water_levels"

    @pytest.mark.asyncio
    async def test_single_agent_records_single_mode_tool_calls(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {}}')
        bridge = MCPBridge(str(config))

        bridge.tools = [
            {"type": "function", "function": {"name": "search_documentation", "description": "", "parameters": {}}}
        ]
        bridge.tool_server_map = {"search_documentation": "rag"}

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "Docs result"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result
        bridge.tool_map["search_documentation"] = (mock_session, "rag")

        agent = CoralAgent(model="test", mcp_bridge=bridge)

        tool_response = MagicMock()
        tc = MagicMock()
        tc.function.name = "search_documentation"
        tc.function.arguments = {"query": "what does schism_init do"}
        tool_response.message.tool_calls = [tc]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Answer"

        with patch("coral.agent.ollama") as mock_ollama, \
             patch("coral.mcp_bridge.record_audit_event") as mock_record:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            result = await agent.chat("What does schism_init do?")

        assert result == "Answer"
        tool_events = [
            call for call in mock_record.call_args_list
            if call.args and call.args[0] == "tool_call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0].kwargs["server"] == "rag"
