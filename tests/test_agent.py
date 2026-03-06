"""Tests for the CORAL agent.

Unit tests mock Ollama and MCP so they run without external services.
Integration tests (marked @pytest.mark.integration) require Ollama running.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.agent import CoralAgent, _select_tools


# ── Tool selection tests ──


class TestSelectTools:
    """Test keyword-based tool filtering."""

    def _make_tool(self, name: str) -> dict:
        return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}

    def setup_method(self):
        self.all_tools = [
            self._make_tool("coops_get_water_levels"),
            self._make_tool("coops_get_station"),
            self._make_tool("nhc_get_active_storms"),
            self._make_tool("nhc_get_best_track"),
            self._make_tool("stofs_get_station_forecast"),
            self._make_tool("erddap_search_datasets"),
            self._make_tool("recon_list_missions"),
            self._make_tool("ofs_get_forecast_at_point"),
            self._make_tool("search_documentation"),
            self._make_tool("inspect_netcdf"),
            self._make_tool("query_netcdf"),
            self._make_tool("get_my_jobs"),
            self._make_tool("diagnose_job_failure"),
            self._make_tool("execute_python"),
            self._make_tool("adcirc_parse_fort15"),
            self._make_tool("schism_parse_param_nml"),
            self._make_tool("usgs_get_flood_status"),
            self._make_tool("winds_get_latest_observation"),
            self._make_tool("ww3_get_forecast_at_point"),
            self._make_tool("goes_get_latest_image"),
        ]

    def test_water_level_selects_coops(self):
        tools = _select_tools("What is the current water level at Newport?", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "coops_get_water_levels" in names
        assert "nhc_get_active_storms" not in names

    def test_hurricane_selects_nhc(self):
        tools = _select_tools("Show me active hurricanes", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "nhc_get_active_storms" in names
        assert "coops_get_water_levels" not in names

    def test_surge_selects_stofs_and_coops(self):
        tools = _select_tools("What is the storm surge forecast?", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "stofs_get_station_forecast" in names
        assert "coops_get_water_levels" in names

    def test_schism_selects_schism(self):
        tools = _select_tools("Parse my param.nml file", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "schism_parse_param_nml" in names

    def test_adcirc_selects_adcirc(self):
        tools = _select_tools("Explain the ADCIRC fort.15 parameters", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "adcirc_parse_fort15" in names

    def test_waves_selects_ww3(self):
        tools = _select_tools("What is the wave height at the buoy?", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "ww3_get_forecast_at_point" in names

    def test_usgs_selects_usgs(self):
        tools = _select_tools("What is the streamflow at this USGS gauge?", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "usgs_get_flood_status" in names

    def test_winds_selects_winds(self):
        tools = _select_tools("Get wind observations at the weather station", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "winds_get_latest_observation" in names

    def test_goes_selects_goes(self):
        tools = _select_tools("Show me the latest GOES satellite image", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "goes_get_latest_image" in names

    def test_netcdf_selects_netcdf(self):
        tools = _select_tools("Inspect the netcdf file output.nc", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "inspect_netcdf" in names
        assert "query_netcdf" in names

    def test_slurm_selects_slurm(self):
        tools = _select_tools("Show me my failed Slurm jobs", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "get_my_jobs" in names
        assert "diagnose_job_failure" in names

    def test_plot_selects_execute_python(self):
        tools = _select_tools("Plot the water levels", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "execute_python" in names

    def test_unknown_query_returns_all(self):
        tools = _select_tools("Hello, how are you?", self.all_tools)
        assert len(tools) == len(self.all_tools)

    def test_documentation_selects_rag(self):
        tools = _select_tools("What does the subroutine schism_init do?", self.all_tools)
        names = {t["function"]["name"] for t in tools}
        assert "search_documentation" in names


# ── Agent chat tests (mocked) ──


class TestCoralAgent:
    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = [
            {"type": "function", "function": {"name": "coops_get_water_levels", "description": "Get water levels", "parameters": {}}}
        ]
        bridge.call_tool = AsyncMock(return_value="Water level: 0.5m MLLW")
        return bridge

    @pytest.mark.asyncio
    async def test_simple_text_response(self, mock_bridge):
        """Agent returns text when model doesn't call tools."""
        mock_response = MagicMock()
        mock_response.message.tool_calls = None
        mock_response.message.content = "Hello! I'm CORAL."

        with patch("coral.agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            agent = CoralAgent(model="test", mcp_bridge=mock_bridge)
            result = await agent.chat("Hello")

        assert result == "Hello! I'm CORAL."
        assert len(agent.history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_tool_call_flow(self, mock_bridge):
        """Agent calls tool and returns final text response."""
        # First response: model wants to call a tool
        tool_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "coops_get_water_levels"
        tool_call.function.arguments = {"station_id": "8452660"}
        tool_response.message.tool_calls = [tool_call]
        tool_response.message.content = ""

        # Second response: model gives text answer
        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Water level at Newport is 0.5m MLLW."

        with patch("coral.agent.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            agent = CoralAgent(model="test", mcp_bridge=mock_bridge)
            result = await agent.chat("Water level at Newport?")

        assert "0.5m" in result
        mock_bridge.call_tool.assert_called_once_with("coops_get_water_levels", {"station_id": "8452660"})

    @pytest.mark.asyncio
    async def test_tool_call_callback(self, mock_bridge):
        """on_tool_call callback is invoked with tool name, args, and result."""
        tool_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "coops_get_water_levels"
        tool_call.function.arguments = {"station_id": "8452660"}
        tool_response.message.tool_calls = [tool_call]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Done."

        callback_calls = []

        with patch("coral.agent.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            agent = CoralAgent(
                model="test", mcp_bridge=mock_bridge,
                on_tool_call=lambda name, args, result: callback_calls.append((name, args, result)),
            )
            await agent.chat("Water level?")

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "coops_get_water_levels"

    @pytest.mark.asyncio
    async def test_tool_error_handled(self, mock_bridge):
        """Agent handles tool call errors gracefully."""
        mock_bridge.call_tool = AsyncMock(side_effect=Exception("Connection refused"))

        tool_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "coops_get_water_levels"
        tool_call.function.arguments = {}
        tool_response.message.tool_calls = [tool_call]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Sorry, error occurred."

        with patch("coral.agent.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            agent = CoralAgent(model="test", mcp_bridge=mock_bridge)
            result = await agent.chat("Water level?")

        assert "error" in result.lower() or "Sorry" in result

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self, mock_bridge):
        """Agent stops after MAX_TOOL_ITERATIONS even if model keeps calling tools."""
        tool_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "coops_get_water_levels"
        tool_call.function.arguments = {}
        tool_response.message.tool_calls = [tool_call]
        tool_response.message.content = ""

        final_response = MagicMock()
        final_response.message.tool_calls = None
        final_response.message.content = "Finally done."

        with patch("coral.agent.ollama") as mock_ollama:
            # Return tool calls 10 times, then text
            mock_ollama.chat.side_effect = [tool_response] * 10 + [final_response]
            agent = CoralAgent(model="test", mcp_bridge=mock_bridge)
            result = await agent.chat("Keep going")

        assert result == "Finally done."

    @pytest.mark.asyncio
    async def test_reset_clears_history(self, mock_bridge):
        mock_response = MagicMock()
        mock_response.message.tool_calls = None
        mock_response.message.content = "Hi"

        with patch("coral.agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            agent = CoralAgent(model="test", mcp_bridge=mock_bridge)
            await agent.chat("Hello")
            assert len(agent.history) > 0
            agent.reset()
            assert len(agent.history) == 0

    @pytest.mark.asyncio
    async def test_large_tool_response_truncated(self, mock_bridge):
        """Very large tool responses get truncated."""
        large_result = "x" * 10000
        mock_bridge.call_tool = AsyncMock(return_value=large_result)

        tool_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "coops_get_water_levels"
        tool_call.function.arguments = {}
        tool_response.message.tool_calls = [tool_call]
        tool_response.message.content = ""

        text_response = MagicMock()
        text_response.message.tool_calls = None
        text_response.message.content = "Done."

        with patch("coral.agent.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [tool_response, text_response]
            agent = CoralAgent(model="test", mcp_bridge=mock_bridge)
            await agent.chat("test")

        # Check the tool message in history was truncated
        tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert len(tool_msgs[0]["content"]) < 10000
        assert "truncated" in tool_msgs[0]["content"]
