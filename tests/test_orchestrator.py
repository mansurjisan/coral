"""Tests for the multi-agent orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.agents.orchestrator import Orchestrator, _keyword_classify
from coral.config import set_cli_model


# ── Keyword classification tests ──


class TestKeywordClassify:
    """Test fast keyword-based pre-router."""

    @pytest.mark.parametrize("query,expected", [
        # Single DATA queries
        ("What is the water level at The Battery?", ["DATA"]),
        ("Are there any active hurricanes?", ["DATA"]),
        ("Get SST from ERDDAP for the Gulf of Maine", ["DATA"]),
        ("Inspect the NetCDF file output.nc", ["DATA"]),
        ("What is the current tide at station 8518750?", ["DATA"]),
        ("What is the streamflow at this USGS gauge?", ["DATA"]),
        ("Show me the latest GOES satellite image", ["DATA"]),
        ("What is the wave height at the buoy?", ["DATA"]),
        ("Get wind observations at the weather station", ["DATA"]),
        ("What is the STOFS forecast at The Battery?", ["DATA"]),

        # Single CODE queries
        ("What does schism_init do?", ["CODE"]),
        ("Explain the wetting drying algorithm in ADCIRC", ["CODE"]),
        ("What parameters are in the core namelist?", ["CODE"]),

        # Single WORKFLOW queries
        ("Why did my Slurm job fail?", ["WORKFLOW"]),
        ("Show me my recent jobs", ["WORKFLOW"]),
        ("What tasks are aborted in the stofs suite?", ["WORKFLOW"]),
        ("Read the log for job 12345", ["WORKFLOW"]),
    ])
    def test_single_category(self, query, expected):
        result = _keyword_classify(query)
        assert result is not None
        assert result[0] == expected[0]

    @pytest.mark.parametrize("query,expected_contains", [
        # Multi-category queries
        ("Plot the water levels from this NetCDF file", ["DATA", "CODE"]),
        ("Compare STOFS forecast against observations and plot it", ["DATA", "CODE"]),
        ("My Slurm job failed, explain the error from the docs", ["WORKFLOW", "CODE"]),
    ])
    def test_multi_category(self, query, expected_contains):
        result = _keyword_classify(query)
        assert result is not None
        for cat in expected_contains:
            assert cat in result

    @pytest.mark.parametrize("query,expected", [
        ("Compare STOFS forecast against observations and plot it", ["DATA", "CODE"]),
        ("My Slurm job failed, explain the error from the docs", ["WORKFLOW", "CODE"]),
    ])
    def test_multi_category_order(self, query, expected):
        result = _keyword_classify(query)
        assert result == expected

    def test_ambiguous_returns_none(self):
        """Unknown queries should return None to trigger LLM fallback."""
        result = _keyword_classify("Hello, how are you?")
        assert result is None

    def test_ambiguous_model_name_returns_none(self):
        """Model names alone should not force a route."""
        result = _keyword_classify("Tell me about STOFS")
        assert result is None

    def test_empty_query_returns_none(self):
        result = _keyword_classify("")
        assert result is None


# ── Orchestrator classification tests (mocked LLM) ──


class TestOrchestratorClassify:
    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        return bridge

    @pytest.mark.asyncio
    async def test_keyword_route_skips_llm(self, mock_bridge):
        """Queries with clear keywords should not call Ollama."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            result = await orch.classify("What is the water level at Newport?")

        assert "DATA" in result
        mock_ollama.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_falls_back_to_llm(self, mock_bridge):
        """Ambiguous queries should call Ollama for classification."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_response = MagicMock()
        mock_response.message.content = "DATA"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.classify("Hello there")

        assert result == ["DATA"]
        mock_ollama.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_multiple_categories(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_response = MagicMock()
        mock_response.message.content = "WORKFLOW, CODE"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.classify("Tell me a joke")  # no keyword match

        assert "WORKFLOW" in result
        assert "CODE" in result

    @pytest.mark.asyncio
    async def test_llm_garbage_defaults_to_data(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_response = MagicMock()
        mock_response.message.content = "I'm not sure what you mean"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.classify("xyz")

        assert result == ["DATA"]


# ── Orchestrator chat flow tests ──


class TestOrchestratorChat:
    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = [
            {"type": "function", "function": {"name": "coops_get_water_levels", "description": "", "parameters": {}}}
        ]
        bridge.tool_server_map = {"coops_get_water_levels": "coops"}
        bridge.call_tool = AsyncMock(return_value="Water level: 0.5m")
        return bridge

    @pytest.mark.asyncio
    async def test_single_agent_flow(self, mock_bridge):
        """Single-category query routes to one agent and returns response."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        # Mock the data agent's chat method
        mock_response = MagicMock()
        mock_response.message.tool_calls = None
        mock_response.message.content = "Water level is 0.5m MLLW."

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.chat("What is the water level at Newport?")

        assert "0.5m" in result
        assert len(orch.history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_multi_agent_calls_synthesize(self, mock_bridge):
        """Multi-category query should call multiple agents and synthesize."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_agent_response = MagicMock()
        mock_agent_response.message.tool_calls = None
        mock_agent_response.message.content = "Agent result."

        mock_synth_response = MagicMock()
        mock_synth_response.message.content = "Synthesized answer."

        with patch("coral.agents.base.ollama") as mock_base_ollama, \
             patch("coral.agents.orchestrator.ollama") as mock_orch_ollama:
            mock_base_ollama.chat.return_value = mock_agent_response
            # First call = classification (keyword handles it), rest = synthesis
            mock_orch_ollama.chat.return_value = mock_synth_response

            # Force multi-agent by patching classify
            orch.classify = AsyncMock(return_value=["DATA", "CODE"])
            result = await orch.chat("Plot the water levels from this file")

        assert result == "Synthesized answer."

    @pytest.mark.asyncio
    async def test_agent_failure_returns_partial_results(self, mock_bridge):
        """If one agent fails, the orchestrator should still synthesize partial results."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        # DATA agent will succeed, CODE agent will fail
        call_count = 0

        async def mock_chat(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # CODE agent (second call)
                raise RuntimeError("API timeout")
            return "Water level data found."

        orch.agents["DATA"].chat = mock_chat
        orch.agents["CODE"].chat = mock_chat

        mock_synth_response = MagicMock()
        mock_synth_response.message.content = "Partial answer with data."

        with patch("coral.agents.orchestrator.ollama") as mock_orch_ollama:
            mock_orch_ollama.chat.return_value = mock_synth_response
            orch.classify = AsyncMock(return_value=["DATA", "CODE"])
            result = await orch.chat("Plot the water levels")

        assert result == "Partial answer with data."

    @pytest.mark.asyncio
    async def test_reset_clears_all(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        # Add some fake history
        orch.history.append({"role": "user", "content": "test"})
        orch.agents["DATA"].history.append({"role": "user", "content": "test"})

        orch.reset()

        assert len(orch.history) == 0
        assert len(orch.agents["DATA"].history) == 0


# ── Per-stage model resolution tests ──


class TestOrchestratorModels:
    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        return bridge

    def test_default_all_same_model(self, mock_bridge, monkeypatch):
        monkeypatch.delenv("CORAL_MODEL", raising=False)
        monkeypatch.delenv("CORAL_MODEL_ROUTER", raising=False)
        monkeypatch.delenv("CORAL_MODEL_SYNTHESIS", raising=False)
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        monkeypatch.delenv("CORAL_MODEL_CODE", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        orch = Orchestrator(model="qwen3:32b", mcp_bridge=mock_bridge)
        assert orch.router_model == "qwen3:32b"
        assert orch.synthesis_model == "qwen3:32b"
        assert orch.agents["DATA"].model == "qwen3:32b"
        assert orch.agents["CODE"].model == "qwen3:32b"
        assert orch.agents["WORKFLOW"].model == "qwen3:32b"

    def test_stage_override_code(self, mock_bridge, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_CODE", "qwen3-coder")
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        monkeypatch.delenv("CORAL_MODEL_ROUTER", raising=False)
        monkeypatch.delenv("CORAL_MODEL_SYNTHESIS", raising=False)
        orch = Orchestrator(model="qwen3:32b", mcp_bridge=mock_bridge)
        assert orch.agents["CODE"].model == "qwen3-coder"
        assert orch.agents["DATA"].model == "qwen3:32b"
        assert orch.agents["WORKFLOW"].model == "qwen3:32b"

    def test_router_and_synthesis_override(self, mock_bridge, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_ROUTER", "small-router")
        monkeypatch.setenv("CORAL_MODEL_SYNTHESIS", "synth-model")
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        monkeypatch.delenv("CORAL_MODEL_CODE", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        orch = Orchestrator(model="qwen3:32b", mcp_bridge=mock_bridge)
        assert orch.router_model == "small-router"
        assert orch.synthesis_model == "synth-model"

    def test_cli_model_is_tier_3(self, mock_bridge, monkeypatch):
        """CLI --model is tier 3: stage env -> CORAL_MODEL -> CLI --model."""
        monkeypatch.delenv("CORAL_MODEL", raising=False)
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        monkeypatch.delenv("CORAL_MODEL_CODE", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        monkeypatch.delenv("CORAL_MODEL_ROUTER", raising=False)
        monkeypatch.delenv("CORAL_MODEL_SYNTHESIS", raising=False)
        # Simulate CLI setting --model
        set_cli_model("cli-model")
        try:
            orch = Orchestrator(model="cli-model", mcp_bridge=mock_bridge)
            assert orch.agents["DATA"].model == "cli-model"
            assert orch.router_model == "cli-model"
        finally:
            set_cli_model("")

    def test_coral_model_env_beats_cli(self, mock_bridge, monkeypatch):
        """CORAL_MODEL env var takes precedence over CLI --model."""
        monkeypatch.setenv("CORAL_MODEL", "env-model")
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        monkeypatch.delenv("CORAL_MODEL_CODE", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        monkeypatch.delenv("CORAL_MODEL_ROUTER", raising=False)
        monkeypatch.delenv("CORAL_MODEL_SYNTHESIS", raising=False)
        set_cli_model("cli-model")
        try:
            orch = Orchestrator(model="cli-model", mcp_bridge=mock_bridge)
            # CORAL_MODEL wins over CLI
            assert orch.agents["DATA"].model == "env-model"
            assert orch.router_model == "env-model"
        finally:
            set_cli_model("")

    def test_stage_env_beats_coral_model(self, mock_bridge, monkeypatch):
        """Stage env var takes precedence over CORAL_MODEL."""
        monkeypatch.setenv("CORAL_MODEL", "env-model")
        monkeypatch.setenv("CORAL_MODEL_CODE", "stage-coder")
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        monkeypatch.delenv("CORAL_MODEL_ROUTER", raising=False)
        monkeypatch.delenv("CORAL_MODEL_SYNTHESIS", raising=False)
        orch = Orchestrator(model="env-model", mcp_bridge=mock_bridge)
        assert orch.agents["CODE"].model == "stage-coder"
        assert orch.agents["DATA"].model == "env-model"


# ── Agent tool filtering tests ──


class TestAgentToolFiltering:
    def _make_tool(self, name: str) -> dict:
        return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}

    def test_data_agent_sees_only_data_tools(self):
        from coral.agents.data_agent import create_data_agent

        bridge = MagicMock()
        bridge.tools = [
            self._make_tool("coops_get_water_levels"),
            self._make_tool("nhc_get_storms"),
            self._make_tool("search_documentation"),
            self._make_tool("get_my_jobs"),
            self._make_tool("execute_python"),
        ]
        bridge.tool_server_map = {
            "coops_get_water_levels": "coops",
            "nhc_get_storms": "nhc",
            "search_documentation": "rag",
            "get_my_jobs": "slurm",
            "execute_python": "viz",
        }

        agent = create_data_agent("test", bridge)
        tool_names = {t["function"]["name"] for t in agent.tools}

        assert "coops_get_water_levels" in tool_names
        assert "nhc_get_storms" in tool_names
        assert "search_documentation" not in tool_names
        assert "get_my_jobs" not in tool_names
        assert "execute_python" not in tool_names

    def test_code_agent_sees_only_code_tools(self):
        from coral.agents.code_agent import create_code_agent

        bridge = MagicMock()
        bridge.tools = [
            self._make_tool("coops_get_water_levels"),
            self._make_tool("search_documentation"),
            self._make_tool("execute_python"),
            self._make_tool("get_my_jobs"),
        ]
        bridge.tool_server_map = {
            "coops_get_water_levels": "coops",
            "search_documentation": "rag",
            "execute_python": "viz",
            "get_my_jobs": "slurm",
        }

        agent = create_code_agent("test", bridge)
        tool_names = {t["function"]["name"] for t in agent.tools}

        assert "search_documentation" in tool_names
        assert "execute_python" in tool_names
        assert "coops_get_water_levels" not in tool_names
        assert "get_my_jobs" not in tool_names

    def test_workflow_agent_sees_only_workflow_tools(self):
        from coral.agents.workflow_agent import create_workflow_agent

        bridge = MagicMock()
        bridge.tools = [
            self._make_tool("get_my_jobs"),
            self._make_tool("get_suite_status"),
            self._make_tool("read_file"),
            self._make_tool("coops_get_water_levels"),
            self._make_tool("search_documentation"),
        ]
        bridge.tool_server_map = {
            "get_my_jobs": "slurm",
            "get_suite_status": "ecflow",
            "read_file": "filesystem",
            "coops_get_water_levels": "coops",
            "search_documentation": "rag",
        }

        agent = create_workflow_agent("test", bridge)
        tool_names = {t["function"]["name"] for t in agent.tools}

        assert "get_my_jobs" in tool_names
        assert "get_suite_status" in tool_names
        assert "read_file" not in tool_names
        assert "coops_get_water_levels" not in tool_names
        assert "search_documentation" not in tool_names

    def test_no_filter_returns_all_tools(self):
        from coral.agents.base import BaseAgent

        bridge = MagicMock()
        bridge.tools = [
            self._make_tool("tool_a"),
            self._make_tool("tool_b"),
        ]

        agent = BaseAgent("test", "model", "prompt", bridge, tool_filter=None)
        assert len(agent.tools) == 2
