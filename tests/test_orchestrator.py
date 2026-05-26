"""Tests for the multi-agent orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.agents.orchestrator import (
    MAX_DELEGATION_DEPTH,
    Orchestrator,
    _DELEGATION_STACK,
    _delegation_scope,
    _keyword_classify,
    create_orchestrator,
)
from coral.config import set_cli_model


# ── Keyword classification tests ──


class TestKeywordClassify:
    """Test fast keyword-based pre-router."""

    @pytest.mark.parametrize(
        "query,expected",
        [
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
            # UFS experiment queries -> WORKFLOW
            ("Set up a SCHISM experiment", ["WORKFLOW"]),
            ("Submit experiment to Slurm", ["WORKFLOW"]),
            ("Create experiment for UFS-Coastal", ["WORKFLOW"]),
            ("Check the run status of my experiment", ["WORKFLOW"]),
            # HPC system queries -> WORKFLOW
            ("How much scratch space am I using?", ["WORKFLOW"]),
            ("Show my disk quota", ["WORKFLOW"]),
            ("What's my FairShare status?", ["WORKFLOW"]),
            ("What modules loaded in my environment?", ["WORKFLOW"]),
            ("What groups am I in? Show my group membership", ["WORKFLOW"]),
            ("Show my allocation usage", ["WORKFLOW"]),
            ("What partitions are available?", ["WORKFLOW"]),
        ],
    )
    def test_single_category(self, query, expected):
        result = _keyword_classify(query)
        assert result is not None
        categories, confidence, matched = result
        assert categories[0] == expected[0]
        assert 0 < confidence <= 1.0
        assert isinstance(matched, list)

    @pytest.mark.parametrize(
        "query,expected_contains",
        [
            # Multi-category queries
            ("Plot the water levels from this NetCDF file", ["DATA", "CODE"]),
            ("Compare STOFS forecast against observations and plot it", ["DATA", "CODE"]),
            ("My Slurm job failed, explain the error from the docs", ["WORKFLOW", "CODE"]),
            ("Set up a UFS experiment and plot the outputs", ["WORKFLOW", "CODE"]),
        ],
    )
    def test_multi_category(self, query, expected_contains):
        result = _keyword_classify(query)
        assert result is not None
        categories, confidence, _matched = result
        for cat in expected_contains:
            assert cat in categories

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Compare STOFS forecast against observations and plot it", ["DATA", "CODE"]),
            ("My Slurm job failed, explain the error from the docs", ["WORKFLOW", "CODE"]),
        ],
    )
    def test_multi_category_order(self, query, expected):
        result = _keyword_classify(query)
        assert result is not None
        categories, _conf, _matched = result
        assert categories == expected

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
    async def test_llm_substring_does_not_inject_category(self, mock_bridge):
        """A reply mentioning 'DECODE' must not be read as the CODE category."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_response = MagicMock()
        mock_response.message.content = "DECODE THE DATA"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.classify("zzz")  # no keyword match -> LLM path

        assert result == ["DATA"]

    @pytest.mark.asyncio
    async def test_llm_plural_category_accepted(self, mock_bridge):
        """A pluralized category name ('WORKFLOWS') is still recognized."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_response = MagicMock()
        mock_response.message.content = "WORKFLOWS"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.classify("zzz")

        assert result == ["WORKFLOW"]

    @pytest.mark.asyncio
    async def test_llm_garbage_defaults_to_data(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        mock_response = MagicMock()
        mock_response.message.content = "I'm not sure what you mean"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            result = await orch.classify("xyz")

        assert result == ["DATA"]


class TestLastRouteDecision:
    """The orchestrator records the routing decision for /route to read back."""

    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        return bridge

    @pytest.mark.asyncio
    async def test_keyword_route_records_decision(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        with patch("coral.agents.orchestrator.ollama"):
            await orch.classify("What is the water level at Newport?")

        decision = orch.last_route_decision
        assert decision is not None
        assert decision["method"] == "keyword"
        assert "DATA" in decision["categories"]
        assert decision["confidence"] > 0
        assert "water level" in decision["matched_keywords"]
        assert decision["query"].startswith("What is the water level")
        # Backwards-compat property
        assert orch.last_route_confidence == decision["confidence"]

    @pytest.mark.asyncio
    async def test_llm_route_records_decision(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        mock_response = MagicMock()
        mock_response.message.content = "WORKFLOW, CODE"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            await orch.classify("Hello there")

        decision = orch.last_route_decision
        assert decision is not None
        assert decision["method"] == "llm"
        assert decision["matched_keywords"] == []
        assert "router_model" in decision

    @pytest.mark.asyncio
    async def test_default_route_records_decision(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        mock_response = MagicMock()
        mock_response.message.content = "no idea here"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_response
            await orch.classify("xyz")

        decision = orch.last_route_decision
        assert decision is not None
        assert decision["method"] == "default"
        assert decision["categories"] == ["DATA"]
        assert decision["confidence"] == 0.3


class TestConfidenceEscalation:
    """Ambiguous (low-confidence) multi-section keyword routes defer to the LLM."""

    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        return bridge

    @pytest.mark.asyncio
    async def test_ambiguous_multi_route_escalates_to_llm(self, mock_bridge):
        # "explain the tide" -> keyword route [DATA, CODE] at confidence 0.59.
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        llm_response = MagicMock()
        llm_response.message.content = "DATA"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = llm_response
            result = await orch.classify("explain the tide")

        mock_ollama.chat.assert_called_once()
        assert result == ["DATA"]
        decision = orch.last_route_decision
        assert decision["method"] == "llm"
        assert decision["escalated_from_keyword"]["categories"] == ["DATA", "CODE"]

    @pytest.mark.asyncio
    async def test_strong_multi_route_skips_llm(self, mock_bridge):
        # Confident cross-domain query (0.77) is trusted without the LLM.
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            result = await orch.classify("Plot the water levels from this NetCDF file")

        mock_ollama.chat.assert_not_called()
        assert set(result) == {"DATA", "CODE"}
        assert orch.last_route_decision["method"] == "keyword"

    @pytest.mark.asyncio
    async def test_single_category_never_escalates(self, mock_bridge):
        # Single-section route is trusted even below the threshold (conf 0.6).
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            result = await orch.classify("What is the tide?")

        mock_ollama.chat.assert_not_called()
        assert result == ["DATA"]
        assert orch.last_route_decision["method"] == "keyword"

    @pytest.mark.asyncio
    async def test_keyword_fallback_when_llm_unhelpful(self, mock_bridge):
        # Escalated, but the LLM returns nothing usable -> keep the keyword guess.
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        llm_response = MagicMock()
        llm_response.message.content = "i'm not sure"

        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            mock_ollama.chat.return_value = llm_response
            result = await orch.classify("explain the tide")

        mock_ollama.chat.assert_called_once()
        assert set(result) == {"DATA", "CODE"}
        assert orch.last_route_decision["method"] == "keyword_fallback"

    @pytest.mark.asyncio
    async def test_env_var_can_disable_escalation(self, mock_bridge, monkeypatch):
        monkeypatch.setenv("CORAL_ROUTE_MIN_CONFIDENCE", "0")
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        with patch("coral.agents.orchestrator.ollama") as mock_ollama:
            result = await orch.classify("explain the tide")

        mock_ollama.chat.assert_not_called()
        assert set(result) == {"DATA", "CODE"}
        assert orch.last_route_decision["method"] == "keyword"


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

        with (
            patch("coral.agents.base.ollama") as mock_base_ollama,
            patch("coral.agents.orchestrator.ollama") as mock_orch_ollama,
        ):
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


class TestOrchestratorConstructor:
    """Direct construction: constructor args are authoritative."""

    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        return bridge

    def test_default_all_same_model(self, mock_bridge):
        orch = Orchestrator(model="qwen3:32b", mcp_bridge=mock_bridge)
        assert orch.router_model == "qwen3:32b"
        assert orch.synthesis_model == "qwen3:32b"
        assert orch.agents["DATA"].model == "qwen3:32b"
        assert orch.agents["CODE"].model == "qwen3:32b"
        assert orch.agents["WORKFLOW"].model == "qwen3:32b"

    def test_explicit_stage_models(self, mock_bridge):
        orch = Orchestrator(
            model="base",
            mcp_bridge=mock_bridge,
            router_model="router-m",
            synthesis_model="synth-m",
            code_model="code-m",
        )
        assert orch.router_model == "router-m"
        assert orch.synthesis_model == "synth-m"
        assert orch.agents["CODE"].model == "code-m"
        # Unset stages fall back to model=
        assert orch.agents["DATA"].model == "base"
        assert orch.agents["WORKFLOW"].model == "base"


class TestCreateOrchestratorFactory:
    """Factory uses the central config resolver (env vars + CLI model)."""

    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        return bridge

    def _clear_env(self, monkeypatch):
        for var in [
            "CORAL_MODEL",
            "CORAL_MODEL_ROUTER",
            "CORAL_MODEL_SYNTHESIS",
            "CORAL_MODEL_DATA",
            "CORAL_MODEL_CODE",
            "CORAL_MODEL_WORKFLOW",
            "CORAL_MODEL_ESCALATION",
        ]:
            monkeypatch.delenv(var, raising=False)
        set_cli_model("")

    def test_stage_override_code(self, mock_bridge, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_CODE", "qwen3-coder")
        orch = create_orchestrator(model="qwen3:32b", mcp_bridge=mock_bridge)
        assert orch.agents["CODE"].model == "qwen3-coder"
        assert orch.agents["DATA"].model == "qwen3:32b"
        assert orch.agents["WORKFLOW"].model == "qwen3:32b"

    def test_router_and_synthesis_override(self, mock_bridge, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_ROUTER", "small-router")
        monkeypatch.setenv("CORAL_MODEL_SYNTHESIS", "synth-model")
        orch = create_orchestrator(model="qwen3:32b", mcp_bridge=mock_bridge)
        assert orch.router_model == "small-router"
        assert orch.synthesis_model == "synth-model"

    def test_cli_model_is_tier_3(self, mock_bridge, monkeypatch):
        """CLI --model is tier 3: stage env -> CORAL_MODEL -> CLI --model."""
        self._clear_env(monkeypatch)
        set_cli_model("cli-model")
        try:
            orch = create_orchestrator(model="cli-model", mcp_bridge=mock_bridge)
            assert orch.agents["DATA"].model == "cli-model"
            assert orch.router_model == "cli-model"
        finally:
            set_cli_model("")

    def test_coral_model_env_beats_cli(self, mock_bridge, monkeypatch):
        """CORAL_MODEL env var takes precedence over CLI --model."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "env-model")
        set_cli_model("cli-model")
        try:
            orch = create_orchestrator(model="cli-model", mcp_bridge=mock_bridge)
            assert orch.agents["DATA"].model == "env-model"
            assert orch.router_model == "env-model"
        finally:
            set_cli_model("")

    def test_stage_env_beats_coral_model(self, mock_bridge, monkeypatch):
        """Stage env var takes precedence over CORAL_MODEL."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "env-model")
        monkeypatch.setenv("CORAL_MODEL_CODE", "stage-coder")
        orch = create_orchestrator(model="env-model", mcp_bridge=mock_bridge)
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


# ── Agent-to-agent delegation tests ──


class TestDelegation:
    """The orchestrator wires delegate() into each agent as an ask_section tool."""

    @pytest.fixture
    def mock_bridge(self):
        bridge = MagicMock()
        bridge.tools = []
        bridge.tool_server_map = {}
        bridge.call_tool = AsyncMock(return_value="tool result")
        return bridge

    def test_wiring_gives_agents_their_peers(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        data = orch.agents["DATA"]
        assert data.delegate_fn is not None
        assert set(data.delegate_peers) == {"CODE", "WORKFLOW"}
        names = {t["function"]["name"] for t in data.tools}
        assert "ask_section" in names

    def test_delegation_disabled_via_env(self, mock_bridge, monkeypatch):
        monkeypatch.setenv("CORAL_DELEGATION", "off")
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        data = orch.agents["DATA"]
        assert data.delegate_fn is None
        names = {t["function"]["name"] for t in data.tools}
        assert "ask_section" not in names

    @pytest.mark.asyncio
    async def test_delegate_runs_peer_and_clears_history(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)

        peer_response = MagicMock()
        peer_response.message.tool_calls = None
        peer_response.message.content = "Peer answer."

        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.return_value = peer_response
            result = await orch.delegate("DATA", "CODE", "explain the CFL condition")

        assert result == "Peer answer."
        # Target session must not retain the delegated turn.
        assert orch.agents["CODE"].history == []

    @pytest.mark.asyncio
    async def test_delegate_accepts_lowercase_section(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        peer_response = MagicMock()
        peer_response.message.tool_calls = None
        peer_response.message.content = "ok"
        with patch("coral.agents.base.ollama") as mock_ollama:
            mock_ollama.chat.return_value = peer_response
            result = await orch.delegate("DATA", "code", "q")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_delegate_unknown_section(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        result = await orch.delegate("DATA", "NONSENSE", "q")
        assert "unknown section" in result.lower()

    @pytest.mark.asyncio
    async def test_delegate_to_self_refused(self, mock_bridge):
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        result = await orch.delegate("DATA", "DATA", "q")
        assert "your own section" in result.lower()

    @pytest.mark.asyncio
    async def test_delegate_reentrancy_refused(self, mock_bridge):
        """Cannot delegate into a section already on the call stack."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        with _delegation_scope("CODE"):
            result = await orch.delegate("DATA", "CODE", "q")
        assert "already handling" in result.lower()

    @pytest.mark.asyncio
    async def test_delegate_depth_limit_refused(self, mock_bridge):
        """Once the delegation stack is full, further delegation is refused."""
        orch = Orchestrator(model="test", mcp_bridge=mock_bridge)
        full_stack = tuple(f"X{i}" for i in range(MAX_DELEGATION_DEPTH))
        token = _DELEGATION_STACK.set(full_stack)
        try:
            result = await orch.delegate("DATA", "CODE", "q")
        finally:
            _DELEGATION_STACK.reset(token)
        assert "limit reached" in result.lower()
