"""Tests for the web UI module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coral.web_ui import _respond


class TestRespond:
    @pytest.mark.asyncio
    async def test_creates_agent_on_first_call(self):
        """First call with None agent should create a new CoralAgent in single mode."""
        import coral.web_ui as web_ui

        mock_bridge = MagicMock()
        mock_bridge.tools = []
        web_ui._bridge = mock_bridge
        web_ui._model = "test-model"
        web_ui._mode = "single"

        with patch("coral.web_ui.CoralAgent") as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.chat = AsyncMock(return_value="Hello!")

            response, agent = await _respond("hi", [], None)

        assert response == "Hello!"
        assert agent is mock_agent
        MockAgent.assert_called_once_with(model="test-model", mcp_bridge=mock_bridge)

    @pytest.mark.asyncio
    async def test_reuses_existing_agent(self):
        """Subsequent calls with existing agent should reuse it."""
        import coral.web_ui as web_ui

        mock_bridge = MagicMock()
        mock_bridge.tools = []
        web_ui._bridge = mock_bridge
        web_ui._mode = "multi"

        existing_agent = MagicMock()
        existing_agent.chat = AsyncMock(return_value="World!")

        with patch("coral.web_ui.CoralAgent") as MockAgent:
            response, agent = await _respond("hello", [], existing_agent)

        assert response == "World!"
        assert agent is existing_agent
        # Should NOT create a new agent
        MockAgent.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_sessions_get_different_agents(self):
        """Two sessions with None agents create separate CoralAgent instances in single mode."""
        import coral.web_ui as web_ui

        mock_bridge = MagicMock()
        mock_bridge.tools = []
        web_ui._bridge = mock_bridge
        web_ui._model = "test"
        web_ui._mode = "single"

        agents_created = []

        with patch("coral.web_ui.CoralAgent") as MockAgent:

            def make_agent(**kwargs):
                agent = MagicMock()
                agent.chat = AsyncMock(return_value="reply")
                agents_created.append(agent)
                return agent

            MockAgent.side_effect = make_agent

            _, agent1 = await _respond("hi", [], None)
            _, agent2 = await _respond("hi", [], None)

        assert len(agents_created) == 2
        assert agent1 is not agent2

    @pytest.mark.asyncio
    async def test_multi_mode_creates_orchestrator(self):
        """Multi mode should create an Orchestrator via create_orchestrator."""
        import coral.web_ui as web_ui

        mock_bridge = MagicMock()
        mock_bridge.tools = []
        web_ui._bridge = mock_bridge
        web_ui._model = "test-model"
        web_ui._mode = "multi"

        with patch("coral.web_ui.create_orchestrator") as MockFactory:
            mock_agent = MockFactory.return_value
            mock_agent.chat = AsyncMock(return_value="Hello from multi!")

            response, agent = await _respond("hi", [], None)

        assert response == "Hello from multi!"
        assert agent is mock_agent
        MockFactory.assert_called_once_with(model="test-model", mcp_bridge=mock_bridge)
