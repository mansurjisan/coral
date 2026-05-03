"""Regression tests for CLI slash commands and operational features."""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# /watch validation
# ---------------------------------------------------------------------------


class TestWatchValidation:
    """Test /watch job ID acceptance."""

    def test_slurm_numeric_id_accepted(self):
        assert re.match(r"^[\d]+(\.\w+)*$", "9848988")

    def test_pbs_id_with_server_accepted(self):
        assert re.match(r"^[\d]+(\.\w+)*$", "12345.svc")
        assert re.match(r"^[\d]+(\.\w+)*$", "254645577.cbqs01")

    def test_pbs_id_multi_dot_accepted(self):
        assert re.match(r"^[\d]+(\.\w+)*$", "12345.svc.host")

    def test_injection_rejected(self):
        assert not re.match(r"^[\d]+(\.\w+)*$", "123; rm -rf /")
        assert not re.match(r"^[\d]+(\.\w+)*$", "$(whoami)")
        assert not re.match(r"^[\d]+(\.\w+)*$", "")
        assert not re.match(r"^[\d]+(\.\w+)*$", "abc")


# ---------------------------------------------------------------------------
# /branch
# ---------------------------------------------------------------------------


class TestBranch:
    """Test conversation branching."""

    def test_branch_clears_chat_log(self):
        from coral.cli import _branch_conversation, _branches

        chat_log = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent = MagicMock()
        agent.reset = MagicMock()
        console = MagicMock()

        _branch_conversation("test_branch", chat_log, agent, console)

        assert len(chat_log) == 0
        assert "test_branch" in _branches
        assert len(_branches["test_branch"]) == 2

    def test_branch_resets_agent(self):
        from coral.cli import _branch_conversation

        chat_log = [{"role": "user", "content": "x"}]
        agent = MagicMock()
        agent.reset = MagicMock()
        console = MagicMock()

        _branch_conversation("reset_test", chat_log, agent, console)

        agent.reset.assert_called_once()

    def test_branch_auto_names(self):
        from coral.cli import _branch_conversation, _branches

        chat_log = [{"role": "user", "content": "x"}]
        agent = MagicMock()
        console = MagicMock()

        initial_count = len(_branches)
        _branch_conversation("", chat_log, agent, console)

        # Should have created a branch with auto-generated name
        assert len(_branches) == initial_count + 1


# ---------------------------------------------------------------------------
# /status scheduler detection
# ---------------------------------------------------------------------------


class TestStatusSchedulerDetection:
    """Test that /status detects the right scheduler."""

    def test_prefers_slurm_when_available(self):
        import shutil

        # This test verifies the logic, not actual command execution
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/squeue" if cmd == "squeue" else None
            assert shutil.which("squeue") is not None

    def test_falls_back_to_pbs(self):
        import shutil

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/qstat" if cmd == "qstat" else None
            assert shutil.which("squeue") is None
            assert shutil.which("qstat") is not None


# ---------------------------------------------------------------------------
# /alert input validation
# ---------------------------------------------------------------------------


class TestAlertValidation:
    """Test alert input parsing."""

    def test_valid_alert_args(self):
        parts = "8518750 > 1.5".split()
        assert len(parts) == 3
        assert parts[1] in (">", "<", ">=", "<=")
        assert float(parts[2]) == 1.5

    def test_invalid_operator_detected(self):
        parts = "8518750 == 1.5".split()
        assert parts[1] not in (">", "<", ">=", "<=")

    def test_invalid_threshold_detected(self):
        parts = "8518750 > abc".split()
        with pytest.raises(ValueError):
            float(parts[2])


# ---------------------------------------------------------------------------
# Session save/load
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    """Test session save and load."""

    def test_save_and_load(self, tmp_path):
        from coral.cli import _save_session, _load_session

        chat_log = [
            {"role": "user", "content": "What is SCHISM?"},
            {"role": "assistant", "content": "SCHISM is a model."},
        ]

        # Override session file location
        with patch("coral.cli._session_file", return_value=tmp_path / "session.json"):
            _save_session(chat_log)

            loaded = _load_session()
            assert len(loaded) == 2
            assert loaded[0]["content"] == "What is SCHISM?"

    def test_load_empty_returns_empty(self, tmp_path):
        from coral.cli import _load_session

        with patch("coral.cli._session_file", return_value=tmp_path / "nonexistent.json"):
            loaded = _load_session()
            assert loaded == []

    def test_load_corrupted_returns_empty(self, tmp_path):
        from coral.cli import _load_session

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")

        with patch("coral.cli._session_file", return_value=bad_file):
            loaded = _load_session()
            assert loaded == []


# ---------------------------------------------------------------------------
# /audit
# ---------------------------------------------------------------------------


class TestAudit:
    """Test audit log display."""

    def test_audit_empty(self):
        from coral.cli import _show_audit, _audit_log

        _audit_log.clear()
        console = MagicMock()
        _show_audit(console)
        console.print.assert_called_once()

    def test_audit_with_entries(self):
        from coral.cli import _show_audit, _audit_log

        _audit_log.clear()
        _audit_log.append(
            {
                "tool": "hpc_disk_quota",
                "args": "{}",
                "time": "14:30:00",
                "result_len": 500,
            }
        )

        console = MagicMock()
        _show_audit(console)
        # Should have been called with content containing the tool name
        output = str(console.print.call_args_list)
        assert "hpc_disk_quota" in output


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestAlertMCP:
    """Test the MCP-mediated /alert flow."""

    @pytest.mark.asyncio
    async def test_alert_create_calls_mcp_tool(self):
        from coral.cli import _set_alert_via_mcp

        agent = MagicMock()
        bridge = MagicMock()
        bridge.tool_map = {"coral_create_alert": ("session", "alerts")}
        bridge.call_tool = AsyncMock(return_value="## Alert Created\n\n| ID | `a1` |")
        agent.mcp_bridge = bridge
        console = MagicMock()

        await _set_alert_via_mcp("8518750 > 1.5", agent, console)

        bridge.call_tool.assert_called_once_with(
            "coral_create_alert",
            {
                "station_id": "8518750",
                "operator": ">",
                "threshold": 1.5,
            },
        )

    @pytest.mark.asyncio
    async def test_alert_list_subcommand(self):
        from coral.cli import _set_alert_via_mcp

        agent = MagicMock()
        bridge = MagicMock()
        bridge.tool_map = {"coral_list_alerts": ("session", "alerts")}
        bridge.call_tool = AsyncMock(return_value="No alerts configured.")
        agent.mcp_bridge = bridge
        console = MagicMock()

        await _set_alert_via_mcp("list", agent, console)

        bridge.call_tool.assert_called_once_with("coral_list_alerts", {})

    @pytest.mark.asyncio
    async def test_alert_check_subcommand(self):
        from coral.cli import _set_alert_via_mcp

        agent = MagicMock()
        bridge = MagicMock()
        bridge.tool_map = {"coral_check_alerts": ("session", "alerts")}
        bridge.call_tool = AsyncMock(return_value="No active alerts to check.")
        agent.mcp_bridge = bridge
        console = MagicMock()

        await _set_alert_via_mcp("check", agent, console)

        bridge.call_tool.assert_called_once_with("coral_check_alerts", {})

    @pytest.mark.asyncio
    async def test_alert_no_server_shows_message(self):
        from coral.cli import _set_alert_via_mcp

        agent = MagicMock()
        bridge = MagicMock()
        bridge.tool_map = {}  # No alert tools
        agent.mcp_bridge = bridge
        console = MagicMock()

        await _set_alert_via_mcp("8518750 > 1.5", agent, console)

        output = str(console.print.call_args_list)
        assert "not available" in output

    @pytest.mark.asyncio
    async def test_alert_invalid_operator(self):
        from coral.cli import _set_alert_via_mcp

        console = MagicMock()
        await _set_alert_via_mcp("8518750 == 1.5", MagicMock(), console)

        output = str(console.print.call_args_list)
        assert "Invalid operator" in output

    @pytest.mark.asyncio
    async def test_alert_invalid_threshold(self):
        from coral.cli import _set_alert_via_mcp

        console = MagicMock()
        await _set_alert_via_mcp("8518750 > abc", MagicMock(), console)

        output = str(console.print.call_args_list)
        assert "Invalid threshold" in output

    @pytest.mark.asyncio
    async def test_alert_too_few_args_shows_help(self):
        from coral.cli import _set_alert_via_mcp

        console = MagicMock()
        await _set_alert_via_mcp("", MagicMock(), console)

        output = str(console.print.call_args_list)
        assert "Usage" in output


class TestGetMCPBridge:
    """Test bridge extraction from agent/orchestrator."""

    def test_from_single_agent(self):
        from coral.cli import _get_mcp_bridge

        agent = MagicMock()
        agent.mcp_bridge = MagicMock()
        assert _get_mcp_bridge(agent) is agent.mcp_bridge

    def test_from_orchestrator(self):
        from coral.cli import _get_mcp_bridge

        agent = MagicMock()
        del agent.mcp_bridge  # Orchestrator doesn't have direct bridge
        sub_agent = MagicMock()
        sub_agent.mcp_bridge = MagicMock()
        agent.agents = {"DATA": sub_agent}
        assert _get_mcp_bridge(agent) is sub_agent.mcp_bridge

    def test_returns_none_when_unavailable(self):
        from coral.cli import _get_mcp_bridge

        agent = MagicMock(spec=[])  # No attributes
        assert _get_mcp_bridge(agent) is None


class TestGracefulDegradation:
    """Test that failures don't crash the session."""

    def test_save_conversation_empty(self):
        from coral.cli import _save_conversation

        console = MagicMock()
        _save_conversation([], "", console)
        console.print.assert_called_once()

    def test_list_branches_empty(self):
        from coral.cli import _list_branches, _branches

        _branches.clear()
        console = MagicMock()
        _list_branches(console)
        console.print.assert_called_once()


# ---------------------------------------------------------------------------
# /route slash command
# ---------------------------------------------------------------------------


class TestShowRoute:
    def test_no_decision_yet(self):
        from coral.cli import _show_route

        agent = MagicMock()
        agent.last_route_decision = None
        agent.agents = {"DATA": MagicMock()}  # multi-agent, just no query yet
        console = MagicMock()

        _show_route(agent, console)

        output = str(console.print.call_args_list)
        assert "No query routed yet" in output or "ask a question" in output

    def test_single_mode_warns(self):
        from coral.cli import _show_route

        agent = MagicMock(spec=["last_route_decision"])
        agent.last_route_decision = None
        console = MagicMock()

        _show_route(agent, console)

        output = str(console.print.call_args_list)
        assert "multi-agent" in output

    def test_keyword_decision_rendered(self):
        from coral.cli import _show_route

        agent = MagicMock()
        agent.last_route_decision = {
            "categories": ["DATA"],
            "method": "keyword",
            "confidence": 0.85,
            "matched_keywords": ["water level", "tide"],
            "query": "What is the water level at The Battery?",
        }
        console = MagicMock()

        _show_route(agent, console)

        output = str(console.print.call_args_list)
        assert "DATA" in output
        assert "keyword" in output
        assert "0.85" in output
        assert "water level" in output

    def test_llm_decision_shows_router_model(self):
        from coral.cli import _show_route

        agent = MagicMock()
        agent.last_route_decision = {
            "categories": ["WORKFLOW"],
            "method": "llm",
            "confidence": 0.7,
            "matched_keywords": [],
            "query": "Hello there",
            "router_model": "qwen3:8b",
            "raw_response": "WORKFLOW",
        }
        console = MagicMock()

        _show_route(agent, console)

        output = str(console.print.call_args_list)
        assert "WORKFLOW" in output
        assert "qwen3:8b" in output


# ---------------------------------------------------------------------------
# coral audit (--since parsing + filtering)
# ---------------------------------------------------------------------------


class TestParseSince:
    def test_relative_hours(self):
        from datetime import datetime, timezone
        from coral.cli import _parse_since

        result = _parse_since("2h")
        assert result is not None
        delta = datetime.now(timezone.utc) - result
        assert 7100 < delta.total_seconds() < 7300  # ~2h

    def test_relative_minutes(self):
        from coral.cli import _parse_since

        assert _parse_since("30m") is not None

    def test_relative_days(self):
        from coral.cli import _parse_since

        assert _parse_since("1d") is not None

    def test_iso_timestamp(self):
        from coral.cli import _parse_since

        result = _parse_since("2026-04-01T00:00:00")
        assert result is not None
        assert result.year == 2026

    def test_garbage_returns_none(self):
        from coral.cli import _parse_since

        assert _parse_since("not a time") is None
        assert _parse_since("") is None


class TestAuditCommand:
    """Tests for the `coral audit` Typer command using a fixture JSONL file."""

    @pytest.fixture
    def fixture_log(self, tmp_path, monkeypatch):
        import json

        path = tmp_path / "coral_audit.jsonl"
        entries = [
            {
                "timestamp": "2026-05-01T10:00:00+00:00",
                "event": "query_start",
                "query_id": "abc123",
                "mode": "multi",
                "routed_sections": ["DATA"],
                "section": None,
            },
            {
                "timestamp": "2026-05-01T10:00:05+00:00",
                "event": "tool_call",
                "query_id": "abc123",
                "mode": "multi",
                "routed_sections": ["DATA"],
                "section": "DATA",
                "tool": "coops_get_water_levels",
                "args_summary": '{"station": "8518750"}',
                "duration_ms": 250,
                "success": True,
            },
            {
                "timestamp": "2026-05-01T11:00:00+00:00",
                "event": "tool_call",
                "query_id": "def456",
                "mode": "multi",
                "routed_sections": ["WORKFLOW"],
                "section": "WORKFLOW",
                "tool": "slurm_squeue",
                "args_summary": "{}",
                "duration_ms": 100,
                "success": True,
            },
        ]
        with path.open("w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        monkeypatch.setenv("CORAL_AUDIT_LOG", str(path))
        return path

    def test_filter_by_query_id(self, fixture_log):
        from typer.testing import CliRunner

        from coral.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["audit", "--query-id", "abc123"])
        assert result.exit_code == 0
        assert "coops_get_water_levels" in result.stdout
        assert "slurm_squeue" not in result.stdout

    def test_filter_by_tool_substring(self, fixture_log):
        from typer.testing import CliRunner

        from coral.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["audit", "--tool", "coops"])
        assert result.exit_code == 0
        assert "coops_get_water_levels" in result.stdout
        assert "slurm_squeue" not in result.stdout

    def test_filter_by_section(self, fixture_log):
        from typer.testing import CliRunner

        from coral.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["audit", "--section", "WORKFLOW"])
        assert result.exit_code == 0
        assert "slurm_squeue" in result.stdout
        assert "coops_get_water_levels" not in result.stdout

    def test_format_json(self, fixture_log):
        from typer.testing import CliRunner

        from coral.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["audit", "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON containing all 3 entries
        # (rich's print_json writes to stdout)
        assert "abc123" in result.stdout
        assert "def456" in result.stdout

    def test_no_log_file(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from coral.cli import app

        # Point at a non-existent path
        monkeypatch.setenv("CORAL_AUDIT_LOG", str(tmp_path / "missing.jsonl"))
        runner = CliRunner()
        result = runner.invoke(app, ["audit"])
        assert result.exit_code == 0
        assert "No audit log" in result.stdout

    def test_limit_truncates(self, fixture_log):
        from typer.testing import CliRunner

        from coral.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["audit", "--limit", "1"])
        assert result.exit_code == 0
        # Most recent entry only
        assert "slurm_squeue" in result.stdout
        assert "coops_get_water_levels" not in result.stdout
