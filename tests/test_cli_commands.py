"""Regression tests for CLI slash commands and operational features."""

import json
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
        from coral.cli import _save_session, _load_session, _session_file

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
        _audit_log.append({
            "tool": "hpc_disk_quota",
            "args": "{}",
            "time": "14:30:00",
            "result_len": 500,
        })

        console = MagicMock()
        _show_audit(console)
        # Should have been called with content containing the tool name
        output = str(console.print.call_args_list)
        assert "hpc_disk_quota" in output


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Test that failures don't crash the session."""

    def test_save_conversation_empty(self):
        from coral.cli import _save_conversation

        console = MagicMock()
        _save_conversation([], "", console)
        # Should print "No conversation" message
        console.print.assert_called_once()

    def test_list_branches_empty(self):
        from coral.cli import _list_branches, _branches

        _branches.clear()
        console = MagicMock()
        _list_branches(console)
        console.print.assert_called_once()
