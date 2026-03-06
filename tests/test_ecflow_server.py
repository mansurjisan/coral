"""Tests for the ecFlow MCP server tool functions.

ecflow_client is not available in dev environments, so tests mock subprocess.
"""

from unittest.mock import patch

from coral.servers.ecflow_server import (
    get_aborted_tasks,
    get_suite_status,
    read_ecflow_job_output,
)

_SAMPLE_STATE = """\
suite stofs
  family forecast
    task prep_forcing state:complete
    task run_model state:active
    task post_process state:queued
  endfamily
  family analysis
    task validate state:aborted
    task archive state:queued
  endfamily
endsuite
"""


class TestGetSuiteStatus:
    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_returns_state(self, mock_run):
        mock_run.return_value = _SAMPLE_STATE
        result = get_suite_status("stofs")
        assert "stofs" in result
        assert "run_model" in result
        mock_run.assert_called_once_with(["--get_state", "/stofs"])

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_empty_suite_name_gets_all(self, mock_run):
        mock_run.return_value = _SAMPLE_STATE
        get_suite_status("")
        mock_run.assert_called_once_with(["--get_state"])

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_truncates_long_output(self, mock_run):
        mock_run.return_value = "x" * 6000
        result = get_suite_status("big")
        assert len(result) < 5200
        assert "truncated" in result

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_empty_output(self, mock_run):
        mock_run.return_value = ""
        result = get_suite_status("empty")
        assert "No suite found" in result


class TestGetAbortedTasks:
    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_finds_aborted(self, mock_run):
        mock_run.return_value = _SAMPLE_STATE
        result = get_aborted_tasks("stofs")
        assert "1 aborted" in result
        assert "validate" in result

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_no_aborted(self, mock_run):
        mock_run.return_value = "task a state:complete\ntask b state:active\n"
        result = get_aborted_tasks("clean_suite")
        assert "No aborted tasks" in result

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_ecflow_error(self, mock_run):
        mock_run.return_value = "ecflow_client not found."
        result = get_aborted_tasks("stofs")
        assert "ecflow_client" in result


class TestReadEcflowJobOutput:
    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_returns_output(self, mock_run):
        mock_run.return_value = "Job output line 1\nJob output line 2\n"
        result = read_ecflow_job_output("/stofs/forecast/run_model")
        assert "Job output line 1" in result

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_empty_output(self, mock_run):
        mock_run.return_value = ""
        result = read_ecflow_job_output("/stofs/forecast/run_model")
        assert "No output found" in result

    @patch("coral.servers.ecflow_server._run_ecflow")
    def test_truncates_large_output(self, mock_run):
        mock_run.return_value = "line\n" * 2000
        result = read_ecflow_job_output("/stofs/forecast/run_model")
        assert "truncated" in result
