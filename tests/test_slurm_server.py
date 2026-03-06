"""Tests for the Slurm MCP server tool functions.

Since Slurm commands (sacct, scontrol) aren't available in dev environments,
these tests focus on:
- Input validation
- Log reading from fixtures
- Diagnosis pattern matching
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from coral.servers.slurm_server import (
    diagnose_job_failure,
    get_job_details,
    get_my_jobs,
    read_job_log,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestGetMyJobs:
    def test_invalid_state(self):
        result = get_my_jobs(state="bogus")
        assert "Invalid state" in result

    @patch("coral.servers.slurm_server._run")
    def test_valid_state_calls_sacct(self, mock_run):
        mock_run.return_value = "12345  stofs_run  COMPLETED  0:0  02:15:13"
        result = get_my_jobs(state="completed")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "sacct" in cmd
        assert "-t" in cmd
        assert "COMPLETED" in cmd
        assert "12345" in result

    @patch("coral.servers.slurm_server._run")
    def test_all_state_no_filter(self, mock_run):
        mock_run.return_value = "some output"
        get_my_jobs(state="all")
        cmd = mock_run.call_args[0][0]
        assert "-t" not in cmd

    @patch("coral.servers.slurm_server._run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = ""
        result = get_my_jobs()
        assert "No jobs found" in result


class TestGetJobDetails:
    def test_invalid_job_id(self):
        result = get_job_details("abc")
        assert "Invalid job ID" in result

    def test_injection_blocked(self):
        result = get_job_details("123; rm -rf /")
        assert "Invalid job ID" in result

    @patch("coral.servers.slurm_server._run")
    def test_valid_job_id(self, mock_run):
        mock_run.return_value = "JobId=12345 JobName=stofs_run"
        result = get_job_details("12345")
        assert "12345" in result


class TestReadJobLog:
    def test_invalid_job_id(self):
        result = read_job_log("abc")
        assert "Invalid job ID" in result

    def test_finds_log_file(self):
        # Create a temp slurm log file
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "slurm-99999.out")
            with open(log_path, "w") as f:
                f.write("line 1\nline 2\nline 3\n")

            # Temporarily change to tmpdir so glob finds it
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = read_job_log("99999")
                assert "Log:" in result
                assert "line 3" in result
            finally:
                os.chdir(old_cwd)

    def test_no_log_found(self):
        result = read_job_log("00000")
        assert "No log file found" in result

    def test_tail_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "slurm-88888.out")
            with open(log_path, "w") as f:
                for i in range(200):
                    f.write(f"log line {i}\n")

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = read_job_log("88888", tail_lines=5)
                assert "log line 199" in result
                assert "log line 0" not in result
            finally:
                os.chdir(old_cwd)


class TestDiagnoseJobFailure:
    def test_invalid_job_id(self):
        result = diagnose_job_failure("abc")
        assert "Invalid job ID" in result

    @patch("coral.servers.slurm_server._run")
    def test_timeout_detection(self, mock_run):
        mock_run.return_value = "12345  TIMEOUT  0:0  08:00:00"
        result = diagnose_job_failure("12345")
        assert "TIMED OUT" in result

    @patch("coral.servers.slurm_server._run")
    def test_oom_detection(self, mock_run):
        mock_run.return_value = "12345  OUT_OF_ME+  137:0"
        result = diagnose_job_failure("12345")
        assert "OUT OF MEMORY" in result

    @patch("coral.servers.slurm_server.read_job_log")
    @patch("coral.servers.slurm_server._run")
    def test_mpi_abort_detection(self, mock_run, mock_log):
        mock_run.return_value = "12345  FAILED  1:0"
        mock_log.return_value = "MPI_ABORT was invoked on rank 42\nProcess killed"
        result = diagnose_job_failure("12345")
        assert "MPI ERROR" in result

    @patch("coral.servers.slurm_server.read_job_log")
    @patch("coral.servers.slurm_server._run")
    def test_signal9_detection(self, mock_run, mock_log):
        mock_run.return_value = "12345  FAILED  137:0"
        mock_log.return_value = "KILLED BY SIGNAL 9 (Killed)"
        result = diagnose_job_failure("12345")
        assert "SIGNAL 9" in result

    @patch("coral.servers.slurm_server.read_job_log")
    @patch("coral.servers.slurm_server._run")
    def test_segfault_detection(self, mock_run, mock_log):
        mock_run.return_value = "12345  FAILED  139:0"
        mock_log.return_value = "Segmentation fault (core dumped)"
        result = diagnose_job_failure("12345")
        assert "SEGFAULT" in result

    @patch("coral.servers.slurm_server.read_job_log")
    @patch("coral.servers.slurm_server._run")
    def test_missing_file_detection(self, mock_run, mock_log):
        mock_run.return_value = "12345  FAILED  1:0"
        mock_log.return_value = "Error: No such file or directory: /scratch/input.nc"
        result = diagnose_job_failure("12345")
        assert "MISSING FILE" in result
