"""Tests for PBS MCP server tools."""

from unittest.mock import patch


from coral.servers.pbs_server import (
    _validate_pbs_job_id,
    pbs_get_my_jobs,
    pbs_get_job_details,
    pbs_diagnose_job_failure,
    pbs_get_queue_status,
)


class TestValidation:
    def test_numeric_id_valid(self):
        assert _validate_pbs_job_id("12345") is None

    def test_server_suffix_valid(self):
        assert _validate_pbs_job_id("12345.svc") is None
        assert _validate_pbs_job_id("254645577.cbqs01") is None

    def test_multi_dot_valid(self):
        assert _validate_pbs_job_id("12345.svc.host") is None

    def test_injection_rejected(self):
        assert _validate_pbs_job_id("123; rm -rf /") is not None
        assert _validate_pbs_job_id("$(whoami)") is not None
        assert _validate_pbs_job_id("") is not None
        assert _validate_pbs_job_id("abc") is not None


class TestGetMyJobs:
    @patch("coral.servers.pbs_server._run")
    def test_returns_output(self, mock_run):
        mock_run.return_value = (
            "Job ID     Username  Queue  Jobname  SessID NDS TSK\n12345.svc  testuser  workq  my_job   1234    4  96\n"
        )
        result = pbs_get_my_jobs()
        assert "12345.svc" in result
        assert "my_job" in result

    @patch("coral.servers.pbs_server._run")
    def test_empty_returns_no_jobs(self, mock_run):
        mock_run.return_value = ""
        result = pbs_get_my_jobs()
        assert "No jobs" in result

    @patch("coral.servers.pbs_server._run")
    def test_state_filter(self, mock_run):
        mock_run.return_value = (
            "Job ID     Username  Queue  Jobname  SessID NDS TSK\n"
            "---\n"
            "12345.svc  testuser  workq  my_job   1234    4  96  R  01:30\n"
            "12346.svc  testuser  workq  other    --      4  96  Q  00:00\n"
        )
        result = pbs_get_my_jobs(state="running")
        assert "12345" in result


class TestGetJobDetails:
    @patch("coral.servers.pbs_server._run")
    def test_valid_id(self, mock_run):
        mock_run.return_value = "Job Id: 12345.svc\n    Job_Name = my_job\n    job_state = R"
        result = pbs_get_job_details("12345")
        assert "Job Id" in result

    def test_invalid_id(self):
        result = pbs_get_job_details("abc; rm -rf /")
        assert "Invalid" in result

    @patch("coral.servers.pbs_server._run")
    def test_pbs_format_id(self, mock_run):
        mock_run.return_value = "Job Id: 12345.svc"
        result = pbs_get_job_details("12345.svc")
        assert "Job Id" in result


class TestDiagnoseFailure:
    @patch("coral.servers.pbs_server.pbs_read_job_log")
    @patch("coral.servers.pbs_server._run")
    def test_h_c_error_detected(self, mock_run, mock_log):
        mock_run.return_value = "Job Id: 12345\n    Exit_status = 1"
        mock_log.return_value = "0: ABORT:  h_c needs to be larger:   30.0"
        result = pbs_diagnose_job_failure("12345")
        assert "h_c" in result
        assert "SCHISM" in result

    @patch("coral.servers.pbs_server.pbs_read_job_log")
    @patch("coral.servers.pbs_server._run")
    def test_oom_detected(self, mock_run, mock_log):
        mock_run.return_value = "Job Id: 12345\n    Exit_status = 137"
        mock_log.return_value = "Job killed by OOM killer"
        result = pbs_diagnose_job_failure("12345")
        assert "OUT OF MEMORY" in result

    @patch("coral.servers.pbs_server.pbs_read_job_log")
    @patch("coral.servers.pbs_server._run")
    def test_cfl_detected(self, mock_run, mock_log):
        mock_run.return_value = "Job Id: 12345"
        mock_log.return_value = "CFL violation at timestep 500"
        result = pbs_diagnose_job_failure("12345")
        assert "CFL" in result

    @patch("coral.servers.pbs_server.pbs_read_job_log")
    @patch("coral.servers.pbs_server._run")
    def test_no_pattern_found(self, mock_run, mock_log):
        mock_run.return_value = "Job Id: 12345\n    Exit_status = 0"
        mock_log.return_value = "Normal completion"
        result = pbs_diagnose_job_failure("12345")
        assert "No specific failure pattern" in result

    def test_invalid_id(self):
        result = pbs_diagnose_job_failure("bad$(id)")
        assert "Invalid" in result


class TestQueueStatus:
    @patch("coral.servers.pbs_server._run")
    def test_returns_output(self, mock_run):
        mock_run.side_effect = [
            "Queue  Max  Tot  Ena  Str\nworkq  200   50  yes  yes",
            "vnode  state  njobs\nt001   free   0",
        ]
        result = pbs_get_queue_status()
        assert "Queue Status" in result
        assert "workq" in result
