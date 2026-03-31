"""Tests for the viz (code execution) MCP server tool functions."""

import os
from unittest.mock import patch

import coral.servers.viz_server as viz_server
from coral.servers.viz_server import execute_python


class TestExecutePython:
    def test_simple_print(self):
        result = execute_python('print("hello coral")')
        assert "hello coral" in result

    def test_numpy_available(self):
        result = execute_python("print(np.array([1,2,3]).sum())")
        assert "6" in result

    def test_pandas_available(self):
        result = execute_python('import pandas as pd; print(pd.DataFrame({"a": [1,2]}).shape)')
        assert "(2, 1)" in result

    def test_matplotlib_import(self):
        result = execute_python("print(matplotlib.get_backend())")
        assert "Agg" in result

    def test_syntax_error(self):
        result = execute_python("def foo(:\n  pass")
        assert "failed" in result.lower() or "SyntaxError" in result

    def test_runtime_error(self):
        result = execute_python("x = 1/0")
        assert "failed" in result.lower() or "ZeroDivisionError" in result

    def test_plot_saves_file(self):
        plot_path = "/tmp/coral_plot.png"
        # Remove existing plot
        if os.path.exists(plot_path):
            os.unlink(plot_path)

        result = execute_python("""\
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Test Plot")
plt.savefig("/tmp/coral_plot.png")
""")
        assert "Plot saved" in result
        assert os.path.exists(plot_path)
        # Cleanup
        os.unlink(plot_path)

    def test_no_output(self):
        result = execute_python("x = 42")
        assert "Script completed successfully" in result

    def test_output_truncated(self):
        result = execute_python('print("x" * 5000)')
        # Should truncate to 3000 chars
        assert len(result) < 4000

    def test_requires_sandbox_refuses_host_execution(self, monkeypatch):
        monkeypatch.setenv("CORAL_REQUIRE_SANDBOX", "1")

        with (
            patch.object(viz_server, "_resolve_sandbox_sif", return_value=""),
            patch("coral.servers.viz_server.shutil.which", return_value=None),
            patch("coral.servers.viz_server.subprocess.run") as mock_run,
        ):
            result = execute_python('print("hello")')

        assert "Sandboxed Python execution is required" in result
        mock_run.assert_not_called()

    def test_sandbox_runtime_failure_falls_back_to_host_python(self, monkeypatch):
        monkeypatch.delenv("CORAL_REQUIRE_SANDBOX", raising=False)

        sandbox_failure = viz_server.subprocess.CompletedProcess(
            args=["apptainer"],
            returncode=1,
            stdout="",
            stderr="ERROR  : Could not write info to setgroups: Permission denied\n"
            "ERROR  : Error while waiting event for user namespace mappings: no event received\n",
        )
        host_success = viz_server.subprocess.CompletedProcess(
            args=["python3"],
            returncode=0,
            stdout="hello from host\n",
            stderr="",
        )

        with (
            patch.object(viz_server, "_resolve_sandbox_sif", return_value="/tmp/fake.sif"),
            patch("coral.servers.viz_server.shutil.which", return_value="/usr/bin/apptainer"),
            patch("coral.servers.viz_server.subprocess.run", side_effect=[sandbox_failure, host_success]) as mock_run,
        ):
            result = execute_python('print("hello from host")')

        assert "hello from host" in result
        assert mock_run.call_count == 2
