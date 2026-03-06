"""Tests for the viz (code execution) MCP server tool functions."""

import os

from coral.servers.viz_server import execute_python


class TestExecutePython:
    def test_simple_print(self):
        result = execute_python('print("hello coral")')
        assert "hello coral" in result

    def test_numpy_available(self):
        result = execute_python('print(np.array([1,2,3]).sum())')
        assert "6" in result

    def test_pandas_available(self):
        result = execute_python(
            'import pandas as pd; print(pd.DataFrame({"a": [1,2]}).shape)'
        )
        assert "(2, 1)" in result

    def test_matplotlib_import(self):
        result = execute_python(
            'print(matplotlib.get_backend())'
        )
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
