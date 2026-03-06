"""MCP server for executing Python analysis scripts and generating plots."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coral-viz")

_SCRIPT_HEADER = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""

_SCRIPT_TIMEOUT = 120


@mcp.tool()
def execute_python(code: str, description: str = "") -> str:
    """Execute a Python script for data analysis or visualization.

    The script has access to: xarray, netCDF4, matplotlib, cartopy, numpy, pandas, f90nml.
    If the script creates a figure, save it to '/tmp/coral_plot.png'.

    Args:
        code: Python code to execute.
        description: Brief description of what the code does.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(_SCRIPT_HEADER + code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )

        output = ""
        if result.stdout:
            output += result.stdout[:3000]
        if result.stderr:
            # Filter out common matplotlib/numpy warnings
            stderr_lines = [
                line for line in result.stderr.split("\n")
                if line.strip() and "UserWarning" not in line and "FutureWarning" not in line
            ]
            if stderr_lines:
                output += "\nSTDERR:\n" + "\n".join(stderr_lines[:50])

        if result.returncode != 0:
            output = f"Script failed (exit code {result.returncode}):\n{output}"
        else:
            plot_path = "/tmp/coral_plot.png"
            if os.path.exists(plot_path):
                size_kb = os.path.getsize(plot_path) // 1024
                output += f"\n[Plot saved to {plot_path} ({size_kb} KB)]"

        return output.strip() if output.strip() else "Script completed successfully (no output)."

    except subprocess.TimeoutExpired:
        return f"Script timed out after {_SCRIPT_TIMEOUT} seconds."
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


if __name__ == "__main__":
    mcp.run()
