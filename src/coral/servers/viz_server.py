"""MCP server for executing Python analysis scripts and generating plots."""

from __future__ import annotations

import os
import shutil
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

# Auto-detect Apptainer sandbox image (built from containers/coral_sandbox.def)
_SANDBOX_SIF = os.environ.get("CORAL_SANDBOX_SIF", "")
if not _SANDBOX_SIF:
    # Check common locations
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "containers", "coral_sandbox.sif"),
        "/scratch5/purged/{}/coral_sandbox.sif".format(os.environ.get("USER", "")),
    ]:
        if os.path.isfile(candidate):
            _SANDBOX_SIF = candidate
            break

_USE_SANDBOX = bool(_SANDBOX_SIF and shutil.which("apptainer"))


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
        if _USE_SANDBOX:
            cmd = [
                "apptainer", "exec",
                "--nv",
                "--bind", "/tmp:/tmp",
                "--bind", "/scratch:/scratch",
                _SANDBOX_SIF,
                "python3", script_path,
            ]
        else:
            cmd = [sys.executable, script_path]

        result = subprocess.run(
            cmd,
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
