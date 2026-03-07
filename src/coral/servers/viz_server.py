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

def _sandbox_candidates() -> list[str]:
    """Return candidate sandbox image paths in priority order."""
    configured = os.environ.get("CORAL_SANDBOX_SIF", "")
    candidates = []
    if configured:
        candidates.append(configured)

    candidates.extend([
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "containers", "coral_sandbox.sif"),
        f"/scratch5/purged/{os.environ.get('USER', '')}/coral_sandbox.sif",
    ])
    return candidates


def _resolve_sandbox_sif() -> str:
    """Find the first sandbox image that exists on disk."""
    for candidate in _sandbox_candidates():
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


def _sandbox_required() -> bool:
    """Whether host-side execution is forbidden in this environment."""
    return os.environ.get("CORAL_REQUIRE_SANDBOX", "").lower() in {"1", "true", "yes", "on"}


def _is_sandbox_runtime_failure(stderr: str) -> bool:
    """Detect environment-level Apptainer failures where host fallback is acceptable."""
    lowered = stderr.lower()
    markers = [
        "could not write info to setgroups",
        "user namespace",
        "no event received",
    ]
    return any(marker in lowered for marker in markers)


def _build_execution_command(script_path: str) -> tuple[list[str] | None, str | None, bool]:
    """Return the command to execute, enforcement error, and sandbox usage flag."""
    sandbox_sif = _resolve_sandbox_sif()
    apptainer = shutil.which("apptainer")

    if sandbox_sif and apptainer:
        return [
            apptainer,
            "exec",
            "--nv",
            "--bind",
            "/tmp:/tmp",
            "--bind",
            "/scratch:/scratch",
            sandbox_sif,
            "python3",
            script_path,
        ], None, True

    if _sandbox_required():
        return None, (
            "Sandboxed Python execution is required, but Apptainer or the sandbox image "
            "is unavailable. Build `containers/coral_sandbox.sif` or set "
            "`CORAL_SANDBOX_SIF` to a valid image before using execute_python."
        ), False

    return [sys.executable, script_path], None, False


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
        cmd, error, using_sandbox = _build_execution_command(script_path)
        if error is not None:
            return error

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
        if (
            using_sandbox
            and result.returncode != 0
            and not _sandbox_required()
            and _is_sandbox_runtime_failure(result.stderr)
        ):
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
