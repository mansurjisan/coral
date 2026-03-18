"""MCP server for executing Python analysis scripts and generating plots."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from mcp.server.fastmcp import FastMCP

from coral.audit import with_tool_audit_payload

mcp = FastMCP("coral-viz")

_SCRIPT_HEADER = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
"""

_SCRIPT_TIMEOUT = int(os.environ.get("CORAL_SCRIPT_TIMEOUT", "120"))


def _plot_dir() -> str:
    """Return the directory for saving plots — prefer scratch over /tmp."""
    user = os.environ.get("USER", "")
    scratch = f"/scratch5/purged/{user}/CORAL"
    if os.path.isdir(scratch):
        return scratch
    return tempfile.gettempdir()

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
        "mount source",
        "hook function failure",
        "operation not permitted",
        "getsockopt",
        "socket communication error",
        "permission denied",
        "failed to create",
        "container creation failed",
        "overlay",
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
    plot_dir = _plot_dir()
    plot_path = os.path.join(plot_dir, "coral_plot.png")

    # Clean up previous plot before running new code
    if os.path.exists(plot_path):
        try:
            os.unlink(plot_path)
        except OSError:
            pass

    # Also clean /tmp/coral_plot.png if using scratch for plots
    if plot_dir != tempfile.gettempdir() and os.path.exists("/tmp/coral_plot.png"):
        try:
            os.unlink("/tmp/coral_plot.png")
        except OSError:
            pass

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(_SCRIPT_HEADER + code)
        script_path = f.name

    try:
        cmd, error, using_sandbox = _build_execution_command(script_path)
        if error is not None:
            return with_tool_audit_payload(error, sandbox_used=False)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )

        # Sandbox fallback: retry on host if Apptainer had an environment issue
        if (
            using_sandbox
            and result.returncode != 0
            and not _sandbox_required()
            and _is_sandbox_runtime_failure(result.stderr)
        ):
            using_sandbox = False
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=_SCRIPT_TIMEOUT,
                cwd=tempfile.gettempdir(),
            )

        output = ""
        if result.stdout:
            stdout_text = result.stdout[:3000]
            if len(result.stdout) > 3000:
                stdout_text += "\n... (output truncated)"
            output += stdout_text

        if result.stderr:
            # Filter out common warnings that clutter output
            stderr_lines = [
                line for line in result.stderr.split("\n")
                if line.strip()
                and "UserWarning" not in line
                and "FutureWarning" not in line
                and "DeprecationWarning" not in line
                and "RuntimeWarning" not in line
            ]
            if stderr_lines:
                output += "\nSTDERR:\n" + "\n".join(stderr_lines[:30])

        if result.returncode != 0:
            # Classify the error for better user feedback
            error_type = _classify_error(result.stderr)
            output = f"Script failed ({error_type}, exit code {result.returncode}):\n{output}"
        else:
            # Check both /tmp and scratch for plot output
            found_plot = None
            for candidate in [plot_path, "/tmp/coral_plot.png"]:
                if os.path.exists(candidate):
                    found_plot = candidate
                    break
            if found_plot:
                size_kb = os.path.getsize(found_plot) // 1024
                output += f"\n[Plot saved to {found_plot} ({size_kb} KB)]"

        final_output = output.strip() if output.strip() else "Script completed successfully (no output)."
        return with_tool_audit_payload(final_output, sandbox_used=using_sandbox)

    except subprocess.TimeoutExpired:
        return with_tool_audit_payload(
            f"Script timed out after {_SCRIPT_TIMEOUT} seconds. "
            f"Consider simplifying the computation or using a smaller dataset.",
            sandbox_used=False,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _classify_error(stderr: str) -> str:
    """Classify a script error for user-friendly feedback."""
    lowered = stderr.lower()
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "missing library"
    if "syntaxerror" in lowered:
        return "syntax error"
    if "memoryerror" in lowered or "killed" in lowered:
        return "out of memory"
    if "connectionerror" in lowered or "urlopen" in lowered:
        return "network error"
    if "filenotfounderror" in lowered:
        return "file not found"
    if "permissionerror" in lowered:
        return "permission denied"
    return "runtime error"


if __name__ == "__main__":
    mcp.run()
