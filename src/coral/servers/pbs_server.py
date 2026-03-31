"""MCP server for PBS job management and log analysis (WCOSS2)."""

from __future__ import annotations

import glob
import os
import re
import subprocess
from collections import deque

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coral-pbs")

_MAX_TAIL_LINES = 1000


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a command and return stdout, or an error message."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 and result.stderr:
            return f"Command failed: {result.stderr.strip()}"
        return result.stdout if result.stdout else ""
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}. PBS may not be available on this node."
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as e:
        return f"Error: {e}"


def _validate_pbs_job_id(job_id: str) -> str | None:
    """Validate a PBS job ID. Returns None if valid, error message if not."""
    if not re.match(r"^\d+(\.\w+)*$", job_id):
        return f"Invalid PBS job ID: '{job_id}'. Expected numeric or numeric.server format."
    return None


def _read_tail(path: str, n: int) -> str:
    """Read the last N lines of a file."""
    with open(path, errors="replace") as f:
        return "".join(deque(f, maxlen=n))


@mcp.tool()
def pbs_get_my_jobs(state: str = "all") -> str:
    """Get current user's PBS jobs.

    Args:
        state: Filter: 'all', 'running', 'queued', 'held', 'completed'.
    """
    user = os.environ.get("USER", "")
    cmd = ["qstat", "-u", user]

    output = _run(cmd)
    if not output.strip() or "Command not found" in output:
        return output or "No jobs found."

    if state != "all":
        state_map = {
            "running": " R ",
            "queued": " Q ",
            "held": " H ",
            "completed": " C ",
        }
        marker = state_map.get(state, "")
        if marker:
            lines = output.split("\n")
            header = [line for line in lines[:5] if "---" in line or "Job" in line or line.strip() == ""]
            filtered = [line for line in lines if marker in line]
            return "\n".join(header + filtered) if filtered else f"No {state} jobs found."

    return output


@mcp.tool()
def pbs_get_job_details(job_id: str) -> str:
    """Get detailed information about a specific PBS job.

    Args:
        job_id: PBS job ID (e.g. '12345' or '12345.svc').
    """
    err = _validate_pbs_job_id(job_id)
    if err:
        return err
    return _run(["qstat", "-f", job_id]) or f"Job {job_id} not found."


@mcp.tool()
def pbs_read_job_log(job_id: str, tail_lines: int = 100) -> str:
    """Read the output/error log for a PBS job.

    Searches common WCOSS2 log paths for the job output.

    Args:
        job_id: PBS job ID.
        tail_lines: Number of lines from end of log (default 100, max 1000).
    """
    err = _validate_pbs_job_id(job_id)
    if err:
        return err
    tail_lines = max(1, min(tail_lines, _MAX_TAIL_LINES))

    # Extract numeric part for file matching
    numeric_id = job_id.split(".")[0]
    user = os.environ.get("USER", "")

    # Common WCOSS2 log paths
    patterns = [
        f"*.o{numeric_id}",
        f"*.e{numeric_id}",
        f"/lfs/h1/nos/ptmp/{user}/**/*.o{numeric_id}",
        f"/lfs/h1/nos/ptmp/{user}/**/*.log",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            log_path = matches[0]
            try:
                content = _read_tail(log_path, tail_lines)
                return f"Log: {log_path}\n{content}"
            except Exception as e:
                return f"Error reading {log_path}: {e}"

    return f"No log file found for job {job_id}."


@mcp.tool()
def pbs_diagnose_job_failure(job_id: str) -> str:
    """Analyze a failed PBS job: check status, resources, and error patterns.

    Args:
        job_id: PBS job ID.
    """
    err = _validate_pbs_job_id(job_id)
    if err:
        return err

    info = _run(["qstat", "-f", job_id])
    log = pbs_read_job_log(job_id, tail_lines=50)

    combined = info + "\n" + log
    diagnosis = []

    # Parse PBS-specific fields
    if "job_state = H" in info:
        diagnosis.append("JOB HELD: Job is in held state. Check qhold reason or dependency.")
    if "Exit_status = " in info:
        exit_match = re.search(r"Exit_status\s*=\s*(\S+)", info)
        if exit_match and exit_match.group(1) != "0":
            diagnosis.append(f"NON-ZERO EXIT: Exit_status = {exit_match.group(1)}")

    # Common error patterns in logs
    lowered = combined.lower()
    if "walltime" in lowered and "exceed" in lowered:
        diagnosis.append("WALLTIME EXCEEDED: Job ran past its requested wall time.")
    if "oom" in lowered or "killed" in lowered or "out of memory" in lowered:
        diagnosis.append("OUT OF MEMORY: Job exceeded memory allocation.")
    if "mpi_abort" in lowered or "mpi error" in lowered:
        diagnosis.append("MPI ERROR: MPI rank failure or communication error.")
    if "segmentation fault" in lowered or "sigsegv" in lowered:
        diagnosis.append("SEGFAULT: Memory access violation.")
    if "no such file" in lowered or "file not found" in lowered:
        diagnosis.append("MISSING FILE: Required input file not found.")
    if "h_c needs to be larger" in combined:
        diagnosis.append("SCHISM h_c ERROR: Vertical coordinate h_c too small for domain bathymetry.")
    if "cfl" in lowered or "courant" in lowered:
        diagnosis.append("CFL VIOLATION: Time step too large for grid resolution.")

    output = f"=== PBS Job {job_id} Summary ===\n{info[:2000]}\n"
    if diagnosis:
        output += "\n=== Diagnosis ===\n" + "\n".join(diagnosis) + "\n"
    else:
        output += "\n=== No specific failure pattern detected ===\n"
    output += f"\n=== Log (last 50 lines) ===\n{log}"
    return output


@mcp.tool()
def pbs_get_queue_status() -> str:
    """Show PBS queue health and node availability.

    Returns queue summary and node status.
    """
    queue_info = _run(["qstat", "-Q"])
    node_info = _run(["pbsnodes", "-aSj"])

    output = "=== Queue Status ===\n" + queue_info
    if node_info and "Command not found" not in node_info:
        output += "\n=== Node Status ===\n" + node_info[:3000]
    return output


if __name__ == "__main__":
    mcp.run()
