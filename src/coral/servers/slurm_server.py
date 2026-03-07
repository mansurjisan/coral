"""MCP server for Slurm job management and log analysis."""

from __future__ import annotations

from collections import deque
import glob
import re
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coral-slurm")

_SACCT_FORMAT = "JobID,JobName%30,State,ExitCode,Elapsed,Start,End,MaxRSS,NodeList"
_ALLOWED_STATES = {"all", "running", "pending", "failed", "completed"}
_MAX_TAIL_LINES = 1000


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a command and return stdout, or an error message."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 and result.stderr:
            return f"Command failed: {result.stderr.strip()}"
        return result.stdout if result.stdout else ""
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}. Slurm may not be available on this node."
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as e:
        return f"Error: {e}"


def _normalize_tail_lines(tail_lines: int) -> int | None:
    """Validate and cap requested tail length."""
    if not isinstance(tail_lines, int) or tail_lines < 1:
        return None
    return min(tail_lines, _MAX_TAIL_LINES)


def _read_tail(log_path: str, tail_lines: int) -> str:
    """Read only the last N lines of a log file without loading the whole file."""
    with open(log_path, errors="replace") as f:
        return "".join(deque(f, maxlen=tail_lines))


@mcp.tool()
def get_my_jobs(state: str = "all") -> str:
    """Get current user's Slurm jobs.

    Args:
        state: Job state filter: 'all', 'running', 'pending', 'failed', 'completed'
    """
    if state not in _ALLOWED_STATES:
        return f"Invalid state '{state}'. Use one of: {', '.join(sorted(_ALLOWED_STATES))}"

    cmd = ["sacct", "-X", f"--format={_SACCT_FORMAT}", "-n"]
    if state != "all":
        cmd.extend(["-t", state.upper()])

    output = _run(cmd)
    return output if output.strip() else "No jobs found."


@mcp.tool()
def get_job_details(job_id: str) -> str:
    """Get detailed information about a specific Slurm job.

    Args:
        job_id: Slurm job ID.
    """
    if not re.match(r"^\d+$", job_id):
        return f"Invalid job ID: {job_id}"

    return _run(["scontrol", "show", "job", job_id]) or f"Job {job_id} not found."


@mcp.tool()
def read_job_log(job_id: str, tail_lines: int = 100) -> str:
    """Read the output/error log for a Slurm job.

    Searches for common log patterns: slurm-{job_id}.out, logs/*_{job_id}.log

    Args:
        job_id: Slurm job ID.
        tail_lines: Number of lines from end of log to return (default 100).
    """
    if not re.match(r"^\d+$", job_id):
        return f"Invalid job ID: {job_id}"
    normalized_tail_lines = _normalize_tail_lines(tail_lines)
    if normalized_tail_lines is None:
        return f"Invalid tail_lines: {tail_lines}. Use a positive integer up to {_MAX_TAIL_LINES}."

    patterns = [
        f"slurm-{job_id}.out",
        f"logs/*_{job_id}.log",
        f"logs/*_{job_id}.out",
        f"**/slurm-{job_id}.out",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            log_path = matches[0]
            try:
                content = _read_tail(log_path, normalized_tail_lines)
                return f"Log: {log_path}\n{content}"
            except Exception as e:
                return f"Error reading {log_path}: {e}"

    return f"No log file found for job {job_id}."


@mcp.tool()
def diagnose_job_failure(job_id: str) -> str:
    """Analyze a failed Slurm job: get exit code, resource usage, and common error patterns.

    Args:
        job_id: Slurm job ID.
    """
    if not re.match(r"^\d+$", job_id):
        return f"Invalid job ID: {job_id}"

    diag_format = "JobID,State,ExitCode,MaxRSS,MaxVMSize,Elapsed,TimelimitRaw,ReqMem,ReqCPUS"
    info = _run(["sacct", "-j", job_id, f"--format={diag_format}", "-n", "-X"])
    log = read_job_log(job_id, tail_lines=50)

    combined = info + "\n" + log
    diagnosis = []

    if "TIMEOUT" in info:
        diagnosis.append("JOB TIMED OUT: Wall time limit exceeded. Consider increasing --time.")
    if "OUT_OF_ME" in info or "oom" in combined.lower():
        diagnosis.append("OUT OF MEMORY: Job exceeded memory limit. Increase --mem or reduce problem size.")
    if "SIGKILL" in log or "signal 9" in log:
        diagnosis.append("KILLED BY SIGNAL 9: Likely OOM killer. Check memory usage.")
    if "MPI_ABORT" in log or "mpirun" in log.lower():
        diagnosis.append("MPI ERROR: Check for MPI rank failures, missing libraries, or node communication issues.")
    if "segmentation fault" in log.lower() or "sigsegv" in log.lower():
        diagnosis.append("SEGFAULT: Segmentation fault detected. Check array bounds and memory access.")
    if "no such file" in log.lower() or "file not found" in log.lower():
        diagnosis.append("MISSING FILE: A required file was not found. Check input paths and staging.")

    output = f"=== Job {job_id} Summary ===\n{info}\n"
    if diagnosis:
        output += "\n=== Diagnosis ===\n" + "\n".join(diagnosis) + "\n"
    output += f"\n=== Log (last 50 lines) ===\n{log}"
    return output


if __name__ == "__main__":
    mcp.run()
