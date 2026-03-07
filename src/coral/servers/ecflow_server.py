"""MCP server for querying ecFlow suite status and task outputs."""

from __future__ import annotations

import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coral-ecflow")


def _run_ecflow(args: list[str], timeout: int = 30) -> str:
    """Run ecflow_client with given arguments."""
    cmd = ["ecflow_client"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 and result.stderr:
            return f"ecflow_client error: {result.stderr.strip()}"
        return result.stdout
    except FileNotFoundError:
        return "ecflow_client not found. Make sure ecFlow module is loaded."
    except subprocess.TimeoutExpired:
        return f"ecflow_client timed out after {timeout}s."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_suite_status(suite_name: str = "") -> str:
    """Get ecFlow suite status. Shows task states (complete, active, aborted, queued).

    Args:
        suite_name: Name of the ecFlow suite. If empty, shows all suites.
    """
    args = ["--get_state"]
    if suite_name:
        args.append(f"/{suite_name}")
    output = _run_ecflow(args)
    if len(output) > 5000:
        return output[:5000] + "\n...[truncated]..."
    return output if output.strip() else "No suite found or ecflow_client not available."


@mcp.tool()
def get_aborted_tasks(suite_name: str) -> str:
    """Find all aborted tasks in an ecFlow suite.

    Args:
        suite_name: Name of the ecFlow suite.
    """
    output = _run_ecflow(["--get_state", f"/{suite_name}"])
    if output.startswith("ecflow_client"):
        return output  # error message

    aborted = [line.strip() for line in output.split("\n") if "state:aborted" in line]
    if aborted:
        return f"Found {len(aborted)} aborted tasks:\n" + "\n".join(aborted[:50])
    return f"No aborted tasks in /{suite_name}."


@mcp.tool()
def read_ecflow_job_output(task_path: str) -> str:
    """Read the job output (.1 file) for an ecFlow task.

    Args:
        task_path: Full ecFlow task path, e.g., /stofs/forecast/run_model
    """
    output = _run_ecflow(["--file", task_path, "jobout"])
    if not output.strip():
        return "No output found."
    if len(output) > 5000:
        return output[:2000] + "\n...[truncated]...\n" + output[-2000:]
    return output


if __name__ == "__main__":
    mcp.run()
