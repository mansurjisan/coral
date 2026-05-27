"""CORAL CLI: interactive chat, web UI server, and RAG indexing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.text import Text

from coral.config import get_model, set_cli_model

logger = logging.getLogger(__name__)

_SLASH_COMMANDS = [
    "/help",
    "/clear",
    "/reset",
    "/mode",
    "/save",
    "/memory",
    "/remember",
    "/forget",
    "/tools",
    "/status",
    "/report",
    "/watch",
    "/audit",
    "/route",
    "/techmemo",
    "/alert",
    "/branch",
    "/branches",
    "/retry",
    "/undo",
]
_slash_completer = WordCompleter(_SLASH_COMMANDS, sentence=True)

CORAL_BANNER = r"""[bold cyan]
   ██████╗ ██████╗ ██████╗  █████╗ ██╗
  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ╚██████╗╚██████╔╝██║  ██║██║  ██║███████╗
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝[/]"""

app = typer.Typer(
    name="coral",
    help="CORAL - Coastal Ocean Research AI Layer",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Auto Ollama detection
# ---------------------------------------------------------------------------


def _auto_detect_ollama() -> None:
    """Read coral_host.env to set OLLAMA_HOST if not already set."""
    if os.environ.get("OLLAMA_HOST"):
        return  # User already set it

    # Check common locations for coral_host.env
    user = os.environ.get("USER", "")
    work = os.environ.get("WORK", "")
    candidates = [
        # Ursa
        Path(f"/scratch5/purged/{user}/coral_host.env"),
        # TACC Vista
        Path(f"{work}/coral_install/coral_host.env") if work else None,
        # Generic
        Path.home() / "coral_host.env",
        Path("coral_host.env"),
    ]
    candidates = [c for c in candidates if c is not None]
    for candidate in candidates:
        if candidate.exists():
            try:
                for line in candidate.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("OLLAMA_NODE="):
                        node = line.split("=", 1)[1].strip()
                        os.environ["OLLAMA_HOST"] = f"http://{node}:11434"
                        console.print(f"[dim]Auto-detected Ollama at {node}:11434 (from {candidate})[/]")
                        return
            except OSError:
                continue


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

SLASH_COMMANDS_HELP = """\
[bold]Available commands:[/]
  [cyan]/clear[/]        Clear conversation history
  [cyan]/reset[/]        Reset all agents and history
  [cyan]/mode[/]         Show current agent mode
  [cyan]/save[/]         Save conversation (markdown by default; .json for JSON)
  [cyan]/memory[/]       Show saved memories
  [cyan]/remember[/]     Save a memory (e.g. /remember account = coastal-act)
  [cyan]/forget[/]       Remove a memory (e.g. /forget account)
  [cyan]/tools[/]        Show tool count per section
  [cyan]/status[/]       Quick dashboard: jobs, quota, Ollama health
  [cyan]/report[/]       Generate HPC status report as markdown
  [cyan]/watch[/]        Watch a job (e.g. /watch 9848988)
  [cyan]/audit[/]        Show tool call history and stats (current session only;
                  for persistent history use [bold]coral audit[/])
  [cyan]/route[/]        Show how the last query was classified (multi-agent mode)
  [cyan]/techmemo[/]     Auto-generate NOAA tech memo draft
  [cyan]/alert[/]        Set a threshold alert (e.g. /alert 8518750 > 1.5)
                  /alert list — show all alerts  /alert check — check now
  [cyan]/branch[/]       Save/load/delete persistent branches:
                  /branch <name> — save current chat as branch, start fresh
                  /branch load <name> — restore a saved branch
                  /branch delete <name> — remove a saved branch
  [cyan]/branches[/]     List saved branches (persisted to ~/.coral/branches.json)
  [cyan]/retry[/]        Rerun the last query (drops the last answer first)
  [cyan]/undo[/]         Drop the last user/assistant exchange
  [cyan]/help[/]         Show this help

  [bold]Query prefixes:[/]
  [cyan]@model[/]        Use a different model for one query (e.g. @qwen3:8b What is SCHISM?)
"""


async def _handle_slash_command(
    cmd: str,
    agent,
    memory,
    chat_log: list[dict],
    console: Console,
) -> bool:
    """Handle a slash command. Returns True if handled."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        console.print(SLASH_COMMANDS_HELP)
        return True

    if command == "/clear":
        if hasattr(agent, "reset"):
            agent.reset()
        elif hasattr(agent, "clear_history"):
            agent.clear_history()
        chat_log.clear()
        console.print("[dim]History cleared.[/]")
        return True

    if command == "/reset":
        if hasattr(agent, "reset"):
            agent.reset()
        elif hasattr(agent, "clear_history"):
            agent.clear_history()
        chat_log.clear()
        console.print("[dim]All agents reset.[/]")
        return True

    if command == "/mode":
        mode_name = "multi-agent" if hasattr(agent, "agents") else "single-agent"
        console.print(f"[dim]Mode: {mode_name}[/]")
        return True

    if command == "/save":
        _save_conversation(chat_log, arg, console)
        return True

    if command == "/memory":
        entries = memory.list_all()
        if not entries:
            console.print("[dim]No memories saved yet.[/]")
        else:
            console.print("[bold]Saved memories:[/]")
            for key, value in entries.items():
                console.print(f"  [cyan]{key}[/] = {value}")
        return True

    if command == "/remember":
        if "=" not in arg:
            console.print("[dim]Usage: /remember key = value[/]")
            return True
        key, _, value = arg.partition("=")
        memory.set(key.strip(), value.strip())
        console.print(f"[dim]Remembered: {key.strip()} = {value.strip()}[/]")
        return True

    if command == "/forget":
        key = arg.strip()
        if not key:
            console.print("[dim]Usage: /forget key[/]")
            return True
        if memory.delete(key):
            console.print(f"[dim]Forgot: {key}[/]")
        else:
            console.print(f"[dim]No memory found for: {key}[/]")
        return True

    if command == "/tools":
        if hasattr(agent, "agents"):
            for name, ag in agent.agents.items():
                console.print(f"  [cyan]{name:10s}[/] {len(ag.tools)} tools")
        else:
            console.print(f"  Tools: {len(agent.tools)}")
        return True

    if command == "/status":
        await _show_status_dashboard(agent, console)
        return True

    if command == "/report":
        await _generate_report(agent, chat_log, console)
        return True

    if command == "/watch":
        import re as _re

        job_id = arg.strip()
        # Accept Slurm numeric IDs and PBS IDs like 12345.server
        if not job_id or not _re.match(r"^[\d]+(\.\w+)*$", job_id):
            console.print("[dim]Usage: /watch <job_id>  (e.g. /watch 9848988 or /watch 12345.svc)[/]")
            return True
        _start_job_watcher(job_id, console)
        return True

    if command == "/audit":
        _show_audit(console)
        return True

    if command == "/route":
        _show_route(agent, console)
        return True

    if command == "/techmemo":
        await _generate_techmemo(agent, console)
        return True

    if command == "/alert":
        await _set_alert_via_mcp(arg, agent, console)
        return True

    if command == "/branch":
        _branch_conversation(arg, chat_log, agent, console)
        return True

    if command == "/branches":
        _list_branches(console)
        return True

    if command == "/undo":
        _undo_last_turn(agent, chat_log, console)
        return True

    if command == "/retry":
        await _retry_last_query(agent, chat_log, console)
        return True

    return False


# ---------------------------------------------------------------------------
# Alert system
# ---------------------------------------------------------------------------

_alert_mcp_bridge = None  # Set during chat() init if alerts server is available


async def _set_alert_via_mcp(arg: str, agent, console: Console) -> None:
    """Create a threshold alert via the MCP alert server.

    Usage: /alert <station_id> <operator> <threshold>
    Example: /alert 8518750 > 1.5
    """

    parts = arg.strip().split()

    # Sub-commands (before length check)
    if parts and parts[0] == "list":
        await _alert_subcommand(agent, "coral_list_alerts", {}, console)
        return
    if parts and parts[0] == "check":
        await _alert_subcommand(agent, "coral_check_alerts", {}, console)
        return

    if len(parts) < 3:
        console.print(
            "[dim]Usage: /alert <station_id> <operator> <value>\n"
            "  Example: /alert 8518750 > 1.5\n"
            "  Operators: > < >= <=\n"
            "  Also: /alert list | /alert check[/]"
        )
        return

    station_id = parts[0]
    operator = parts[1]
    try:
        threshold = float(parts[2])
    except ValueError:
        console.print(f"[red]Invalid threshold: {parts[2]}[/]")
        return

    if operator not in (">", "<", ">=", "<="):
        console.print(f"[red]Invalid operator: {operator}. Use > < >= <=[/]")
        return

    # Create alert via MCP tool
    bridge = _get_mcp_bridge(agent)
    if bridge and "coral_create_alert" in bridge.tool_map:
        try:
            result = await bridge.call_tool(
                "coral_create_alert",
                {
                    "station_id": station_id,
                    "operator": operator,
                    "threshold": threshold,
                },
            )
            console.print(Markdown(result))

            # Start background MCP-mediated check loop
            _start_mcp_alert_loop(bridge, console)
        except Exception as e:
            console.print(f"[red]Alert creation failed:[/] {e}")
    else:
        console.print(
            "[yellow]Alert MCP server not available.[/]\n"
            "[dim]Install alert-mcp: pip install -e ocean-mcp/servers/alert-mcp[/]"
        )


async def _alert_subcommand(agent, tool_name: str, args: dict, console: Console) -> None:
    """Run an alert sub-command via MCP."""
    bridge = _get_mcp_bridge(agent)
    if bridge and tool_name in bridge.tool_map:
        try:
            result = await bridge.call_tool(tool_name, args)
            console.print(Markdown(result))
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
    else:
        console.print(f"[yellow]Tool '{tool_name}' not available.[/]")


def _get_mcp_bridge(agent):
    """Extract MCP bridge from agent or orchestrator."""
    if hasattr(agent, "mcp_bridge"):
        return agent.mcp_bridge
    if hasattr(agent, "agents"):
        # Orchestrator — get bridge from any sub-agent
        for ag in agent.agents.values():
            if hasattr(ag, "mcp_bridge"):
                return ag.mcp_bridge
    return None


_alert_loop_started = False


def _start_mcp_alert_loop(bridge, console: Console) -> None:
    """Start a background thread that periodically checks alerts via MCP."""
    global _alert_loop_started
    if _alert_loop_started:
        return  # Only one loop
    _alert_loop_started = True

    import asyncio
    import threading

    def poll():
        loop = asyncio.new_event_loop()
        while True:
            import time as _t

            _t.sleep(300)  # Check every 5 minutes
            try:
                if "coral_check_alerts" in bridge.tool_map:
                    result = loop.run_until_complete(bridge.call_tool("coral_check_alerts", {}))
                    if "TRIGGERED" in result:
                        console.print(f"\n[bold red]🚨 {result}[/]")
                        _audit_log.append(
                            {
                                "tool": "coral_check_alerts",
                                "args": "background_poll",
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "result_len": len(result),
                            }
                        )
            except Exception as e:
                logger.debug("Alert check cycle failed: %s", e)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Conversation branching (persisted to ~/.coral/branches.json)
# ---------------------------------------------------------------------------


def _branches_file() -> Path:
    """Path for the persisted branches file (alongside memory + sessions)."""
    from coral.memory import _default_memory_dir

    return _default_memory_dir() / "branches.json"


# In-process cache of branches loaded from disk. Populated on first access.
_branches: dict[str, list[dict]] = {}
_branches_loaded = False


def _load_branches() -> dict[str, list[dict]]:
    """Lazily load branches from disk into the module cache and return it."""
    global _branches_loaded
    if _branches_loaded:
        return _branches

    path = _branches_file()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            stored = data.get("branches", {}) if isinstance(data, dict) else {}
            _branches.clear()
            _branches.update(stored)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load branches from %s: %s", path, exc)
    _branches_loaded = True
    return _branches


def _save_branches() -> None:
    """Persist the in-memory branches dict to disk."""
    path = _branches_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "branches": _branches}, indent=2))
    except OSError as exc:
        logger.warning("Could not save branches to %s: %s", path, exc)


def _reset_agent_state(agent) -> None:
    """Clear agent/orchestrator histories so a new branch starts fresh."""
    if hasattr(agent, "reset"):
        agent.reset()
    elif hasattr(agent, "clear_history"):
        agent.clear_history()


def _branch_conversation(arg: str, chat_log: list[dict], agent, console: Console) -> None:
    """Handle /branch [save|load|delete] <name>.

    Bare ``/branch <name>`` preserves the legacy save-and-reset semantic.
    """
    parts = arg.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""
    name = parts[1].strip() if len(parts) > 1 else ""

    branches = _load_branches()

    if sub == "load":
        if not name:
            console.print("[dim]Usage: /branch load <name>[/]")
            return
        if name not in branches:
            console.print(f"[red]No branch named '{name}'.[/]")
            return
        chat_log.clear()
        chat_log.extend(branches[name])
        _reset_agent_state(agent)
        console.print(
            f"[green]Loaded branch '{name}' ({len(chat_log)} messages).[/] "
            "[dim]Agent context starts fresh; history is for /save export only.[/]"
        )
        return

    if sub == "delete":
        if not name:
            console.print("[dim]Usage: /branch delete <name>[/]")
            return
        if name not in branches:
            console.print(f"[red]No branch named '{name}'.[/]")
            return
        del branches[name]
        _save_branches()
        console.print(f"[dim]Deleted branch '{name}'.[/]")
        return

    # Default and explicit "save" both save-and-reset.
    if sub == "save":
        save_name = name
    else:
        save_name = arg.strip()

    if not save_name:
        save_name = f"branch_{len(branches) + 1}"

    branches[save_name] = list(chat_log)
    _save_branches()
    chat_log.clear()
    _reset_agent_state(agent)

    console.print(f"[green]Saved branch '{save_name}' ({len(branches[save_name])} messages). Starting fresh.[/]")


def _list_branches(console: Console) -> None:
    """List all saved conversation branches."""
    branches = _load_branches()
    if not branches:
        console.print("[dim]No branches saved. Use /branch <name> to create one.[/]")
        return

    console.print("[bold]Conversation branches:[/]")
    for name, log in branches.items():
        msg_count = len(log)
        last_msg = log[-1]["content"][:60] + "..." if log else ""
        console.print(f"  [cyan]{name}[/] — {msg_count} messages — {last_msg}")


# ---------------------------------------------------------------------------
# /undo and /retry
# ---------------------------------------------------------------------------


def _truncate_history_to_before_last_user(history: list[dict]) -> bool:
    """Trim the agent's history list to before its most recent user entry.

    Returns True if anything was removed.
    """
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            del history[i:]
            return True
    return False


def _undo_last_turn(agent, chat_log: list[dict], console: Console) -> bool:
    """Drop the most recent user/assistant exchange from chat_log and agent state.

    Returns True if something was removed. Called by both /undo and /retry.
    """
    if not chat_log:
        console.print("[dim]Nothing to undo.[/]")
        return False

    # Pop the trailing assistant turn (if any) plus the user that produced it.
    if chat_log and chat_log[-1].get("role") == "assistant":
        chat_log.pop()
    if chat_log and chat_log[-1].get("role") == "user":
        chat_log.pop()

    # Keep agent histories in sync.
    if hasattr(agent, "history"):
        _truncate_history_to_before_last_user(agent.history)
    if hasattr(agent, "agents"):
        for sub in agent.agents.values():
            if hasattr(sub, "history"):
                _truncate_history_to_before_last_user(sub.history)

    console.print("[dim]Last turn dropped.[/]")
    return True


async def _retry_last_query(agent, chat_log: list[dict], console: Console) -> None:
    """Rerun the most recent user query after rolling back its last answer."""
    last_user = next(
        (entry["content"] for entry in reversed(chat_log) if entry.get("role") == "user"),
        None,
    )
    if not last_user:
        console.print("[dim]Nothing to retry.[/]")
        return

    _undo_last_turn(agent, chat_log, console)

    chat_log.append({"role": "user", "content": last_user})
    try:
        with Status(
            "🪸 [cyan]Retrying...[/]",
            console=console,
            spinner="dots",
        ):
            response = await agent.chat(last_user)
    except Exception as exc:
        console.print(f"[red]Retry failed:[/] {exc}")
        return

    chat_log.append({"role": "assistant", "content": response})
    console.print()
    console.print(Rule(style="cyan"))
    console.print("[bold cyan]CORAL:[/]")
    console.print(Markdown(response))
    console.print(Rule(style="dim"))
    console.print()


# Tool call audit log (populated by the tool callback)
_audit_log: list[dict] = []


def _show_audit(console: Console) -> None:
    """Show tool call history and stats."""
    if not _audit_log:
        console.print("[dim]No tool calls recorded yet.[/]")
        return

    lines = ["[bold]Tool Call History:[/]\n"]
    for entry in _audit_log[-20:]:  # Last 20 calls
        time_str = entry.get("time", "?")
        result_len = entry.get("result_len", 0)
        size_label = f"{result_len} chars" if result_len else ""
        lines.append(f"  [dim]{time_str}[/] [cyan]{entry['tool']:30s}[/] [dim]{size_label}[/]")

    # Summary stats
    total = len(_audit_log)
    tools_used = {}
    for entry in _audit_log:
        tools_used[entry["tool"]] = tools_used.get(entry["tool"], 0) + 1
    top_tools = sorted(tools_used.items(), key=lambda x: -x[1])[:5]

    lines.append(f"\n[bold]Summary:[/] {total} total calls")
    lines.append("[bold]Most used:[/]")
    for tool, count in top_tools:
        lines.append(f"  [cyan]{tool}[/] — {count}x")

    console.print("\n".join(lines))


def _show_route(agent, console: Console) -> None:
    """Show how the last query was classified by the orchestrator."""
    decision = getattr(agent, "last_route_decision", None)
    if not decision:
        if not hasattr(agent, "agents"):
            console.print(
                "[dim]Routing is only used in multi-agent mode. Restart with [bold]coral chat --mode multi[/].[/]"
            )
        else:
            console.print("[dim]No query routed yet — ask a question first.[/]")
        return

    cats = ", ".join(decision.get("categories", [])) or "(none)"
    method = decision.get("method", "?")
    confidence = decision.get("confidence", 0.0)
    matched = decision.get("matched_keywords") or []
    query = (decision.get("query") or "").strip()

    lines = ["[bold]Last route decision:[/]"]
    lines.append(f"  [dim]Query:[/]      {query[:120]}{'…' if len(query) > 120 else ''}")
    lines.append(f"  [dim]Route:[/]      [cyan]{cats}[/]")
    lines.append(f"  [dim]Method:[/]     {method}")
    lines.append(f"  [dim]Confidence:[/] {confidence:.2f}")
    if matched:
        shown = ", ".join(matched[:8])
        more = f" (+{len(matched) - 8} more)" if len(matched) > 8 else ""
        lines.append(f"  [dim]Matched:[/]    {shown}{more}")
    if method in ("llm", "default"):
        rm = decision.get("router_model")
        if rm:
            lines.append(f"  [dim]Router model:[/] {rm}")
        raw = decision.get("raw_response")
        if raw:
            lines.append(f"  [dim]Raw LLM:[/]   {raw[:80]}")

    console.print("\n".join(lines))


async def _show_status_dashboard(agent, console: Console) -> None:
    """Quick dashboard: running jobs, Ollama health, key stats."""
    import httpx

    sections: list[str] = []

    # Ollama health
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_host}/api/version")
            version = resp.json().get("version", "unknown")
        sections.append(f"  [green]✓[/] Ollama [dim]{ollama_host}[/] — v{version}")
    except Exception:
        sections.append(f"  [red]✗[/] Ollama [dim]{ollama_host}[/] — unreachable")

    # Running jobs — try Slurm first, fall back to PBS
    import shutil
    import subprocess

    if shutil.which("squeue"):
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T %M"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            jobs = result.stdout.strip().split("\n") if result.stdout.strip() else []
            running = [j for j in jobs if "RUNNING" in j]
            pending = [j for j in jobs if "PENDING" in j]
            sections.append(f"  [green]✓[/] Slurm — {len(running)} running, {len(pending)} pending")
            for job in running[:5]:
                sections.append(f"    [dim]{job}[/]")
        except Exception:
            sections.append("  [yellow]?[/] Slurm — error querying")
    elif shutil.which("qstat"):
        try:
            result = subprocess.run(
                ["qstat", "-u", os.environ.get("USER", "")],
                capture_output=True,
                text=True,
                timeout=10,
            )
            job_lines = [
                line
                for line in result.stdout.strip().split("\n")
                if line.strip() and not line.startswith("---") and "Job ID" not in line
            ]
            sections.append(f"  [green]✓[/] PBS — {len(job_lines)} jobs")
            for job in job_lines[:5]:
                sections.append(f"    [dim]{job.strip()[:80]}[/]")
        except Exception:
            sections.append("  [yellow]?[/] PBS — error querying")
    else:
        sections.append("  [dim]-[/] No scheduler (Slurm/PBS) found")

    # Disk usage summary
    user = os.environ.get("USER", "")
    scratch5 = f"/scratch5/purged/{user}"
    if os.path.isdir(scratch5):
        try:
            import subprocess

            result = subprocess.run(
                ["du", "-sh", scratch5],
                capture_output=True,
                text=True,
                timeout=30,
            )
            size = result.stdout.strip().split()[0] if result.stdout.strip() else "?"
            sections.append(f"  [green]✓[/] Scratch5 — {size} used")
        except Exception:
            sections.append("  [yellow]?[/] Scratch5 — could not check")

    console.print(
        Panel(
            "\n".join(sections),
            title="[bold cyan]Status[/]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _start_job_watcher(job_id: str, console: Console) -> None:
    """Start a background thread that polls job status and notifies when done.

    Supports both Slurm (sacct) and PBS (qstat) schedulers.
    """
    import shutil
    import subprocess
    import threading

    use_pbs = shutil.which("qstat") and not shutil.which("sacct")

    def _check_slurm() -> str | None:
        """Check Slurm job state. Returns terminal state or None if still running."""
        result = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-X", "--format=State", "--parsable2"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        state = result.stdout.strip().split("\n")[0].strip() if result.stdout.strip() else ""
        if state and state not in ("RUNNING", "PENDING", "REQUEUED", "SUSPENDED", ""):
            return state
        return None

    def _check_pbs() -> str | None:
        """Check PBS job state. Returns terminal state or None if still running."""
        result = subprocess.run(
            ["qstat", "-f", job_id],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            # Job gone from PBS = finished
            return "COMPLETED (no longer in PBS)"
        for line in result.stdout.split("\n"):
            if "job_state" in line:
                state = line.split("=")[-1].strip()
                if state in ("F", "E", "X"):  # Finished, Exiting, terminated
                    return f"FINISHED (state={state})"
        return None

    def poll():
        console.print(f"[dim]Watching job {job_id} ({'PBS' if use_pbs else 'Slurm'})... (will notify when done)[/]")
        poll_interval = 30
        max_polls = 480

        for _ in range(max_polls):
            import time

            time.sleep(poll_interval)
            try:
                state = _check_pbs() if use_pbs else _check_slurm()
                if state:
                    console.print(f"\n[bold yellow]🔔 Job {job_id} finished: {state}[/]")
                    return
            except Exception:
                continue

        console.print(f"[dim]Stopped watching job {job_id} (timeout after 4 hours)[/]")

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()


async def _generate_report(agent, chat_log: list[dict], console: Console) -> None:
    """Generate an HPC status report by querying tools."""
    if not hasattr(agent, "chat"):
        console.print("[dim]Report requires an active agent.[/]")
        return

    console.print("[dim]Generating HPC status report...[/]")
    report_query = (
        "Generate a brief HPC status report. Include: "
        "1) My disk quota summary, "
        "2) My recent jobs from the last 3 days, "
        "3) My Slurm account info, "
        "4) Any running experiments. "
        "Format as a clean markdown report."
    )
    try:
        response = await agent.chat(report_query)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coral_report_{ts}.md"
        Path(filename).write_text(
            f"# CORAL HPC Status Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{response}\n"
        )
        console.print()
        console.print(Rule(style="cyan"))
        console.print("[bold cyan]CORAL Report:[/]")
        console.print(Markdown(response))
        console.print(Rule(style="dim"))
        console.print(f"\n[green]Report saved to {filename}[/]\n")
    except Exception as e:
        console.print(f"[red]Report generation failed:[/] {e}")


async def _generate_techmemo(agent, console: Console) -> None:
    """Auto-generate a NOAA tech memo draft about CORAL."""
    console.print("[dim]Generating NOAA Tech Memo draft...[/]")

    # Gather system stats
    tool_count = 0
    server_count = 0
    if hasattr(agent, "agents"):
        for ag in agent.agents.values():
            tool_count += len(ag.tools)
        server_count = len(
            set(
                ag.mcp_bridge.tool_server_map.get(t["function"]["name"], "")
                for ag in agent.agents.values()
                for t in ag.tools
            )
        )
    elif hasattr(agent, "tools"):
        tool_count = len(agent.tools)

    memo_prompt = f"""\
Generate a NOAA Technical Memorandum draft about CORAL (Coastal Ocean Research AI Layer).

Use this system information:
- Total tools: {tool_count}
- MCP servers: {server_count}
- Agent architecture: Multi-agent (DATA, CODE, WORKFLOW) with orchestrator routing
- Deployment: Self-hosted on NOAA RDHPCS (Ursa HPC) using Ollama + local LLM
- No cloud LLM dependency

Structure the memo as:
1. Abstract (1 paragraph)
2. Introduction — problem statement (tool overload in single-agent HPC assistants)
3. System Architecture — multi-agent routing, MCP integration, section boundaries
4. Capabilities — ocean data retrieval (CO-OPS, STOFS, ERDDAP), code analysis (RAG),
   HPC workflow management (Slurm, ecFlow, UFS experiments), system administration
5. Operational Workflow Integration — NOS OFS configs, failure diagnostics, ensemble support
6. Deployment — Ollama on GPU node, CORAL on service node, Apptainer sandbox
7. Results — cross-domain query examples, tool call routing accuracy
8. Conclusion and Future Work

Write in formal NOAA technical report style. Include specific numbers and examples.
Format as clean markdown.
"""

    try:
        with Status("🪸 [cyan]Writing tech memo...[/]", console=console, spinner="dots"):
            response = await agent.chat(memo_prompt)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coral_techmemo_{ts}.md"
        Path(filename).write_text(
            f"# NOAA Technical Memorandum — CORAL\n"
            f"# Coastal Ocean Research AI Layer\n"
            f"# Draft generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{response}\n"
        )

        console.print()
        console.print(Rule(style="cyan"))
        console.print("[bold cyan]Tech Memo Draft:[/]")
        console.print(Markdown(response))
        console.print(Rule(style="dim"))
        console.print(f"\n[green]Saved to {filename}[/]\n")
    except Exception as e:
        console.print(f"[red]Tech memo generation failed:[/] {e}")


def _save_conversation(
    chat_log: list[dict],
    arg: str,
    console: Console,
) -> None:
    """Save conversation. Markdown by default; JSON if filename ends in .json
    or arg starts with --json.
    """
    if not chat_log:
        console.print("[dim]No conversation to save.[/]")
        return

    arg = (arg or "").strip()
    use_json = False
    filename = arg

    if arg.startswith("--json"):
        use_json = True
        filename = arg[len("--json") :].strip()

    if filename and filename.lower().endswith(".json"):
        use_json = True

    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "json" if use_json else "md"
        filename = f"coral_chat_{ts}.{ext}"

    if use_json:
        payload = {
            "version": 1,
            "saved_at": datetime.now().astimezone().isoformat(),
            "model": os.environ.get("CORAL_MODEL", ""),
            "messages": [
                {"role": entry.get("role"), "content": entry.get("content", "")}
                for entry in chat_log
                if entry.get("role") in ("user", "assistant")
            ],
        }
        Path(filename).write_text(json.dumps(payload, indent=2))
    else:
        lines = [f"# CORAL Chat — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        for entry in chat_log:
            if entry["role"] == "user":
                lines.append(f"## You\n\n{entry['content']}\n")
            elif entry["role"] == "assistant":
                lines.append(f"## CORAL\n\n{entry['content']}\n")
        Path(filename).write_text("\n".join(lines))

    console.print(f"[green]Saved to {filename}[/]")


def _session_file() -> Path:
    """Return the path for the session history file."""
    from coral.memory import _default_memory_dir

    return _default_memory_dir() / "last_session.json"


def _save_session(chat_log: list[dict]) -> None:
    """Persist chat history for session restore."""
    path = _session_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chat_log, indent=2))


def _load_session() -> list[dict]:
    """Load previous session's chat history."""
    path = _session_file()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return []


# ---------------------------------------------------------------------------
# Tool call display callback
# ---------------------------------------------------------------------------


def _get_agent_stats(agent) -> dict:
    """Extract token stats from the agent's last response."""
    # Multi-agent: check sub-agents for stats
    if hasattr(agent, "agents"):
        combined: dict = {}
        for ag in agent.agents.values():
            stats = getattr(ag, "last_stats", {})
            if stats.get("tokens"):
                combined["tokens"] = combined.get("tokens", 0) + stats["tokens"]
            if stats.get("tokens_per_sec"):
                combined["tokens_per_sec"] = stats["tokens_per_sec"]  # Use last agent's speed
        return combined
    # Single agent
    return getattr(agent, "last_stats", {})


def _make_tool_callback(console: Console, memory=None):
    """Create a tool-call callback that displays calls and auto-learns."""

    def on_tool_call(name, args, result):
        args_short = str(args)
        if len(args_short) > 80:
            args_short = args_short[:80] + "..."
        console.print(f"  [yellow]⚡ {name}[/]({args_short})")

        # Record in audit log
        _audit_log.append(
            {
                "tool": name,
                "args": args_short,
                "time": datetime.now().strftime("%H:%M:%S"),
                "result_len": len(str(result)),
            }
        )

        # Auto-learn from tool results
        if memory is None:
            return
        result_str = str(result)
        try:
            _auto_learn_from_tool(memory, name, args, result_str)
        except Exception:
            pass  # Never crash on auto-learn

    return on_tool_call


def _auto_learn_from_tool(memory, tool_name: str, args: dict, result: str) -> None:
    """Extract useful facts from tool results and save to memory."""
    import re

    if tool_name == "hpc_account_info" and "Account" in result:
        # Extract account names from the table
        accounts = set()
        for line in result.split("\n"):
            if "|" in line and not line.startswith("|--"):
                cols = [c.strip() for c in line.split("|") if c.strip()]
                if cols and cols[0] not in ("Account", ""):
                    accounts.add(cols[0])
        if accounts:
            memory.auto_learn("slurm_accounts", ", ".join(sorted(accounts)))

    elif tool_name == "hpc_disk_quota" and "/scratch" in result:
        # Remember which scratch filesystems the user has
        scratches = re.findall(r"(/scratch\d+)", result)
        if scratches:
            memory.auto_learn("scratch_filesystems", ", ".join(sorted(set(scratches))))

    elif tool_name == "hpc_user_groups" and "gid=" in result:
        # Extract primary group
        match = re.search(r"gid=\d+\((\w+)\)", result)
        if match:
            memory.auto_learn("primary_group", match.group(1))

    elif tool_name == "hpc_fairshare" and "FairShare" in result:
        # Remember FairShare factor
        lines = result.split("\n")
        for line in lines:
            parts = line.split()
            user = os.environ.get("USER", "")
            if user and user[:8] in line and len(parts) >= 7:
                try:
                    factor = float(parts[-1])
                    memory.auto_learn("fairshare_factor", str(factor))
                except (ValueError, IndexError):
                    pass

    elif tool_name == "hpc_system_info" and "PARTITION" in result:
        # Remember available partitions
        partitions = set()
        for line in result.split("\n"):
            parts = line.split()
            if parts and not parts[0].startswith("PARTITION") and not parts[0].startswith("─"):
                partitions.add(parts[0].rstrip("*"))
        if partitions:
            memory.auto_learn("partitions", ", ".join(sorted(partitions)))


# ---------------------------------------------------------------------------
# Chat command
# ---------------------------------------------------------------------------


@app.command()
def chat(
    model: str = typer.Option("", help="Ollama model name (default: CORAL_MODEL or qwen3:32b)"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
    mode: str = typer.Option("multi", help="Agent mode: 'multi' (orchestrator) or 'single' (legacy)"),
):
    """Interactive chat with CORAL."""
    from coral.mcp_bridge import MCPBridge
    from coral.memory import CoralMemory

    set_cli_model(model)
    resolved_model = get_model()

    async def run():
        # Suppress noisy MCP server logs during startup
        logging.getLogger("mcp").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        bridge = MCPBridge(config)
        memory = CoralMemory()
        chat_log: list[dict] = _load_session()

        # Auto-detect Ollama from coral_host.env
        _auto_detect_ollama()

        # Health check: verify Ollama is reachable before connecting
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{ollama_host}/api/tags")
                resp.raise_for_status()
        except Exception:
            console.print(f"[red]Cannot reach Ollama at {ollama_host}[/]")
            console.print("[dim]Check that Ollama is running and OLLAMA_HOST is set correctly.[/]")
            console.print("[dim]On Ursa: sbatch slurm/start_ollama.sh, then export OLLAMA_HOST=http://<node>:11434[/]")
            return

        console.print(CORAL_BANNER)
        with Status("🪸 [cyan]Connecting to MCP servers...[/]", console=console, spinner="dots"):
            await bridge.connect_all()
        tool_count = len(bridge.tools)

        # Tool call display callback with auto-learning
        tool_callback = _make_tool_callback(console, memory=memory)

        if mode == "single":
            from coral.agent import CoralAgent

            agent = CoralAgent(
                model=resolved_model,
                mcp_bridge=bridge,
                on_tool_call=tool_callback,
            )
            mode_label = "single agent"
        else:
            from coral.agents.orchestrator import create_orchestrator

            agent = create_orchestrator(model=resolved_model, mcp_bridge=bridge)
            # Attach tool call display to each sub-agent
            for ag in agent.agents.values():
                ag.on_tool_call = tool_callback
            mode_label = "multi-agent (data + code + workflow)"

        # Inject memory context into agent prompts
        mem_context = memory.to_prompt_context()
        if mem_context:
            if hasattr(agent, "agents"):
                for ag in agent.agents.values():
                    ag.system_prompt = ag.system_prompt + "\n\n" + mem_context
            elif hasattr(agent, "system_prompt"):
                agent.system_prompt = agent.system_prompt + "\n\n" + mem_context

        # Note: previous session is saved in chat_log for /save export,
        # but NOT rehydrated into agent history. Stale tool calls from
        # prior sessions can poison model behavior (e.g. wrong product
        # parameters propagating). Use /clear to reset if needed.
        if chat_log:
            console.print(
                f"[dim]{len(chat_log)} messages from previous session "
                f"(available for /save, not loaded into context).[/]"
            )

        # Status panel
        user = os.environ.get("USER", "unknown")
        host = os.environ.get("HOSTNAME", os.environ.get("HOST", "local"))
        status = Text.assemble(
            ("  🪸 ", ""),
            ("Model  ", "dim"),
            (resolved_model, "green"),
            ("  │  ", "dim"),
            ("Tools  ", "dim"),
            (str(tool_count), "green"),
            ("  │  ", "dim"),
            ("Mode  ", "dim"),
            (mode_label, "green"),
            ("\n  🖥️  ", ""),
            ("User   ", "dim"),
            (user, "cyan"),
            ("  │  ", "dim"),
            ("Host   ", "dim"),
            (host, "cyan"),
        )
        if memory.list_all():
            status.append("\n  🧠 ", style="")
            status.append(f"{len(memory.list_all())} memories loaded", style="dim")
        console.print(
            Panel(
                status,
                border_style="cyan",
                title="[bold cyan]CORAL v0.1.0[/]",
                subtitle="[dim]Coastal Ocean Research AI Layer · /help for commands[/]",
                padding=(0, 1),
            )
        )
        console.print()

        session: PromptSession[str] = PromptSession(
            completer=_slash_completer,
            complete_while_typing=True,
        )
        try:
            while True:
                try:
                    user_input = await session.prompt_async(HTML("<b>You: </b>"))
                except EOFError:
                    break

                stripped = user_input.strip()
                if stripped.lower() in ("exit", "quit"):
                    break
                if not stripped:
                    continue

                # Slash commands
                if stripped.startswith("/"):
                    handled = await _handle_slash_command(
                        stripped,
                        agent,
                        memory,
                        chat_log,
                        console,
                    )
                    if handled:
                        continue

                # Per-query model override: @modelname prefix
                original_models: dict[str, str] = {}
                if stripped.startswith("@") and " " in stripped:
                    model_override, stripped = stripped.split(" ", 1)
                    model_name = model_override[1:]  # Remove @
                    console.print(f"[dim]Using model: {model_name} for this query[/]")
                    if hasattr(agent, "agents"):
                        for ag in agent.agents.values():
                            original_models[ag.name] = ag.model
                            ag.model = model_name
                    elif hasattr(agent, "model"):
                        original_models["_single"] = agent.model
                        agent.model = model_name

                chat_log.append({"role": "user", "content": stripped})

                import time as _time

                t0 = _time.monotonic()

                try:
                    # Run query with Ctrl+C cancellation
                    import signal

                    _cancelled = False

                    def _cancel_handler(sig, frame):
                        nonlocal _cancelled
                        _cancelled = True
                        raise KeyboardInterrupt

                    old_handler = signal.signal(signal.SIGINT, _cancel_handler)
                    try:
                        with Status(
                            "🪸 [cyan]Thinking... (Ctrl+C to cancel)[/]",
                            console=console,
                            spinner="dots",
                        ):
                            response = await agent.chat(stripped)
                    except KeyboardInterrupt:
                        console.print("\n[dim]Query cancelled.[/]")
                        signal.signal(signal.SIGINT, old_handler)
                        continue
                    finally:
                        signal.signal(signal.SIGINT, old_handler)
                    elapsed = _time.monotonic() - t0

                    chat_log.append({"role": "assistant", "content": response})
                    console.print()
                    console.print(Rule(style="cyan"))
                    console.print("[bold cyan]CORAL:[/]")
                    console.print(Markdown(response))

                    # Token/timing stats + routing confidence
                    stats_parts = [f"{elapsed:.1f}s"]
                    stats = _get_agent_stats(agent)
                    if stats.get("tokens"):
                        stats_parts.append(f"{stats['tokens']} tokens")
                    if stats.get("tokens_per_sec"):
                        stats_parts.append(f"{stats['tokens_per_sec']} tok/s")
                    # Show routing confidence
                    if hasattr(agent, "last_route_confidence") and agent.last_route_confidence > 0:
                        conf = agent.last_route_confidence
                        conf_str = f"confidence {conf:.0%}"
                        stats_parts.append(conf_str)
                    console.print(
                        Rule(
                            title=f"[dim]{' · '.join(stats_parts)}[/]",
                            style="dim",
                        )
                    )
                    console.print()
                except Exception as e:
                    console.print(f"\n[red]Error:[/] {e}\n")
                finally:
                    # Restore original models after per-query override
                    if original_models:
                        if hasattr(agent, "agents"):
                            for ag in agent.agents.values():
                                if ag.name in original_models:
                                    ag.model = original_models[ag.name]
                        elif "_single" in original_models:
                            agent.model = original_models["_single"]
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/]")
        finally:
            # Save session for potential restore
            if chat_log:
                try:
                    _save_session(chat_log)
                except Exception:
                    pass
            try:
                await bridge.close()
            except Exception:
                pass

    asyncio.run(run())


@app.command()
def tools(
    config: str = typer.Option("coral_config.json", help="MCP config path"),
):
    """List all available MCP tools."""
    from coral.mcp_bridge import MCPBridge

    async def run():
        bridge = MCPBridge(config)
        await bridge.connect_all()
        for tool in bridge.tools:
            fn = tool["function"]
            server = bridge.tool_map[fn["name"]][1]
            console.print(f"  [cyan]{server:10s}[/] [bold]{fn['name']}[/]")
            if fn.get("description"):
                desc = fn["description"].split("\n")[0][:80]
                console.print(f"             [dim]{desc}[/]")
        console.print(f"\n[green]{len(bridge.tools)} tools total[/]")
        await bridge.close()

    asyncio.run(run())


@app.command()
def doctor(
    config: str = typer.Option("coral_config.json", help="MCP config path"),
):
    """Health-check Ollama and every configured MCP server.

    Exits with a non-zero code equal to the number of unreachable components.
    Useful for first-run setup and post-deploy smoke tests.
    """
    import httpx
    from rich.table import Table

    from coral.mcp_bridge import MCPBridge

    _auto_detect_ollama()

    async def run() -> int:
        table = Table(title="CORAL doctor", show_lines=False)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Tools", justify="right")
        table.add_column("Notes", style="dim")

        failures = 0

        # Ollama
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{ollama_host}/api/version")
                version = resp.json().get("version", "unknown")
            table.add_row("ollama", "[green]up[/]", "—", f"{ollama_host} v{version}")
        except Exception as exc:
            failures += 1
            table.add_row("ollama", "[red]down[/]", "—", f"{ollama_host}: {exc}")

        # MCP servers
        try:
            bridge = MCPBridge(config)
        except Exception as exc:
            console.print(f"[red]Could not load config {config}: {exc}[/]")
            console.print(table)
            return failures + 1

        expected = sorted(bridge.config.get("mcpServers", {}).keys())
        try:
            await bridge.connect_all()
        except Exception as exc:
            console.print(f"[red]Bridge connect_all failed: {exc}[/]")

        tool_counts: dict[str, int] = {}
        for server in bridge.tool_server_map.values():
            tool_counts[server] = tool_counts.get(server, 0) + 1

        for name in expected:
            if name in bridge.sessions:
                count = tool_counts.get(name, 0)
                table.add_row(name, "[green]up[/]", str(count), "")
            else:
                failures += 1
                table.add_row(name, "[red]down[/]", "—", "connect failed")

        try:
            await bridge.close()
        except Exception:
            pass

        console.print(table)
        if failures:
            console.print(f"[red]{failures} component(s) unreachable.[/]")
        else:
            console.print("[green]All components healthy.[/]")
        return failures

    rc = asyncio.run(run())
    if rc:
        raise typer.Exit(code=rc)


@app.command()
def audit(
    query_id: str = typer.Option("", "--query-id", help="Filter by query id"),
    tool: str = typer.Option("", "--tool", help="Substring match against tool name"),
    section: str = typer.Option("", "--section", help="DATA, CODE, or WORKFLOW"),
    event: str = typer.Option("", "--event", help="Filter by event type (e.g. tool_call)"),
    since: str = typer.Option("", "--since", help="ISO timestamp or relative (e.g. '1h', '30m', '2d')"),
    limit: int = typer.Option(50, "--limit", help="Max entries to display"),
    output_format: str = typer.Option("table", "--format", help="table | json | jsonl"),
):
    """Query the persistent CORAL audit log (logs/coral_audit.jsonl)."""
    import json as _json
    from datetime import datetime, timezone

    from rich.table import Table

    from coral.audit import _audit_log_path

    path = _audit_log_path()
    if not path.exists():
        console.print(f"[yellow]No audit log found at {path}.[/]")
        raise typer.Exit(code=0)

    cutoff: datetime | None = None
    if since:
        cutoff = _parse_since(since)
        if cutoff is None:
            console.print(f"[red]Could not parse --since '{since}'.[/]")
            raise typer.Exit(code=2)

    section_upper = section.upper().strip()
    matches: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if query_id and entry.get("query_id") != query_id:
                continue
            if event and entry.get("event") != event:
                continue
            if section_upper:
                entry_section = (entry.get("section") or "").upper()
                routed = [s.upper() for s in entry.get("routed_sections") or []]
                if section_upper != entry_section and section_upper not in routed:
                    continue
            if tool and tool not in (entry.get("tool") or ""):
                continue
            if cutoff is not None:
                ts = entry.get("timestamp")
                try:
                    when = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
                except ValueError:
                    when = None
                if when is None:
                    continue
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                if when < cutoff:
                    continue
            matches.append(entry)

    matches = matches[-limit:]

    if not matches:
        console.print("[dim]No audit entries matched.[/]")
        return

    if output_format == "jsonl":
        for entry in matches:
            console.print_json(data=entry)
        return
    if output_format == "json":
        console.print_json(data=matches)
        return

    table = Table(title=f"Audit log ({len(matches)} entries)", show_lines=False)
    table.add_column("Time", style="dim", no_wrap=True)
    table.add_column("Query", no_wrap=True)
    table.add_column("Event", style="cyan")
    table.add_column("Section", no_wrap=True)
    table.add_column("Tool", no_wrap=True)
    table.add_column("Details", style="dim", overflow="fold")

    for entry in matches:
        ts = (entry.get("timestamp") or "")[:19].replace("T", " ")
        qid = (entry.get("query_id") or "")[:12]
        evt = entry.get("event") or ""
        sec = entry.get("section") or ",".join(entry.get("routed_sections") or [])
        tool_name = entry.get("tool") or ""
        details = entry.get("args_summary") or entry.get("error") or ""
        if not details and entry.get("duration_ms") is not None:
            details = f"{entry['duration_ms']}ms"
        table.add_row(ts, qid, evt, sec, tool_name, str(details)[:120])

    console.print(table)


def _parse_since(spec: str):
    """Parse --since as ISO timestamp or relative (e.g. '1h', '30m', '2d')."""
    from datetime import datetime, timedelta, timezone

    spec = spec.strip()
    if not spec:
        return None

    # Relative: <number><unit> where unit in s/m/h/d
    if spec[-1].lower() in {"s", "m", "h", "d"} and spec[:-1].isdigit():
        n = int(spec[:-1])
        unit = spec[-1].lower()
        delta = {
            "s": timedelta(seconds=n),
            "m": timedelta(minutes=n),
            "h": timedelta(hours=n),
            "d": timedelta(days=n),
        }[unit]
        return datetime.now(timezone.utc) - delta

    try:
        when = datetime.fromisoformat(spec.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    db_path: str = typer.Option("~/.coral/vectordb", help="Vector DB path"),
):
    """Index documents into CORAL's RAG knowledge base."""
    from pathlib import Path as P

    from coral.rag.indexer import CoralIndexer

    indexer = CoralIndexer(db_path=db_path)
    p = P(path)

    if p.is_file():
        count = indexer.index_file(str(p))
        console.print(f"[green]Indexed {p.name}: {count} chunks[/]")
    elif p.is_dir():
        count = indexer.index_directory(str(p))
        console.print(f"[green]Indexed {p}: {count} total chunks[/]")
    else:
        console.print(f"[red]Path not found: {path}[/]")


@app.command(name="index-workflow")
def index_workflow(
    workflow_dir: str = typer.Argument(..., help="Path to nos-workflow repo"),
    db_path: str = typer.Option("", help="Vector DB path (default: auto-detect)"),
):
    """Index NOS workflow configs, scripts, and ecFlow definitions for RAG search."""
    from pathlib import Path as P

    workflow = P(workflow_dir)
    if not workflow.is_dir():
        console.print(f"[red]Directory not found: {workflow_dir}[/]")
        raise typer.Exit(1)

    # Auto-detect DB path
    if not db_path:
        db_path = os.environ.get("CORAL_VECTORDB", "~/.coral/vectordb")

    from coral.rag.indexer import CoralIndexer

    indexer = CoralIndexer(db_path=db_path)
    total = 0

    # Index in priority order
    dirs_to_index = [
        ("YAML configs", workflow / "parm"),
        ("ecFlow definitions", workflow / "ecf"),
        ("Execution scripts", workflow / "scripts"),
        ("Shell utilities", workflow / "ush" / "nosofs"),
        ("Fix files", workflow / "fix"),
        ("Python package", workflow / "ush" / "python" / "nos_ofs"),
    ]

    for label, path in dirs_to_index:
        if path.is_dir():
            with Status(f"🪸 [cyan]Indexing {label}...[/]", console=console, spinner="dots"):
                count = indexer.index_directory(str(path))
            console.print(f"  [green]{label}[/]: {count} chunks")
            total += count
        else:
            console.print(f"  [dim]{label}[/]: skipped (not found)")

    # Also index README
    readme = workflow / "README.md"
    if readme.is_file():
        count = indexer.index_file(str(readme))
        console.print(f"  [green]README[/]: {count} chunks")
        total += count

    console.print(f"\n[bold green]Indexed {total} total chunks from {workflow_dir}[/]")


@app.command(name="eval")
def eval_cmd(
    tasks: str = typer.Option("eval/tasks.jsonl", "--tasks", help="Benchmark task set (JSONL)"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
    mode: str = typer.Option("multi", help="Agent mode: 'multi' or 'single' (run both to ablate)"),
    model: str = typer.Option("", help="Ollama model name (default: CORAL_MODEL or qwen3:32b)"),
    output_format: str = typer.Option("table", "--format", help="table | json"),
    save: str = typer.Option("", "--save", help="Write full results JSON to this path"),
):
    """Run the evaluation benchmark through CORAL and report routing/task metrics.

    Run once with --mode single and once with --mode multi to produce the
    single-vs-multi ablation. Requires a reachable Ollama and MCP servers.
    """
    from coral.eval import aggregate, make_agent_run_fn, render_table, results_to_json, run_suite
    from coral.eval.tasks import load_tasks
    from coral.mcp_bridge import MCPBridge

    set_cli_model(model)
    resolved_model = get_model()
    task_list = load_tasks(tasks)
    console.print(f"[dim]Loaded {len(task_list)} tasks from {tasks}[/]")

    async def run():
        logging.getLogger("mcp").setLevel(logging.WARNING)
        _auto_detect_ollama()
        bridge = MCPBridge(config)
        with Status("🪸 [cyan]Connecting to MCP servers...[/]", console=console, spinner="dots"):
            await bridge.connect_all()

        if mode == "single":
            from coral.agent import CoralAgent

            agent = CoralAgent(model=resolved_model, mcp_bridge=bridge)
        else:
            from coral.agents.orchestrator import create_orchestrator

            agent = create_orchestrator(model=resolved_model, mcp_bridge=bridge)

        run_fn = make_agent_run_fn(agent)
        results = []
        with Status("🪸 [cyan]Running benchmark...[/]", console=console, spinner="dots") as status:
            for i, task in enumerate(task_list, 1):
                status.update(f"🪸 [cyan]Running {i}/{len(task_list)}: {task.id}[/]")
                results.extend(await run_suite([task], run_fn))

        agg = aggregate(results)
        if output_format == "json":
            console.print_json(data=results_to_json(results, agg))
        else:
            render_table(agg, console)
        if save:
            from pathlib import Path as _Path

            _Path(save).write_text(json.dumps(results_to_json(results, agg), indent=2), encoding="utf-8")
            console.print(f"[green]Saved results to {save}[/]")

        await bridge.close()

    asyncio.run(run())


@app.command()
def serve(
    model: str = typer.Option("", help="Ollama model name (default: CORAL_MODEL or qwen3:32b)"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
    port: int = typer.Option(7860, help="Web UI port"),
    mode: str = typer.Option("multi", help="Agent mode: 'multi' (orchestrator) or 'single' (legacy)"),
):
    """Launch CORAL web UI."""
    from coral.web_ui import launch

    set_cli_model(model)
    resolved_model = get_model()
    launch(model=resolved_model, config=config, port=port, mode=mode)


if __name__ == "__main__":
    app()
