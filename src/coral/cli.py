"""CORAL CLI: interactive chat, web UI server, and RAG indexing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

_SLASH_COMMANDS = [
    "/help", "/clear", "/reset", "/mode", "/save",
    "/memory", "/remember", "/forget", "/tools",
    "/status", "/report", "/watch", "/audit", "/techmemo",
]
_slash_completer = WordCompleter(_SLASH_COMMANDS, sentence=True)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.text import Text

from coral.config import get_all_model_assignments, get_model, set_cli_model

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
    candidates = [
        Path(f"/scratch5/purged/{user}/coral_host.env"),
        Path.home() / "coral_host.env",
        Path("coral_host.env"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                for line in candidate.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("OLLAMA_NODE="):
                        node = line.split("=", 1)[1].strip()
                        os.environ["OLLAMA_HOST"] = f"http://{node}:11434"
                        console.print(
                            f"[dim]Auto-detected Ollama at {node}:11434 "
                            f"(from {candidate})[/]"
                        )
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
  [cyan]/save[/]         Save conversation to markdown file
  [cyan]/memory[/]       Show saved memories
  [cyan]/remember[/]     Save a memory (e.g. /remember account = coastal-act)
  [cyan]/forget[/]       Remove a memory (e.g. /forget account)
  [cyan]/tools[/]        Show tool count per section
  [cyan]/status[/]       Quick dashboard: jobs, quota, Ollama health
  [cyan]/report[/]       Generate HPC status report as markdown
  [cyan]/watch[/]        Watch a job (e.g. /watch 9848988)
  [cyan]/audit[/]        Show tool call history and stats
  [cyan]/techmemo[/]     Auto-generate NOAA tech memo draft
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
        job_id = arg.strip()
        if not job_id or not job_id.isdigit():
            console.print("[dim]Usage: /watch <job_id>  (e.g. /watch 9848988)[/]")
            return True
        _start_job_watcher(job_id, console)
        return True

    if command == "/audit":
        _show_audit(console)
        return True

    if command == "/techmemo":
        await _generate_techmemo(agent, console)
        return True

    return False


# Tool call audit log (populated by the tool callback)
_audit_log: list[dict] = []


def _show_audit(console: Console) -> None:
    """Show tool call history and stats."""
    if not _audit_log:
        console.print("[dim]No tool calls recorded yet.[/]")
        return

    lines = ["[bold]Tool Call History:[/]\n"]
    for entry in _audit_log[-20:]:  # Last 20 calls
        lines.append(
            f"  [cyan]{entry['tool']:30s}[/] "
            f"[dim]{entry.get('elapsed', '?')}[/]"
        )

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

    # Running jobs (via agent if available)
    if hasattr(agent, "agents") and "WORKFLOW" in agent.agents:
        try:
            import subprocess
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h",
                 "-o", "%i %j %T %M"],
                capture_output=True, text=True, timeout=10,
            )
            jobs = result.stdout.strip().split("\n") if result.stdout.strip() else []
            running = [j for j in jobs if "RUNNING" in j]
            pending = [j for j in jobs if "PENDING" in j]
            sections.append(f"  [green]✓[/] Slurm — {len(running)} running, {len(pending)} pending")
            for job in running[:5]:
                sections.append(f"    [dim]{job}[/]")
        except Exception:
            sections.append("  [yellow]?[/] Slurm — not available")
    else:
        sections.append("  [dim]-[/] Slurm — not in current mode")

    # Disk usage summary
    user = os.environ.get("USER", "")
    scratch5 = f"/scratch5/purged/{user}"
    if os.path.isdir(scratch5):
        try:
            import subprocess
            result = subprocess.run(
                ["du", "-sh", scratch5],
                capture_output=True, text=True, timeout=30,
            )
            size = result.stdout.strip().split()[0] if result.stdout.strip() else "?"
            sections.append(f"  [green]✓[/] Scratch5 — {size} used")
        except Exception:
            sections.append("  [yellow]?[/] Scratch5 — could not check")

    console.print(Panel(
        "\n".join(sections),
        title="[bold cyan]Status[/]",
        border_style="cyan",
        padding=(0, 1),
    ))


def _start_job_watcher(job_id: str, console: Console) -> None:
    """Start a background thread that polls sacct and notifies when job finishes."""
    import re
    import subprocess
    import threading

    def poll():
        console.print(f"[dim]Watching job {job_id}... (will notify when done)[/]")
        poll_interval = 30  # seconds
        max_polls = 480  # 4 hours at 30s intervals

        for _ in range(max_polls):
            import time
            time.sleep(poll_interval)
            try:
                result = subprocess.run(
                    ["sacct", "-j", job_id, "-n", "-X",
                     "--format=State", "--parsable2"],
                    capture_output=True, text=True, timeout=15,
                )
                state = result.stdout.strip().split("\n")[0].strip() if result.stdout.strip() else ""

                if state and state not in ("RUNNING", "PENDING", "REQUEUED", "SUSPENDED", ""):
                    # Job finished
                    console.print(f"\n[bold yellow]🔔 Job {job_id} finished: {state}[/]")
                    # Get more details
                    detail = subprocess.run(
                        ["sacct", "-j", job_id, "-n", "-X",
                         "--format=JobName%30,State,ExitCode,Elapsed,MaxRSS"],
                        capture_output=True, text=True, timeout=15,
                    )
                    if detail.stdout.strip():
                        console.print(f"[dim]  {detail.stdout.strip()}[/]")
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
            f"# CORAL HPC Status Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{response}\n"
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
        server_count = len(set(
            ag.mcp_bridge.tool_server_map.get(t["function"]["name"], "")
            for ag in agent.agents.values()
            for t in ag.tools
        ))
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
    filename: str,
    console: Console,
) -> None:
    """Save conversation to a markdown file."""
    if not chat_log:
        console.print("[dim]No conversation to save.[/]")
        return

    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coral_chat_{ts}.md"

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
        import time as _t
        args_short = str(args)
        if len(args_short) > 80:
            args_short = args_short[:80] + "..."
        console.print(f"  [yellow]⚡ {name}[/]({args_short})")

        # Record in audit log
        _audit_log.append({
            "tool": name,
            "args": args_short,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

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
        chat_log: list[dict] = []

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

        # Status panel
        user = os.environ.get("USER", "unknown")
        host = os.environ.get("HOSTNAME", os.environ.get("HOST", "local"))
        status = Text.assemble(
            ("  🪸 ", ""),
            ("Model  ", "dim"), (resolved_model, "green"),
            ("  │  ", "dim"),
            ("Tools  ", "dim"), (str(tool_count), "green"),
            ("  │  ", "dim"),
            ("Mode  ", "dim"), (mode_label, "green"),
            ("\n  🖥️  ", ""),
            ("User   ", "dim"), (user, "cyan"),
            ("  │  ", "dim"),
            ("Host   ", "dim"), (host, "cyan"),
        )
        if memory.list_all():
            status.append("\n  🧠 ", style="")
            status.append(f"{len(memory.list_all())} memories loaded", style="dim")
        console.print(Panel(
            status,
            border_style="cyan",
            title="[bold cyan]CORAL v0.1.0[/]",
            subtitle="[dim]Coastal Ocean Research AI Layer · /help for commands[/]",
            padding=(0, 1),
        ))
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
                        stripped, agent, memory, chat_log, console,
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
                    # Run with thinking spinner
                    with Status(
                        "🪸 [cyan]Thinking...[/]",
                        console=console,
                        spinner="dots",
                    ):
                        response = await agent.chat(stripped)
                    elapsed = _time.monotonic() - t0

                    chat_log.append({"role": "assistant", "content": response})
                    console.print()
                    console.print(Rule(style="cyan"))
                    console.print("[bold cyan]CORAL:[/]")
                    console.print(Markdown(response))

                    # Token/timing stats
                    stats_parts = [f"{elapsed:.1f}s"]
                    stats = _get_agent_stats(agent)
                    if stats.get("tokens"):
                        stats_parts.append(f"{stats['tokens']} tokens")
                    if stats.get("tokens_per_sec"):
                        stats_parts.append(f"{stats['tokens_per_sec']} tok/s")
                    console.print(Rule(
                        title=f"[dim]{' · '.join(stats_parts)}[/]",
                        style="dim",
                    ))
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
