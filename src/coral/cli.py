"""CORAL CLI: interactive chat, web UI server, and RAG indexing."""

from __future__ import annotations

import asyncio
import json
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
    "/status", "/report",
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
  [cyan]/help[/]         Show this help
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

    return False


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
        bridge = MCPBridge(config)
        memory = CoralMemory()
        chat_log: list[dict] = []

        # Auto-detect Ollama from coral_host.env
        _auto_detect_ollama()

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
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/]")
        finally:
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
