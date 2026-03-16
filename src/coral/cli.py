"""CORAL CLI: interactive chat, web UI server, and RAG indexing."""

from __future__ import annotations

import asyncio

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from coral.config import get_all_model_assignments, get_model, set_cli_model

CORAL_BANNER = r"""
   ██████╗ ██████╗ ██████╗  █████╗ ██╗
  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ╚██████╗╚██████╔╝██║  ██║██║  ██║███████╗
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
"""

app = typer.Typer(
    name="coral",
    help="CORAL - Coastal Ocean Research AI Layer",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    model: str = typer.Option("", help="Ollama model name (default: CORAL_MODEL or qwen3:32b)"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
    mode: str = typer.Option("multi", help="Agent mode: 'multi' (orchestrator) or 'single' (legacy)"),
):
    """Interactive chat with CORAL."""
    from coral.mcp_bridge import MCPBridge

    # Register CLI model into the central resolver. All get_model() calls
    # now see it as tier 3 in the fallback chain.
    set_cli_model(model)
    resolved_model = get_model()

    async def run():
        bridge = MCPBridge(config)

        # Show banner
        banner = Text(CORAL_BANNER, style="bold cyan")
        console.print(banner, highlight=False)

        console.print("[dim]Connecting to MCP servers...[/]")
        await bridge.connect_all()
        tool_count = len(bridge.tools)

        if mode == "single":
            from coral.agent import CoralAgent

            def on_tool_call(name, args, result):
                args_short = str(args)
                if len(args_short) > 80:
                    args_short = args_short[:80] + "..."
                result_short = str(result)
                if len(result_short) > 200:
                    result_short = result_short[:200] + "..."
                console.print(f"  [yellow]Tool:[/] {name}({args_short})")
                console.print(f"  [dim]{result_short}[/]")

            agent = CoralAgent(model=resolved_model, mcp_bridge=bridge, on_tool_call=on_tool_call)
            mode_label = "single agent"
        else:
            from coral.agents.orchestrator import create_orchestrator

            agent = create_orchestrator(model=resolved_model, mcp_bridge=bridge)
            mode_label = "multi-agent (data + code + workflow)"

        # Status panel
        import os
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
        console.print(Panel(status, border_style="cyan", padding=(0, 1)))
        console.print("[dim]Type 'exit' or 'quit' to leave. Ctrl+C to interrupt.[/]\n")

        session: PromptSession[str] = PromptSession()
        try:
            while True:
                try:
                    user_input = await session.prompt_async(HTML("<b>You: </b>"))
                except EOFError:
                    break

                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue

                console.print("[dim]Thinking...[/]")
                try:
                    response = await agent.chat(user_input)
                    console.print()
                    console.print(Rule(style="cyan"))
                    console.print(f"[bold cyan]CORAL:[/]")
                    console.print(Markdown(response))
                    console.print(Rule(style="dim"))
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
    from pathlib import Path

    from coral.rag.indexer import CoralIndexer

    indexer = CoralIndexer(db_path=db_path)
    p = Path(path)

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
