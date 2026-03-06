"""CORAL CLI: interactive chat, web UI server, and RAG indexing."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

app = typer.Typer(
    name="coral",
    help="CORAL - Coastal Ocean Research AI Layer",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    model: str = typer.Option("qwen3:32b", help="Ollama model name"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
):
    """Interactive chat with CORAL."""
    from coral.agent import CoralAgent
    from coral.mcp_bridge import MCPBridge

    async def run():
        bridge = MCPBridge(config)
        console.print("[bold cyan]CORAL - Connecting to MCP servers...[/]")
        await bridge.connect_all()
        tool_names = [t["function"]["name"] for t in bridge.tools]
        console.print(f"[green]Connected. {len(bridge.tools)} tools available.[/]")
        console.print(f"[dim]Tools: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}[/]\n")

        def on_tool_call(name, args, result):
            args_short = str(args)
            if len(args_short) > 80:
                args_short = args_short[:80] + "..."
            result_short = str(result)
            if len(result_short) > 200:
                result_short = result_short[:200] + "..."
            console.print(f"  [yellow]Tool:[/] {name}({args_short})")
            console.print(f"  [dim]{result_short}[/]")

        agent = CoralAgent(model=model, mcp_bridge=bridge, on_tool_call=on_tool_call)

        console.print("[dim]Type 'exit' or 'quit' to leave. Ctrl+C to interrupt.[/]\n")

        try:
            while True:
                try:
                    user_input = console.input("[bold]You:[/] ")
                except EOFError:
                    break

                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue

                console.print("[dim]Thinking...[/]")
                response = await agent.chat(user_input)
                console.print(f"\n[bold cyan]CORAL:[/] {response}\n")
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/]")
        finally:
            await bridge.close()

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
    model: str = typer.Option("qwen3:32b", help="Ollama model name"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
    port: int = typer.Option(7860, help="Web UI port"),
):
    """Launch CORAL web UI."""
    from coral.web_ui import launch

    launch(model=model, config=config, port=port)


if __name__ == "__main__":
    app()
