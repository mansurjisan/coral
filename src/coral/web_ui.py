"""Gradio web UI for CORAL."""

from __future__ import annotations

import gradio as gr

from coral.agent import CoralAgent
from coral.mcp_bridge import MCPBridge

_agent: CoralAgent | None = None
_model: str = ""
_config: str = ""


async def _ensure_agent():
    """Lazily initialize the agent on first request.

    This runs inside Gradio's event loop so MCP sessions stay alive.
    Using asyncio.run() for init would destroy the event loop and kill
    all MCP connections before Gradio starts.
    """
    global _agent
    if _agent is not None:
        return

    print("CORAL — Connecting to MCP servers...")
    bridge = MCPBridge(_config)
    await bridge.connect_all()
    print(f"Connected. {len(bridge.tools)} tools available.")
    _agent = CoralAgent(model=_model, mcp_bridge=bridge)


async def _respond(message: str, history: list):
    """Handle a chat message from the Gradio UI."""
    await _ensure_agent()
    response = await _agent.chat(message)
    return response


def launch(model: str, config: str, port: int):
    """Launch the Gradio web interface."""
    global _model, _config
    _model = model
    _config = config

    demo = gr.ChatInterface(
        _respond,
        title="CORAL - Coastal Ocean Research AI Layer",
        description="Ask questions about NOAA ocean data, model code, HPC workflows.",
        examples=[
            "What is the current water level at The Battery, NYC?",
            "Are there any active hurricanes in the Atlantic?",
            "Compare STOFS surge forecast vs observations at Tampa Bay",
            "What does the subroutine schism_init do?",
            "Show me my last failed Slurm job and explain the error",
        ],
    )
    demo.launch(server_name="0.0.0.0", server_port=port)
