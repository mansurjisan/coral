"""Gradio web UI for CORAL."""

from __future__ import annotations

import asyncio

import gradio as gr

from coral.agent import CoralAgent
from coral.mcp_bridge import MCPBridge

_agent: CoralAgent | None = None


async def _init_agent(model: str, config: str):
    global _agent
    bridge = MCPBridge(config)
    await bridge.connect_all()
    _agent = CoralAgent(model=model, mcp_bridge=bridge)


async def _respond(message: str, history: list):
    """Handle a chat message from the Gradio UI."""
    response = await _agent.chat(message)
    return response


def launch(model: str, config: str, port: int):
    """Initialize the agent and launch the Gradio web interface."""
    asyncio.run(_init_agent(model, config))

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
