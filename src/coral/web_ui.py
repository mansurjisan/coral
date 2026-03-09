"""Gradio web UI for CORAL."""

from __future__ import annotations

import gradio as gr

from coral.agent import CoralAgent
from coral.agents.orchestrator import create_orchestrator
from coral.mcp_bridge import MCPBridge

_bridge: MCPBridge | None = None
_model: str = ""
_config: str = ""
_mode: str = "multi"


async def _ensure_bridge():
    """Lazily initialize the MCP bridge on first request.

    This runs inside Gradio's event loop so MCP sessions stay alive.
    Using asyncio.run() for init would destroy the event loop and kill
    all MCP connections before Gradio starts.
    """
    global _bridge
    if _bridge is not None:
        return

    print("CORAL — Connecting to MCP servers...")
    _bridge = MCPBridge(_config)
    await _bridge.connect_all()
    print(f"Connected. {len(_bridge.tools)} tools available.")


def _create_session_agent():
    """Create a new agent for a Gradio session based on the configured mode."""
    if _mode == "single":
        return CoralAgent(model=_model, mcp_bridge=_bridge)
    return create_orchestrator(model=_model, mcp_bridge=_bridge)


async def _respond(message: str, history: list, session_agent):
    """Handle a chat message from the Gradio UI.

    Each browser session gets its own agent (own conversation history)
    but they all share the same MCP bridge (expensive to create per user).
    """
    await _ensure_bridge()

    if session_agent is None:
        session_agent = _create_session_agent()

    response = await session_agent.chat(message)
    return response, session_agent


def launch(model: str, config: str, port: int, mode: str = "multi"):
    """Launch the Gradio web interface."""
    global _model, _config, _mode
    _model = model
    _config = config
    _mode = mode

    with gr.Blocks(title="CORAL - Coastal Ocean Research AI Layer") as demo:
        gr.Markdown("# CORAL - Coastal Ocean Research AI Layer")
        gr.Markdown("Ask questions about NOAA ocean data, model code, HPC workflows.")

        # Per-session agent state (invisible to user)
        session_agent = gr.State(None)

        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Type your question...", show_label=False)

        gr.Examples(
            examples=[
                "What is the current water level at The Battery, NYC?",
                "Are there any active hurricanes in the Atlantic?",
                "Compare STOFS surge forecast vs observations at Tampa Bay",
                "What does the subroutine schism_init do?",
                "Show me my last failed Slurm job and explain the error",
            ],
            inputs=msg,
        )

        async def user_submit(message, chat_history, agent):
            chat_history = chat_history + [
                {"role": "user", "content": message},
            ]
            return "", chat_history, agent

        async def bot_respond(chat_history, agent):
            user_message = chat_history[-1]["content"]
            response, agent = await _respond(user_message, chat_history, agent)
            chat_history = chat_history + [
                {"role": "assistant", "content": response},
            ]
            return chat_history, agent

        msg.submit(
            user_submit, [msg, chatbot, session_agent], [msg, chatbot, session_agent]
        ).then(
            bot_respond, [chatbot, session_agent], [chatbot, session_agent]
        )

    demo.launch(server_name="0.0.0.0", server_port=port)
