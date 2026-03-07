"""Base agent class with Ollama tool-calling loop."""

from __future__ import annotations

import logging
from typing import Callable

import ollama

from coral.audit import section_context
from coral.mcp_bridge import MCPBridge

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10


def _truncate_result(result_str: str) -> str:
    """Truncate large tool responses to avoid overwhelming the model."""
    lines = result_str.split("\n")
    if len(lines) > 80:
        kept = lines[:40] + ["\n... [truncated middle rows] ...\n"] + lines[-20:]
        return "\n".join(kept)
    if len(result_str) <= 8000:
        return result_str
    return result_str[:4000] + "\n\n... [truncated] ...\n\n" + result_str[-3000:]


class BaseAgent:
    """An agent with a specific system prompt and subset of tools."""

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        mcp_bridge: MCPBridge,
        tool_filter: list[str] | None = None,
        on_tool_call: Callable | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.mcp_bridge = mcp_bridge
        self.tool_filter = tool_filter  # List of server names this agent can use
        self.on_tool_call = on_tool_call
        self.history: list[dict] = []

    @property
    def tools(self) -> list[dict]:
        """Return only the tools this agent is allowed to use."""
        if self.tool_filter is None:
            return self.mcp_bridge.tools
        return [
            t for t in self.mcp_bridge.tools
            if self.mcp_bridge.tool_server_map.get(t["function"]["name"]) in self.tool_filter
        ]

    async def chat(self, user_message: str) -> str:
        """Run the agent loop: user message -> tool calls -> response."""
        with section_context(self.name):
            self.history.append({"role": "user", "content": user_message})

            messages = [{"role": "system", "content": self.system_prompt}] + self.history
            tools = self.tools

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
            )

            iteration = 0
            while response.message.tool_calls and iteration < MAX_TOOL_ITERATIONS:
                self.history.append({
                    "role": "assistant",
                    "content": response.message.content or "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response.message.tool_calls
                    ],
                })

                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    logger.info("[%s] Calling tool: %s(%s)", self.name, tool_name, tool_args)

                    try:
                        result = await self.mcp_bridge.call_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error calling {tool_name}: {e}"
                        logger.error(result)

                    if self.on_tool_call:
                        self.on_tool_call(tool_name, tool_args, result)

                    result_str = _truncate_result(str(result))
                    self.history.append({"role": "tool", "content": result_str})

                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}] + self.history,
                    tools=tools if tools else None,
                )
                iteration += 1

            assistant_content = response.message.content or ""
            self.history.append({"role": "assistant", "content": assistant_content})
            return assistant_content

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
