"""Core CORAL agent: chat loop with tool calling via Ollama + MCP."""

from __future__ import annotations

import logging
import re
from typing import Callable

import ollama

from coral.mcp_bridge import MCPBridge
from coral.prompts import CORAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10

# Keyword → MCP server prefix mapping for tool filtering.
# Each entry: (keywords_in_query, tool_name_prefixes_to_include)
_TOOL_ROUTES: list[tuple[list[str], list[str]]] = [
    (["water level", "tide", "tidal", "datum", "station", "currents", "flood", "sea level"],
     ["coops_"]),
    (["hurricane", "tropical", "cyclone", "storm track", "nhc"],
     ["nhc_"]),
    (["surge", "stofs", "storm surge"],
     ["stofs_", "coops_"]),
    (["recon", "reconnaissance", "flight", "hunter", "hdob", "sfmr", "vdm"],
     ["recon_"]),
    (["satellite", "sst", "chlorophyll", "erddap", "ocean color"],
     ["erddap_"]),
    (["ofs", "forecast model", "operational forecast"],
     ["ofs_"]),
    (["compare", "vs", "versus", "observation"],
     ["coops_", "stofs_", "ofs_"]),
    (["documentation", "source code", "subroutine", "function", "module", "namelist",
      "schism", "adcirc", "ufs", "fortran", "what does", "how does", "explain", "config"],
     ["search_documentation"]),
    (["netcdf", ".nc", "inspect", "variable", "dimensions", "xarray"],
     ["inspect_netcdf", "query_netcdf", "get_netcdf_timeseries"]),
    (["slurm", "job", "sbatch", "squeue", "sacct", "failed job", "job log"],
     ["get_my_jobs", "get_job_details", "read_job_log", "diagnose_job_failure"]),
    (["ecflow", "suite", "aborted", "task status", "ecf"],
     ["get_suite_status", "get_aborted_tasks", "read_ecflow_job_output"]),
    (["plot", "figure", "visuali", "matplotlib", "execute python", "run python", "script", "code"],
     ["execute_python"]),
]


def _select_tools(query: str, all_tools: list[dict]) -> list[dict]:
    """Select a subset of tools relevant to the user's query.

    With many tools, smaller models get confused. This narrows the set
    based on keyword matching so the LLM sees only relevant tools.
    Falls back to all tools if no keywords match.
    """
    query_lower = query.lower()
    prefixes: set[str] = set()

    for keywords, tool_prefixes in _TOOL_ROUTES:
        if any(kw in query_lower for kw in keywords):
            prefixes.update(tool_prefixes)

    if not prefixes:
        return all_tools

    filtered = [
        t for t in all_tools
        if any(t["function"]["name"].startswith(p) for p in prefixes)
    ]
    return filtered if filtered else all_tools


class CoralAgent:
    """Chat agent that uses Ollama for LLM inference and MCP for tool execution."""

    def __init__(self, model: str, mcp_bridge: MCPBridge, on_tool_call: Callable | None = None):
        self.model = model
        self.mcp = mcp_bridge
        self.history: list[dict] = []
        self.on_tool_call = on_tool_call  # callback(tool_name, tool_args, result)

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent response.

        Implements an agentic loop: the LLM can make multiple tool calls
        before producing a final text response.
        """
        self.history.append({"role": "user", "content": user_message})

        # Filter tools to relevant subset for this query
        tools = _select_tools(user_message, self.mcp.tools)
        logger.info("Selected %d/%d tools for query", len(tools), len(self.mcp.tools))

        messages = [{"role": "system", "content": CORAL_SYSTEM_PROMPT}] + self.history

        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
        )

        iteration = 0
        while response.message.tool_calls and iteration < MAX_TOOL_ITERATIONS:
            # Append the assistant message with tool calls
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

            # Execute each tool call
            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                logger.info("Calling tool: %s(%s)", tool_name, tool_args)

                try:
                    result = await self.mcp.call_tool(tool_name, tool_args)
                except Exception as e:
                    result = f"Error calling {tool_name}: {e}"
                    logger.error(result)

                if self.on_tool_call:
                    self.on_tool_call(tool_name, tool_args, result)

                # Truncate very large tool responses to avoid overwhelming small models.
                # For tabular data, keep header + sampled rows to preserve key info.
                result_str = str(result)
                if len(result_str) > 8000:
                    lines = result_str.split("\n")
                    # Keep first 40 lines (header + early data) and last 20 lines
                    if len(lines) > 80:
                        kept = lines[:40] + ["\n... [truncated middle rows] ...\n"] + lines[-20:]
                        result_str = "\n".join(kept)
                    else:
                        result_str = result_str[:4000] + "\n\n... [truncated] ...\n\n" + result_str[-3000:]

                self.history.append({
                    "role": "tool",
                    "content": result_str,
                })

            # Subsequent rounds use all tools (the model may need to cross-reference)
            messages = [{"role": "system", "content": CORAL_SYSTEM_PROMPT}] + self.history
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
            )
            iteration += 1

        # Final text response
        assistant_content = response.message.content or ""
        self.history.append({"role": "assistant", "content": assistant_content})
        return assistant_content

    def reset(self):
        """Clear conversation history."""
        self.history.clear()
