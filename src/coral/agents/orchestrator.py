"""Orchestrator: routes user queries to specialized agents."""

from __future__ import annotations

import logging

import ollama

from coral.agents.base import _prune_history
from coral.audit import record_audit_event, request_context, set_request_route
from coral.agents.code_agent import create_code_agent
from coral.agents.data_agent import create_data_agent
from coral.agents.workflow_agent import create_workflow_agent
from coral.mcp_bridge import MCPBridge

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are a query router for CORAL, an AI system for NOAA ocean scientists.

Given a user query, classify it into one or more categories. Respond with ONLY the \
category names, comma-separated. No explanation.

Categories:
- DATA: Questions about ocean observations, water levels, tides, hurricanes, storm surge \
forecasts, satellite data, or querying NetCDF model output files
- CODE: Questions about model source code (SCHISM, ADCIRC, UFS-Coastal, Fortran, C), \
documentation, namelists, parameter meanings, or requests to generate/run Python code and plots
- WORKFLOW: Questions about Slurm jobs, ecFlow suites, job failures, log files, \
or HPC system status

Examples:
- "What is the water level at The Battery?" -> DATA
- "What does schism_init do?" -> CODE
- "Why did my Slurm job fail?" -> WORKFLOW
- "My STOFS run failed, check the log and explain the error from the docs" -> WORKFLOW,CODE
- "Plot the water levels from this NetCDF file" -> DATA,CODE
- "Compare STOFS forecast against CO-OPS observations and plot the difference" -> DATA,CODE
"""

_DATA_KEYWORDS = [
    "water level", "water levels", "tide", "tidal", "hurricane", "tropical",
    "cyclone", "storm surge", "surge", "sst", "erddap", "satellite",
    "station", "coops", "co-ops", "netcdf", ".nc", "recon", "buoy",
    "wave", "wave height", "ww3", "wind observations", "weather station",
    "usgs", "streamflow", "goes", "sea level", "currents", "datum",
    "flood", "river", "discharge", "observation", "observations",
    "forecast at", "forecast for", "forecast against",
]
_CODE_KEYWORDS = [
    "subroutine", "function", "module", "namelist", "param.nml",
    "fortran", "source code", "what does", "how does", "explain",
    "plot", "matplotlib", "execute python", "run python", "visuali",
    "code", "script", "figure", "algorithm", "docs", "documentation",
    "parameter", "parameters",
]
_WORKFLOW_KEYWORDS = [
    "slurm", "job", "jobs", "sbatch", "salloc", "squeue", "sacct",
    "ecflow", "suite", "aborted", "task", "tasks", "task status",
    "failed", "failure", "log", "logs", "job output", "recent jobs",
    "scratch", "quota",
]
_AMBIGUOUS_MODEL_TERMS = ["stofs", "schism", "adcirc", "forecast", "run"]


def _matched_keywords(query_lower: str, keywords: list[str]) -> list[str]:
    """Return keywords that appear in the query."""
    return [kw for kw in keywords if kw in query_lower]


def _ordered_categories(
    has_workflow: bool,
    has_data: bool,
    has_code: bool,
) -> list[str]:
    """Return categories in the execution order CORAL should use."""
    if has_workflow and has_data and has_code:
        return ["WORKFLOW", "DATA", "CODE"]
    if has_workflow and has_code:
        return ["WORKFLOW", "CODE"]
    if has_workflow and has_data:
        return ["WORKFLOW", "DATA"]
    if has_data and has_code:
        return ["DATA", "CODE"]
    if has_workflow:
        return ["WORKFLOW"]
    if has_data:
        return ["DATA"]
    if has_code:
        return ["CODE"]
    return []


def _keyword_classify(query: str) -> list[str] | None:
    """Fast intent-based classification. Returns None if ambiguous."""
    query_lower = query.lower().strip()
    if not query_lower:
        return None

    data_matches = _matched_keywords(query_lower, _DATA_KEYWORDS)
    code_matches = _matched_keywords(query_lower, _CODE_KEYWORDS)
    workflow_matches = _matched_keywords(query_lower, _WORKFLOW_KEYWORDS)
    ambiguous_matches = _matched_keywords(query_lower, _AMBIGUOUS_MODEL_TERMS)

    has_data = bool(data_matches)
    has_code = bool(code_matches)
    has_workflow = bool(workflow_matches)

    # Ambiguous model names alone should not force a route.
    if ambiguous_matches and not (has_data or has_code or has_workflow):
        return None

    categories = _ordered_categories(has_workflow, has_data, has_code)
    return categories or None


SYNTHESIS_PROMPT = """\
You are CORAL, an AI assistant for NOAA ocean scientists. \
Synthesize the provided information into a clear, unified response. \
Do not mention 'agents' or internal routing."""


class Orchestrator:
    """Routes queries to specialized agents and synthesizes responses."""

    def __init__(self, model: str, mcp_bridge: MCPBridge):
        self.model = model
        self.mcp_bridge = mcp_bridge
        self.agents = {
            "DATA": create_data_agent(model, mcp_bridge),
            "CODE": create_code_agent(model, mcp_bridge),
            "WORKFLOW": create_workflow_agent(model, mcp_bridge),
        }
        self.history: list[dict] = []

    async def classify(self, query: str) -> list[str]:
        """Classify user query into agent categories.

        Uses fast keyword matching first, falls back to LLM for ambiguous queries.
        """
        # Try keyword route first
        result = _keyword_classify(query)
        if result is not None:
            logger.info("Keyword-routed query to: %s", result)
            return result

        # Fall back to LLM classification
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        raw = response.message.content.strip().upper()

        categories = []
        for cat in ["DATA", "CODE", "WORKFLOW"]:
            if cat in raw:
                categories.append(cat)

        if not categories:
            logger.warning("Could not classify query, defaulting to DATA: %s", raw)
            categories = ["DATA"]

        logger.info("LLM-routed query to: %s", categories)
        return categories

    async def chat(self, user_message: str) -> str:
        """Route query to appropriate agent(s) and combine responses."""
        with request_context(mode="multi"):
            record_audit_event("query_start", message_chars=len(user_message))
            try:
                self.history.append({"role": "user", "content": user_message})
                _prune_history(self.history)

                categories = await self.classify(user_message)
                set_request_route(categories)
                record_audit_event("route_decision", routed_sections=categories)

                if len(categories) == 1:
                    agent = self.agents[categories[0]]
                    response = await agent.chat(user_message)
                else:
                    # Multi-agent: sequential execution, pass context forward
                    responses = []
                    accumulated_context = user_message

                    for cat in categories:
                        agent = self.agents[cat]
                        try:
                            result = await agent.chat(accumulated_context)
                        except Exception as agent_exc:
                            logger.error("Agent %s failed: %s", cat, agent_exc)
                            record_audit_event(
                                "agent_error", section=cat, error=str(agent_exc),
                            )
                            result = f"[{cat} section unavailable: {agent_exc}]"
                        responses.append(f"[{agent.name.upper()} AGENT]\n{result}")

                        accumulated_context = (
                            f"Original question: {user_message}\n\n"
                            f"Previous findings:\n{result}\n\n"
                            f"Based on the above, continue addressing the original question."
                        )
                        agent.clear_history()

                    response = await self._synthesize(user_message, responses)

                self.history.append({"role": "assistant", "content": response})
                record_audit_event("query_end", success=True)
                return response
            except Exception as exc:
                record_audit_event("query_end", success=False, error=str(exc))
                raise

    async def _synthesize(self, original_query: str, agent_responses: list[str]) -> str:
        """Combine multiple agent responses into a coherent answer."""
        combined = "\n\n".join(agent_responses)

        synthesis_prompt = (
            f"The user asked: {original_query}\n\n"
            f"Multiple specialized agents provided these findings:\n\n"
            f"{combined}\n\n"
            f"Synthesize these into a single, coherent response for the user. "
            f"Don't mention 'agents' -- just provide the unified answer."
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": synthesis_prompt},
            ],
        )
        return response.message.content

    def reset(self):
        """Clear all agent histories."""
        self.history.clear()
        for agent in self.agents.values():
            agent.clear_history()
