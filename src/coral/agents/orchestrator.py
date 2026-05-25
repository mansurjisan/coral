"""Orchestrator: routes user queries to specialized agents."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

import ollama

from coral.agents.base import ASK_SECTION_TOOL, _prune_history
from coral.audit import record_audit_event, request_context, set_request_route
from coral.agents.code_agent import create_code_agent
from coral.agents.data_agent import create_data_agent
from coral.agents.workflow_agent import create_workflow_agent
from coral.mcp_bridge import MCPBridge

logger = logging.getLogger(__name__)

# Maximum number of agents active on the delegation stack at once (the top-level
# routed agent plus any nested delegations). Bounds cost and recursion.
MAX_DELEGATION_DEPTH = 3

# Tracks the chain of sections currently executing so delegate() can reject
# re-entering an agent instance already on the stack (which would corrupt its
# in-progress history) and cap delegation depth.
_DELEGATION_STACK: ContextVar[tuple[str, ...]] = ContextVar("coral_delegation_stack", default=())


@contextmanager
def _delegation_scope(section: str):
    """Push a section onto the delegation stack for the duration of its run."""
    stack = _DELEGATION_STACK.get()
    token = _DELEGATION_STACK.set(stack + (section,))
    try:
        yield
    finally:
        _DELEGATION_STACK.reset(token)


def _delegation_enabled() -> bool:
    """Whether agent-to-agent delegation is active (off via CORAL_DELEGATION)."""
    return os.environ.get("CORAL_DELEGATION", "on").strip().lower() not in {"0", "false", "off", "no"}


def _delegation_prompt(peers: list[str]) -> str:
    """System-prompt note telling an agent how and when to consult peers."""
    peer_list = ", ".join(peers)
    return (
        "\n\nCOLLABORATION:\n"
        f"You can consult a specialized peer section: {peer_list}. "
        f"Call {ASK_SECTION_TOOL}(section, query) only when a request needs information or "
        "actions outside your own tools (e.g. observational data you cannot fetch, or "
        "source/docs you cannot access). Do not delegate work your own tools can do. "
        "Ask one self-contained question — the peer does not see this conversation."
    )


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
HPC system status, disk quotas, allocations, FairShare, modules, storage usage, \
group membership, or setting up/submitting/monitoring UFS-Coastal experiments

Examples:
- "What is the water level at The Battery?" -> DATA
- "What does schism_init do?" -> CODE
- "Why did my Slurm job fail?" -> WORKFLOW
- "Set up a SCHISM coastal experiment and submit it" -> WORKFLOW
- "How much scratch space am I using?" -> WORKFLOW
- "What's my FairShare status?" -> WORKFLOW
- "What modules do I have loaded?" -> WORKFLOW
- "Show my disk quota" -> WORKFLOW
- "What groups am I in?" -> WORKFLOW
- "My STOFS run failed, check the log and explain the error from the docs" -> WORKFLOW,CODE
- "Plot the water levels from this NetCDF file" -> DATA,CODE
- "Compare STOFS forecast against CO-OPS observations and plot the difference" -> DATA,CODE
- "Set up a UFS run and plot the outputs" -> WORKFLOW,CODE
"""

_DATA_KEYWORDS = [
    "water level",
    "water levels",
    "tide",
    "tidal",
    "hurricane",
    "tropical",
    "cyclone",
    "storm surge",
    "surge",
    "sst",
    "erddap",
    "satellite",
    "station",
    "coops",
    "co-ops",
    "netcdf",
    ".nc",
    "recon",
    "buoy",
    "wave",
    "wave height",
    "ww3",
    "wind observations",
    "weather station",
    "usgs",
    "streamflow",
    "goes",
    "sea level",
    "currents",
    "datum",
    "flood",
    "river",
    "discharge",
    "observation",
    "observations",
    "forecast at",
    "forecast for",
    "forecast against",
    "datum",
    "vertical datum",
    "vdatum",
    "navd88",
    "mllw",
    "convert datum",
    "datum conversion",
]
_CODE_KEYWORDS = [
    "subroutine",
    "function",
    "module",
    "namelist",
    "param.nml",
    "fortran",
    "source code",
    "what does",
    "how does",
    "explain",
    "plot",
    "matplotlib",
    "execute python",
    "run python",
    "visuali",
    "code",
    "script",
    "figure",
    "algorithm",
    "docs",
    "documentation",
    "parameter",
    "parameters",
]
_WORKFLOW_KEYWORDS = [
    "slurm",
    "job",
    "jobs",
    "sbatch",
    "salloc",
    "squeue",
    "sacct",
    "ecflow",
    "suite",
    "aborted",
    "task",
    "tasks",
    "task status",
    "failed",
    "failure",
    "log",
    "logs",
    "job output",
    "recent jobs",
    "scratch",
    "quota",
    "experiment",
    "set up experiment",
    "create experiment",
    "submit experiment",
    "run experiment",
    "ufs",
    "ufs-coastal",
    "model run",
    "validate experiment",
    "cancel run",
    "run status",
    "collect outputs",
    "disk usage",
    "disk space",
    "storage",
    "fairshare",
    "fair share",
    "allocation",
    "core hours",
    "core-hours",
    "account info",
    "my account",
    "my groups",
    "group membership",
    "groups am i",
    "module list",
    "module avail",
    "modules loaded",
    "loaded modules",
    "partition",
    "partitions",
    "node info",
    "how much space",
    "disk quota",
    "secofs config",
    "stofs config",
    "nosofs",
    "ofs config",
    "ofs system",
    "ensemble config",
    "forcing config",
    "grid config",
    "compare configs",
    "diagnose failure",
    "fatal error",
    "fatal.error",
    "ecflow suite",
    "workflow config",
    "nos workflow",
    "list ofs",
    "list systems",
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


def _keyword_classify(query: str) -> tuple[list[str], float, list[str]] | None:
    """Fast intent-based classification.

    Returns (categories, confidence, matched_keywords) or None when the query
    has no useful keyword signal. ``matched_keywords`` is the union of every
    term that contributed to the decision, for surface in ``/route``.
    """
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
    if not categories:
        return None

    # Compute confidence: more keyword matches = higher confidence
    total_matches = len(data_matches) + len(code_matches) + len(workflow_matches)
    # Base confidence from keyword count, capped at 0.95
    confidence = min(0.95, 0.6 + (total_matches - 1) * 0.1)
    # Reduce confidence for multi-category routing
    if len(categories) > 1:
        confidence *= 0.85

    matched = data_matches + code_matches + workflow_matches
    return categories, round(confidence, 2), matched


SYNTHESIS_PROMPT = """\
You are CORAL, an AI assistant for NOAA ocean scientists. \
Synthesize the provided information into a clear, unified response. \
Do not mention 'agents' or internal routing."""


class Orchestrator:
    """Routes queries to specialized agents and synthesizes responses.

    All model arguments are constructor-authoritative. Callers resolve
    model names (via config.get_model or otherwise) before constructing.
    """

    def __init__(
        self,
        model: str,
        mcp_bridge: MCPBridge,
        *,
        router_model: str | None = None,
        synthesis_model: str | None = None,
        data_model: str | None = None,
        code_model: str | None = None,
        workflow_model: str | None = None,
    ):
        self.model = model
        self.mcp_bridge = mcp_bridge
        self.router_model = router_model or model
        self.synthesis_model = synthesis_model or model
        self.agents = {
            "DATA": create_data_agent(data_model or model, mcp_bridge),
            "CODE": create_code_agent(code_model or model, mcp_bridge),
            "WORKFLOW": create_workflow_agent(workflow_model or model, mcp_bridge),
        }
        self.history: list[dict] = []
        self.last_route_decision: dict | None = None
        self._wire_delegation()

    def _wire_delegation(self) -> None:
        """Give each section agent an ask_section tool that consults its peers.

        Wires the (otherwise inert) delegate() method into every agent and
        appends a short collaboration note to each system prompt. No-op when
        delegation is disabled via CORAL_DELEGATION.
        """
        if not _delegation_enabled():
            return
        sections = list(self.agents)
        for section, agent in self.agents.items():
            peers = [s for s in sections if s != section]
            if not peers:
                continue
            agent.delegate_fn = partial(self.delegate, section)
            agent.delegate_peers = peers
            agent.system_prompt = agent.system_prompt + _delegation_prompt(peers)

    @property
    def last_route_confidence(self) -> float:
        """Confidence of the last routing decision (0.0 if none yet)."""
        if not self.last_route_decision:
            return 0.0
        return float(self.last_route_decision.get("confidence", 0.0))

    # Follow-up phrases that reference prior context
    _FOLLOW_UP_PATTERNS = [
        "now plot",
        "plot it",
        "plot that",
        "plot this",
        "graph it",
        "now show",
        "show it",
        "compare it",
        "analyze it",
        "do the same",
        "same for",
        "repeat for",
        "try again",
        "what about",
        "how about",
    ]

    def _is_follow_up(self, query: str) -> str | None:
        """Detect follow-up queries that need prior context.

        Returns the last assistant message if this is a follow-up, None otherwise.
        """
        query_lower = query.lower().strip()
        is_follow_up = any(p in query_lower for p in self._FOLLOW_UP_PATTERNS)
        if is_follow_up and self.history:
            # Find last assistant response for context
            for msg in reversed(self.history):
                if msg["role"] == "assistant":
                    return msg["content"]
        return None

    async def classify(self, query: str) -> list[str]:
        """Classify user query into agent categories.

        Uses fast keyword matching first, falls back to LLM for ambiguous queries.
        For follow-up queries, includes prior context in classification.
        """
        # Try keyword route first
        kw_result = _keyword_classify(query)
        if kw_result is not None:
            categories, confidence, matched = kw_result
            self.last_route_decision = {
                "categories": list(categories),
                "method": "keyword",
                "confidence": confidence,
                "matched_keywords": matched,
                "query": query,
            }
            logger.info("Keyword-routed query to: %s (confidence=%.2f)", categories, confidence)
            return categories

        # For follow-ups, include prior context in the classification
        prior_context = self._is_follow_up(query)
        classify_query = query
        if prior_context:
            classify_query = f"Previous answer: {prior_context[:500]}\n\nFollow-up: {query}"

        # Fall back to LLM classification
        response = ollama.chat(
            model=self.router_model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": classify_query},
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
            method = "default"
            confidence = 0.3
        else:
            method = "llm"
            confidence = 0.7  # LLM classification is less certain

        self.last_route_decision = {
            "categories": list(categories),
            "method": method,
            "confidence": confidence,
            "matched_keywords": [],
            "query": query,
            "router_model": self.router_model,
            "raw_response": raw,
        }

        logger.info(
            "LLM-routed query to: %s (confidence=%.2f, model=%s)",
            categories,
            confidence,
            self.router_model,
        )
        return categories

    async def chat(self, user_message: str) -> str:
        """Route query to appropriate agent(s) and combine responses."""
        with request_context(mode="multi"):
            record_audit_event(
                "query_start",
                message_chars=len(user_message),
                router_model=self.router_model,
                synthesis_model=self.synthesis_model,
            )
            try:
                self.history.append({"role": "user", "content": user_message})
                _prune_history(self.history)

                categories = await self.classify(user_message)
                set_request_route(categories)
                record_audit_event("route_decision", routed_sections=categories)

                if len(categories) == 1:
                    agent = self.agents[categories[0]]
                    record_audit_event(
                        "section_start",
                        section=categories[0],
                        section_model=agent.model,
                    )
                    with _delegation_scope(categories[0]):
                        response = await agent.chat(user_message)
                else:
                    # Multi-agent: sequential execution, pass context forward
                    responses = []
                    accumulated_context = user_message

                    for cat in categories:
                        agent = self.agents[cat]
                        record_audit_event(
                            "section_start",
                            section=cat,
                            section_model=agent.model,
                        )
                        try:
                            with _delegation_scope(cat):
                                result = await agent.chat(accumulated_context)
                        except Exception as agent_exc:
                            logger.error("Agent %s failed: %s", cat, agent_exc)
                            record_audit_event(
                                "agent_error",
                                section=cat,
                                error=str(agent_exc),
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
            model=self.synthesis_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": synthesis_prompt},
            ],
        )
        return response.message.content

    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        """Route query and stream the response token by token.

        Single-agent queries stream the agent's final response.
        Multi-agent queries stream the synthesis step.
        """
        with request_context(mode="multi"):
            record_audit_event(
                "query_start",
                message_chars=len(user_message),
                router_model=self.router_model,
                synthesis_model=self.synthesis_model,
            )
            self.history.append({"role": "user", "content": user_message})
            _prune_history(self.history)

            categories = await self.classify(user_message)
            set_request_route(categories)

            if len(categories) == 1:
                agent = self.agents[categories[0]]
                full_content = ""
                with _delegation_scope(categories[0]):
                    async for token in agent.chat_stream(user_message):
                        full_content += token
                        yield token
                self.history.append({"role": "assistant", "content": full_content})
            else:
                # Multi-agent: run agents normally, then stream synthesis
                responses = []
                accumulated_context = user_message

                for cat in categories:
                    agent = self.agents[cat]
                    try:
                        with _delegation_scope(cat):
                            result = await agent.chat(accumulated_context)
                    except Exception as agent_exc:
                        logger.error("Agent %s failed: %s", cat, agent_exc)
                        result = f"[{cat} section unavailable: {agent_exc}]"
                    responses.append(f"[{agent.name.upper()} AGENT]\n{result}")

                    accumulated_context = (
                        f"Original question: {user_message}\n\n"
                        f"Previous findings:\n{result}\n\n"
                        f"Based on the above, continue addressing the original question."
                    )
                    agent.clear_history()

                # Stream the synthesis
                combined = "\n\n".join(responses)
                synthesis_prompt = (
                    f"The user asked: {user_message}\n\n"
                    f"Multiple specialized agents provided these findings:\n\n"
                    f"{combined}\n\n"
                    f"Synthesize these into a single, coherent response for the user. "
                    f"Don't mention 'agents' -- just provide the unified answer."
                )

                full_content = ""
                stream = ollama.chat(
                    model=self.synthesis_model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_PROMPT},
                        {"role": "user", "content": synthesis_prompt},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    token = chunk.message.content or ""
                    full_content += token
                    yield token

                self.history.append({"role": "assistant", "content": full_content})

            record_audit_event("query_end", success=True)

    async def delegate(self, from_section: str, to_section: str, query: str) -> str:
        """Run a sub-query on a peer agent and return its answer.

        Wired into each agent as the ``ask_section`` tool. Bypasses the
        classify/synthesize pipeline. Guards against consulting your own
        section, re-entering an agent already running on the delegation
        stack (which would corrupt its history), and unbounded depth.
        Failures are returned as text so the calling model can recover.
        """
        to_section = (to_section or "").strip().upper()
        from_upper = (from_section or "").strip().upper()

        if to_section not in self.agents:
            available = ", ".join(self.agents)
            return f"Cannot consult unknown section '{to_section}'. Available: {available}."
        if to_section == from_upper:
            return f"Cannot consult your own section ({to_section}); use your own tools."

        stack = _DELEGATION_STACK.get()
        if to_section in stack:
            return f"[{to_section} is already handling this request; continue with what you have.]"
        if len(stack) >= MAX_DELEGATION_DEPTH:
            return f"[Delegation limit reached; cannot consult {to_section}. Continue with what you have.]"

        target = self.agents[to_section]
        logger.info("Delegation: %s -> %s: %s", from_upper or "?", to_section, query[:80])
        record_audit_event(
            "delegation",
            from_section=from_upper,
            to_section=to_section,
            query_chars=len(query),
        )

        try:
            with _delegation_scope(to_section):
                result = await target.chat(query)
            target.clear_history()  # Don't pollute the target's session
            return result
        except Exception as e:
            logger.error("Delegation to %s failed: %s", to_section, e)
            return f"Consulting {to_section} failed: {e}"

    def reset(self):
        """Clear all agent histories."""
        self.history.clear()
        for agent in self.agents.values():
            agent.clear_history()


def create_orchestrator(model: str, mcp_bridge: MCPBridge) -> Orchestrator:
    """Create an Orchestrator with models resolved from the central config.

    This is the standard factory for CLI and web UI paths. It reads
    get_model(stage) for each stage, so env vars and set_cli_model()
    are respected. Direct Orchestrator() construction is available for
    tests and library callers who want explicit control.
    """
    from coral.config import get_model

    return Orchestrator(
        model=model,
        mcp_bridge=mcp_bridge,
        router_model=get_model("router"),
        synthesis_model=get_model("synthesis"),
        data_model=get_model("data"),
        code_model=get_model("code"),
        workflow_model=get_model("workflow"),
    )
