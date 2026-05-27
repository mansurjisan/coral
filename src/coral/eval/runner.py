"""Run a benchmark through CORAL and grade the results.

The execution seam is ``run_fn(query) -> RunObservation`` — the only part that
needs a live agent. ``make_agent_run_fn`` wires a real CoralAgent/Orchestrator
into one (capturing route, tool calls, latency, tokens), while tests pass a
scripted ``run_fn`` to exercise grading and aggregation entirely offline.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from coral.eval.tasks import EvalTask

RunFn = Callable[[str], Awaitable["RunObservation"]]


@dataclass
class RunObservation:
    """What we observed from running one query."""

    response: str = ""
    sections: list[str] = field(default_factory=list)  # routed sections (empty in single mode)
    method: str | None = None  # routing method: keyword | llm | keyword_fallback | default
    tools_called: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    tokens: int | None = None
    error: str | None = None


@dataclass
class TaskResult:
    """A graded task: the observation plus per-criterion scores."""

    task_id: str
    kind: str
    observation: RunObservation
    route_exact: bool | None = None
    route_recall: float | None = None
    tool_recall: float | None = None
    answer_pass: bool | None = None
    success: bool | None = None


def _tool_hits(called: list[str], expected_substrings: list[str]) -> int:
    return sum(1 for sub in expected_substrings if any(sub in name for name in called))


def grade(task: EvalTask, obs: RunObservation) -> TaskResult:
    """Grade one observation against a task's deterministic expectations.

    Each criterion is graded only if the task specifies it. ``success`` is the
    AND of every *specified* hard criterion (tool recall == 1 and answer pass);
    route is reported but does not gate success (single mode has no route). A
    task with no gradeable criteria yields ``success=None``.
    """
    route_exact: bool | None = None
    route_recall: float | None = None
    if task.sections:
        predicted = set(obs.sections)
        expected = set(task.sections)
        route_exact = predicted == expected
        route_recall = len(expected & predicted) / len(expected)

    tool_recall: float | None = None
    if task.expect_tools:
        tool_recall = _tool_hits(obs.tools_called, task.expect_tools) / len(task.expect_tools)

    answer_pass: bool | None = None
    if task.expect_substrings or task.forbid_substrings:
        low = obs.response.lower()
        has_all = all(s.lower() in low for s in task.expect_substrings)
        has_none = all(s.lower() not in low for s in task.forbid_substrings)
        answer_pass = bool(has_all and has_none)

    checks: list[bool] = []
    if tool_recall is not None:
        checks.append(tool_recall == 1.0)
    if answer_pass is not None:
        checks.append(answer_pass)
    success: bool | None = all(checks) if checks else None
    if obs.error:
        success = False

    return TaskResult(
        task_id=task.id,
        kind=task.kind,
        observation=obs,
        route_exact=route_exact,
        route_recall=route_recall,
        tool_recall=tool_recall,
        answer_pass=answer_pass,
        success=success,
    )


async def run_suite(tasks: list[EvalTask], run_fn: RunFn) -> list[TaskResult]:
    """Run every task through ``run_fn`` and grade it. Errors become failures."""
    results: list[TaskResult] = []
    for task in tasks:
        try:
            obs = await run_fn(task.query)
        except Exception as exc:  # a crashing query is a failed task, not a crashed run
            obs = RunObservation(error=str(exc))
        results.append(grade(task, obs))
    return results


def _read_tokens(agent, is_multi: bool) -> int | None:
    """Best-effort token count from the agent's last run (None if unavailable)."""
    if is_multi:
        total = 0
        seen = False
        for sub in getattr(agent, "agents", {}).values():
            stats = getattr(sub, "last_stats", {}) or {}
            if "tokens" in stats:
                total += int(stats["tokens"])
                seen = True
        return total if seen else None
    stats = getattr(agent, "last_stats", {}) or {}
    return int(stats["tokens"]) if "tokens" in stats else None


def make_agent_run_fn(agent) -> RunFn:
    """Wrap a live CoralAgent (single) or Orchestrator (multi) as a run_fn.

    Resets the agent before each task (so cases are independent), collects tool
    calls via the on_tool_call hook, times the call, and reads the route from
    ``last_route_decision`` when present. Duck-typed: multi mode is detected by
    the presence of an ``agents`` mapping.
    """
    is_multi = hasattr(agent, "agents")

    async def run_fn(query: str) -> RunObservation:
        if hasattr(agent, "reset"):
            agent.reset()

        collected: list[str] = []

        def on_tool(name, _args, _result):
            collected.append(name)

        if is_multi:
            for sub in agent.agents.values():
                sub.on_tool_call = on_tool
        else:
            agent.on_tool_call = on_tool

        start = time.perf_counter()
        response = await agent.chat(query)
        latency = time.perf_counter() - start

        sections: list[str] = []
        method: str | None = None
        decision = getattr(agent, "last_route_decision", None)
        if decision:
            sections = list(decision.get("categories", []))
            method = decision.get("method")

        return RunObservation(
            response=response,
            sections=sections,
            method=method,
            tools_called=collected,
            latency_s=round(latency, 3),
            tokens=_read_tokens(agent, is_multi),
        )

    return run_fn
