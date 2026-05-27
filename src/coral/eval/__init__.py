"""CORAL evaluation harness.

A small, offline-testable framework for measuring CORAL's routing and
task-success behavior, and for the single-vs-multi ablation that anchors the
paper's results. Execution (which needs live Ollama + MCP) is isolated in a
thin injectable ``run_fn``; loading, grading, aggregation, and reporting are
pure functions exercised by the test suite without any model or server.
"""

from __future__ import annotations

from coral.eval.report import aggregate, render_table, results_to_json
from coral.eval.runner import (
    RunObservation,
    TaskResult,
    grade,
    make_agent_run_fn,
    run_suite,
)
from coral.eval.tasks import EvalTask, load_tasks

__all__ = [
    "EvalTask",
    "load_tasks",
    "RunObservation",
    "TaskResult",
    "grade",
    "run_suite",
    "make_agent_run_fn",
    "aggregate",
    "render_table",
    "results_to_json",
]
