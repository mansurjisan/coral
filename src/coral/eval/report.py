"""Aggregate and render evaluation results.

Pure functions over the TaskResult list (duck-typed, no runtime import of the
runner), so they're trivially testable. Produces both a Rich table for humans
and a JSON blob for the paper's tables/plots.
"""

from __future__ import annotations

from statistics import mean

_METRICS = ["route_exact", "route_recall", "tool_recall", "answer_pass", "success", "latency_s", "tokens"]


def _rate(values: list) -> float | None:
    """Fraction of True among non-None booleans."""
    present = [v for v in values if v is not None]
    return round(sum(1 for v in present if v) / len(present), 3) if present else None


def _mean(values: list) -> float | None:
    present = [v for v in values if v is not None]
    return round(mean(present), 3) if present else None


def _summary(results: list) -> dict:
    return {
        "n": len(results),
        "route_exact": _rate([r.route_exact for r in results]),
        "route_recall": _mean([r.route_recall for r in results]),
        "tool_recall": _mean([r.tool_recall for r in results]),
        "answer_pass": _rate([r.answer_pass for r in results]),
        "success": _rate([r.success for r in results]),
        "latency_s": _mean([r.observation.latency_s for r in results]),
        "tokens": _mean([r.observation.tokens for r in results]),
    }


def aggregate(results: list) -> dict:
    """Overall summary plus a per-kind breakdown."""
    agg = {"overall": _summary(results), "by_kind": {}}
    for kind in sorted({r.kind for r in results}):
        agg["by_kind"][kind] = _summary([r for r in results if r.kind == kind])
    return agg


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table(agg: dict, console) -> None:
    """Render the aggregate summary as a Rich table."""
    from rich.table import Table

    table = Table(title="CORAL eval", show_lines=False)
    table.add_column("Scope", style="cyan", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("route✓", justify="right")
    table.add_column("route_rec", justify="right")
    table.add_column("tool_rec", justify="right")
    table.add_column("answer", justify="right")
    table.add_column("success", justify="right", style="bold")
    table.add_column("lat(s)", justify="right")
    table.add_column("tokens", justify="right")

    def row(label: str, s: dict) -> None:
        table.add_row(
            label,
            str(s["n"]),
            _fmt(s["route_exact"]),
            _fmt(s["route_recall"]),
            _fmt(s["tool_recall"]),
            _fmt(s["answer_pass"]),
            _fmt(s["success"]),
            _fmt(s["latency_s"]),
            _fmt(s["tokens"]),
        )

    row("overall", agg["overall"])
    for kind, summary in agg["by_kind"].items():
        row(f"  {kind}", summary)

    console.print(table)


def results_to_json(results: list, agg: dict) -> dict:
    """Serialize per-task results plus the aggregate, for the paper's tables."""
    return {
        "summary": agg,
        "results": [
            {
                "task_id": r.task_id,
                "kind": r.kind,
                "sections": r.observation.sections,
                "method": r.observation.method,
                "tools_called": r.observation.tools_called,
                "latency_s": r.observation.latency_s,
                "tokens": r.observation.tokens,
                "route_exact": r.route_exact,
                "route_recall": r.route_recall,
                "tool_recall": r.tool_recall,
                "answer_pass": r.answer_pass,
                "success": r.success,
                "error": r.observation.error,
            }
            for r in results
        ],
    }
