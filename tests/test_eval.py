"""Tests for the evaluation harness — fully offline (no Ollama, no MCP)."""

import io
from pathlib import Path

import pytest

from coral.eval import (
    EvalTask,
    RunObservation,
    TaskResult,
    aggregate,
    grade,
    load_tasks,
    make_agent_run_fn,
    render_table,
    results_to_json,
    run_suite,
)


# ── Task loading ──


class TestLoadTasks:
    def test_loads_and_normalizes(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text(
            "# a comment line\n"
            "\n"
            '{"id": "a", "query": "q1", "sections": ["data"], "expect_tools": ["coops_"]}\n'
            '{"id": "b", "query": "q2", "kind": "cross"}\n',
            encoding="utf-8",
        )
        tasks = load_tasks(p)
        assert len(tasks) == 2
        assert tasks[0].sections == ["DATA"]  # uppercased
        assert tasks[0].expect_tools == ["coops_"]
        assert tasks[1].kind == "cross"
        assert tasks[1].sections == []  # default

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tasks(tmp_path / "nope.jsonl")

    def test_bad_line_raises_valueerror(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"query": "no id field"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            load_tasks(p)


# ── Grading ──


class TestGrade:
    def test_route_exact_and_recall(self):
        task = EvalTask(id="t", query="q", sections=["DATA", "CODE"])
        exact = grade(task, RunObservation(sections=["DATA", "CODE"]))
        assert exact.route_exact is True and exact.route_recall == 1.0
        partial = grade(task, RunObservation(sections=["DATA"]))
        assert partial.route_exact is False and partial.route_recall == 0.5
        over = grade(task, RunObservation(sections=["DATA", "CODE", "WORKFLOW"]))
        assert over.route_exact is False and over.route_recall == 1.0

    def test_tool_recall(self):
        task = EvalTask(id="t", query="q", expect_tools=["coops_", "execute_python"])
        assert grade(task, RunObservation(tools_called=["coops_get_water_levels"])).tool_recall == 0.5
        assert grade(task, RunObservation(tools_called=["coops_x", "execute_python"])).tool_recall == 1.0
        assert grade(task, RunObservation(tools_called=[])).tool_recall == 0.0

    def test_answer_pass(self):
        task = EvalTask(id="t", query="q", expect_substrings=["MLLW"], forbid_substrings=["error"])
        assert grade(task, RunObservation(response="1.2 m MLLW")).answer_pass is True
        assert grade(task, RunObservation(response="no datum here")).answer_pass is False  # missing expected
        assert grade(task, RunObservation(response="MLLW but error occurred")).answer_pass is False  # forbidden

    def test_success_is_and_of_specified_criteria(self):
        task = EvalTask(id="t", query="q", expect_tools=["coops_"], expect_substrings=["MLLW"])
        good = grade(task, RunObservation(response="1 m MLLW", tools_called=["coops_x"]))
        assert good.success is True
        bad = grade(task, RunObservation(response="1 m MLLW", tools_called=[]))  # tool missing
        assert bad.success is False

    def test_no_criteria_yields_none(self):
        task = EvalTask(id="t", query="q")
        assert grade(task, RunObservation(response="anything")).success is None

    def test_error_is_failure(self):
        task = EvalTask(id="t", query="q", expect_tools=["coops_"])
        assert grade(task, RunObservation(error="boom")).success is False


# ── Suite execution (scripted run_fn, no live agent) ──


class TestRunSuite:
    @pytest.mark.asyncio
    async def test_grades_each_task(self):
        tasks = [
            EvalTask(id="a", query="q1", sections=["DATA"], expect_tools=["coops_"]),
            EvalTask(id="b", query="q2", sections=["CODE"], expect_tools=["execute_python"]),
        ]
        scripted = {
            "q1": RunObservation(response="ok", sections=["DATA"], tools_called=["coops_get_water_levels"]),
            "q2": RunObservation(response="ok", sections=["DATA"], tools_called=[]),  # wrong route, missing tool
        }

        async def run_fn(query):
            return scripted[query]

        results = await run_suite(tasks, run_fn)
        assert len(results) == 2
        a, b = results
        assert a.route_exact is True and a.tool_recall == 1.0 and a.success is True
        assert b.route_exact is False and b.tool_recall == 0.0 and b.success is False

    @pytest.mark.asyncio
    async def test_crashing_query_becomes_failed_task(self):
        async def run_fn(query):
            raise RuntimeError("kaboom")

        results = await run_suite([EvalTask(id="x", query="boom", expect_tools=["coops_"])], run_fn)
        assert results[0].success is False
        assert "kaboom" in results[0].observation.error


# ── Live-wiring seam (fake agents, no Ollama) ──


class _FakeSingleAgent:
    def __init__(self):
        self.on_tool_call = None
        self.last_stats = {"tokens": 42}
        self.reset_called = 0

    def reset(self):
        self.reset_called += 1

    async def chat(self, query):
        if self.on_tool_call:
            self.on_tool_call("coops_get_water_levels", {}, "1.2 m MLLW")
        return "Water level is 1.2 m MLLW"


class _FakeSub:
    def __init__(self, tool_name, tokens):
        self.on_tool_call = None
        self.last_stats = {"tokens": tokens}
        self._tool = tool_name

    async def chat(self, query):
        if self.on_tool_call:
            self.on_tool_call(self._tool, {}, "ok")
        return "sub done"


class _FakeMultiAgent:
    def __init__(self):
        self.agents = {"DATA": _FakeSub("coops_get_water_levels", 30), "CODE": _FakeSub("execute_python", 12)}
        self.last_route_decision = {"categories": ["DATA", "CODE"], "method": "keyword"}

    def reset(self):
        for sub in self.agents.values():
            sub.on_tool_call = None

    async def chat(self, query):
        return " ".join([await sub.chat(query) for sub in self.agents.values()])


class TestMakeAgentRunFn:
    @pytest.mark.asyncio
    async def test_single_agent(self):
        agent = _FakeSingleAgent()
        run_fn = make_agent_run_fn(agent)
        obs = await run_fn("water level at the battery")
        assert agent.reset_called == 1
        assert obs.tools_called == ["coops_get_water_levels"]
        assert obs.sections == []  # single mode has no route
        assert obs.tokens == 42
        assert "MLLW" in obs.response
        assert obs.latency_s >= 0

    @pytest.mark.asyncio
    async def test_multi_agent(self):
        agent = _FakeMultiAgent()
        run_fn = make_agent_run_fn(agent)
        obs = await run_fn("get water levels and plot")
        assert set(obs.tools_called) == {"coops_get_water_levels", "execute_python"}
        assert obs.sections == ["DATA", "CODE"]
        assert obs.method == "keyword"
        assert obs.tokens == 42  # summed across sub-agents


# ── Aggregation / reporting ──


class TestReport:
    def _results(self):
        return [
            TaskResult(
                "a", "single", RunObservation(latency_s=1.0, tokens=10), route_exact=True, tool_recall=1.0, success=True
            ),
            TaskResult(
                "b",
                "single",
                RunObservation(latency_s=3.0, tokens=30),
                route_exact=False,
                tool_recall=0.0,
                success=False,
            ),
            TaskResult(
                "c", "cross", RunObservation(latency_s=2.0, tokens=20), route_exact=True, tool_recall=1.0, success=True
            ),
        ]

    def test_aggregate_overall_and_by_kind(self):
        agg = aggregate(self._results())
        assert agg["overall"]["n"] == 3
        assert agg["overall"]["success"] == round(2 / 3, 3)
        assert agg["overall"]["route_exact"] == round(2 / 3, 3)
        assert agg["overall"]["latency_s"] == 2.0
        assert set(agg["by_kind"]) == {"single", "cross"}
        assert agg["by_kind"]["cross"]["success"] == 1.0
        assert agg["by_kind"]["single"]["n"] == 2

    def test_results_to_json_shape(self):
        results = self._results()
        blob = results_to_json(results, aggregate(results))
        assert blob["summary"]["overall"]["n"] == 3
        assert blob["results"][0]["task_id"] == "a"
        assert blob["results"][0]["success"] is True

    def test_render_table_does_not_raise(self):
        from rich.console import Console

        render_table(aggregate(self._results()), Console(file=io.StringIO()))


# ── The shipped benchmark must be well-formed ──


def test_seed_task_set_is_valid():
    seed = Path(__file__).parent.parent / "eval" / "tasks.jsonl"
    tasks = load_tasks(seed)
    assert len(tasks) >= 12
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))  # unique ids
    valid_sections = {"DATA", "CODE", "WORKFLOW"}
    for task in tasks:
        assert task.query
        assert set(task.sections) <= valid_sections
