"""Routing regression corpus.

A labeled set of (query -> expected keyword route) cases that guards the
deterministic pre-router in ``orchestrator._keyword_classify``. Editing the
keyword lists used to be unmeasurable — a fix for one query could silently
break others (especially with the old substring matching). This corpus turns
that into a hard signal: add rows as new routing behavior is needed, and CI
fails the moment a change regresses an existing case.

The ``expected`` column is one of:
  - ``DATA`` / ``CODE`` / ``WORKFLOW`` (or ``A|B`` for multi-section), meaning
    the keyword router must return exactly that set, or
  - ``LLM``, meaning the keyword router must decline (return ``None``) so the
    query falls through to the LLM classifier.

To extend coverage, add a line to ``tests/data/routing_eval.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from coral.agents.orchestrator import _keyword_classify

_CORPUS_PATH = Path(__file__).parent / "data" / "routing_eval.csv"


def _load_corpus() -> list[tuple[str, str]]:
    with _CORPUS_PATH.open(encoding="utf-8") as f:
        return [(row["query"], row["expected"]) for row in csv.DictReader(f)]


_CORPUS = _load_corpus()


def _route(query: str) -> set[str] | None:
    """Return the keyword router's category set, or None when it declines."""
    result = _keyword_classify(query)
    if result is None:
        return None
    categories, _confidence, _matched = result
    return set(categories)


def _expected_set(expected: str) -> set[str] | None:
    return None if expected == "LLM" else set(expected.split("|"))


def test_corpus_is_loaded():
    # Guard against the file going missing or silently shrinking.
    assert len(_CORPUS) >= 30


@pytest.mark.parametrize("query,expected", _CORPUS, ids=[q for q, _ in _CORPUS])
def test_routing_case(query: str, expected: str):
    assert _route(query) == _expected_set(expected)


def test_corpus_accuracy_is_total():
    """Report and enforce accuracy across the whole corpus.

    The corpus is hand-labeled to reflect what the keyword router *should*
    do, so a passing suite means 100% on these cases. Surfacing the number
    here makes the metric explicit (and the failure list actionable).
    """
    misses = [
        (query, expected, _route(query)) for query, expected in _CORPUS if _route(query) != _expected_set(expected)
    ]
    accuracy = 1 - len(misses) / len(_CORPUS)
    assert accuracy == 1.0, f"routing accuracy {accuracy:.1%}; misses: {misses}"
