"""Evaluation task set: schema and loader.

A benchmark is a JSONL file, one task per line (``#`` comment lines and blank
lines are ignored). Each task carries the query plus whatever deterministic
expectations apply — expected route, tool substrings, and answer substrings —
so grading needs no LLM judge. Tasks may specify only the criteria that are
checkable for them; unspecified criteria are simply not graded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalTask:
    """One benchmark case.

    Attributes:
        id: Stable identifier (used in reports).
        query: The user query to send to CORAL.
        sections: Expected route, e.g. ["DATA", "CODE"]. Empty = route not graded
            (e.g. single-agent mode, or a genuinely ambiguous case).
        expect_tools: Tool-name substrings that should be called (e.g. "coops_").
        expect_substrings: Strings that should appear in the answer (case-insensitive).
        forbid_substrings: Strings that must NOT appear in the answer.
        kind: Bucket for per-category reporting: single | cross | ambiguous.
        notes: Free-text rationale / provenance.
    """

    id: str
    query: str
    sections: list[str] = field(default_factory=list)
    expect_tools: list[str] = field(default_factory=list)
    expect_substrings: list[str] = field(default_factory=list)
    forbid_substrings: list[str] = field(default_factory=list)
    kind: str = "single"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> EvalTask:
        return cls(
            id=data["id"],
            query=data["query"],
            sections=[s.upper() for s in data.get("sections", [])],
            expect_tools=list(data.get("expect_tools", [])),
            expect_substrings=list(data.get("expect_substrings", [])),
            forbid_substrings=list(data.get("forbid_substrings", [])),
            kind=data.get("kind", "single"),
            notes=data.get("notes", ""),
        )


def load_tasks(path: str | Path) -> list[EvalTask]:
    """Load a JSONL benchmark file into EvalTask objects."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Eval task set not found: {file_path}")

    tasks: list[EvalTask] = []
    with file_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                tasks.append(EvalTask.from_dict(json.loads(stripped)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{file_path}:{lineno}: invalid task: {exc}") from exc
    return tasks
