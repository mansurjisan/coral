"""Fortran source code chunker using regex-based parsing.

Extracts subroutines, functions, modules, and programs as semantic chunks.
Falls back to fixed-size line-based chunks for files that don't match.
"""

from __future__ import annotations

import re


# Patterns for Fortran program units (case-insensitive)
_UNIT_START = re.compile(
    r"^\s*(?:(?:recursive|pure|elemental|impure)\s+)?"
    r"(subroutine|function|module|program)\s+(\w+)",
    re.IGNORECASE | re.MULTILINE,
)
_UNIT_END = re.compile(
    r"^\s*end\s+(subroutine|function|module|program)(?:\s+(\w+))?",
    re.IGNORECASE | re.MULTILINE,
)


class FortranChunker:
    """Parse Fortran source into subroutine/function/module chunks."""

    def chunk(self, source_code: str, file_path: str) -> list[dict]:
        lines = source_code.split("\n")
        chunks = []
        stack: list[dict] = []  # track nested units

        for i, line in enumerate(lines):
            # Check for unit start
            m_start = _UNIT_START.match(line)
            if m_start:
                stack.append(
                    {
                        "type": m_start.group(1).lower(),
                        "name": m_start.group(2),
                        "start": i,
                    }
                )

            # Check for unit end
            m_end = _UNIT_END.match(line)
            if m_end and stack:
                unit = stack.pop()
                text = "\n".join(lines[unit["start"] : i + 1])
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            "file_path": file_path,
                            "node_type": unit["type"],
                            "name": unit["name"],
                            "start_line": unit["start"] + 1,
                            "end_line": i + 1,
                            "language": "fortran",
                        },
                    }
                )

        if not chunks:
            chunks = self._fallback_chunk(source_code, file_path)

        return chunks

    def _fallback_chunk(self, source_code: str, file_path: str, chunk_size: int = 1500) -> list[dict]:
        lines = source_code.split("\n")
        chunks = []
        current: list[str] = []
        current_len = 0
        start_line = 0

        for i, line in enumerate(lines):
            current.append(line)
            current_len += len(line) + 1
            if current_len >= chunk_size:
                chunks.append(
                    {
                        "text": "\n".join(current),
                        "metadata": {
                            "file_path": file_path,
                            "start_line": start_line + 1,
                            "language": "fortran",
                        },
                    }
                )
                # Keep last 3 lines as overlap
                overlap = current[-3:]
                current = overlap
                current_len = sum(len(ln) + 1 for ln in current)
                start_line = i - len(overlap) + 1

        if current:
            chunks.append(
                {
                    "text": "\n".join(current),
                    "metadata": {
                        "file_path": file_path,
                        "start_line": start_line + 1,
                        "language": "fortran",
                    },
                }
            )

        return chunks
