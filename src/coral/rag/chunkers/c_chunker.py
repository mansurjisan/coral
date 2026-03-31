"""C/C++ source code chunker using regex-based parsing.

Extracts top-level functions as semantic chunks by tracking brace depth.
Falls back to fixed-size chunks for headers or files without functions.
"""

from __future__ import annotations

import re

# Match function definitions (simplified: type + name + parens at start of line)
_FUNC_DEF = re.compile(
    r"^[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*\{?\s*$",
    re.MULTILINE,
)


class CChunker:
    """Parse C/C++ source into function-level chunks."""

    def chunk(self, source_code: str, file_path: str) -> list[dict]:
        lines = source_code.split("\n")
        chunks = []
        i = 0

        while i < len(lines):
            m = _FUNC_DEF.match(lines[i])
            if m:
                func_name = m.group(1)
                start = i
                # Find the function body by tracking braces
                depth = 0
                found_open = False
                j = i
                while j < len(lines):
                    for ch in lines[j]:
                        if ch == "{":
                            depth += 1
                            found_open = True
                        elif ch == "}":
                            depth -= 1
                    if found_open and depth <= 0:
                        break
                    j += 1

                text = "\n".join(lines[start : j + 1])
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            "file_path": file_path,
                            "node_type": "function",
                            "name": func_name,
                            "start_line": start + 1,
                            "end_line": j + 1,
                            "language": "c",
                        },
                    }
                )
                i = j + 1
            else:
                i += 1

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
                            "language": "c",
                        },
                    }
                )
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
                        "language": "c",
                    },
                }
            )

        return chunks
