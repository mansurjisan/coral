"""Markdown/RST/text chunker that splits on headers."""

from __future__ import annotations


class MarkdownChunker:
    """Split markdown/rst/text files on headers or by size."""

    def chunk(self, content: str, file_path: str, max_chunk: int = 2000) -> list[dict]:
        chunks = []
        current_section = ""
        current_text = ""

        for line in content.split("\n"):
            # Markdown header or RST underline-style header
            if line.startswith("#") or (
                len(line) > 2 and set(line.strip()) <= {"=", "-", "~", "^"} and current_text.strip()
            ):
                if current_text.strip():
                    for sub in self._split_large(current_text.strip(), max_chunk):
                        chunks.append(
                            {
                                "text": sub,
                                "metadata": {
                                    "file_path": file_path,
                                    "section": current_section,
                                    "type": "markdown",
                                },
                            }
                        )
                if line.startswith("#"):
                    current_section = line.lstrip("# ").strip()
                current_text = line + "\n"
            else:
                current_text += line + "\n"

        if current_text.strip():
            for sub in self._split_large(current_text.strip(), max_chunk):
                chunks.append(
                    {
                        "text": sub,
                        "metadata": {
                            "file_path": file_path,
                            "section": current_section,
                            "type": "markdown",
                        },
                    }
                )

        return chunks

    def _split_large(self, text: str, max_chunk: int) -> list[str]:
        if len(text) <= max_chunk:
            return [text]
        parts = []
        lines = text.split("\n")
        current: list[str] = []
        current_len = 0
        for line in lines:
            current.append(line)
            current_len += len(line) + 1
            if current_len >= max_chunk:
                parts.append("\n".join(current))
                current = []
                current_len = 0
        if current:
            parts.append("\n".join(current))
        return parts
