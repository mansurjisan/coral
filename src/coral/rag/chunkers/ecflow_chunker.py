"""ecFlow suite definition chunker.

Splits ecFlow .def/.ecf files into task/family-level chunks.
"""

from __future__ import annotations



class EcflowChunker:
    """Parse ecFlow suite definitions into family/task chunks."""

    def chunk(self, content: str, file_path: str) -> list[dict]:
        chunks = []
        lines = content.split("\n")
        current_block: list[str] = []
        current_name = ""
        current_type = ""
        start_line = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect block starts
            if stripped.startswith("family ") or stripped.startswith("task "):
                # Save previous block
                if current_block and current_name:
                    chunks.append({
                        "text": "\n".join(current_block),
                        "metadata": {
                            "file_path": file_path,
                            "node_type": current_type,
                            "name": current_name,
                            "start_line": start_line + 1,
                            "type": "ecflow",
                        },
                    })
                parts = stripped.split(None, 1)
                current_type = parts[0]
                current_name = parts[1] if len(parts) > 1 else ""
                current_block = [line]
                start_line = i
            elif stripped.startswith("endfamily") or stripped.startswith("endtask"):
                current_block.append(line)
                if current_name:
                    chunks.append({
                        "text": "\n".join(current_block),
                        "metadata": {
                            "file_path": file_path,
                            "node_type": current_type,
                            "name": current_name,
                            "start_line": start_line + 1,
                            "type": "ecflow",
                        },
                    })
                current_block = []
                current_name = ""
            else:
                current_block.append(line)

        # Remaining content
        if current_block:
            text = "\n".join(current_block)
            if text.strip():
                chunks.append({
                    "text": text,
                    "metadata": {
                        "file_path": file_path,
                        "start_line": start_line + 1,
                        "type": "ecflow",
                    },
                })

        # If no structure found, fall back to whole-file chunk
        if not chunks and content.strip():
            chunks.append({
                "text": content,
                "metadata": {"file_path": file_path, "type": "ecflow"},
            })

        return chunks
