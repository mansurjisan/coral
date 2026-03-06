"""Fortran namelist chunker using f90nml."""

from __future__ import annotations

import f90nml


class NamelistChunker:
    """Parse Fortran namelists into individual namelist group chunks."""

    def chunk(self, content: str, file_path: str) -> list[dict]:
        chunks = []
        try:
            nml = f90nml.reads(content)
            for group_name, group_data in nml.items():
                text = f"&{group_name}\n"
                for key, value in group_data.items():
                    text += f"  {key} = {value}\n"
                text += "/"
                chunks.append({
                    "text": text,
                    "metadata": {
                        "file_path": file_path,
                        "namelist_group": group_name,
                        "type": "namelist",
                    },
                })
        except Exception:
            pass

        # Fall back to plain text if no groups were parsed
        if not chunks and content.strip():
            chunks.append({
                "text": content,
                "metadata": {"file_path": file_path, "type": "namelist"},
            })
        return chunks
