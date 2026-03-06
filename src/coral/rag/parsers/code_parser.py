"""Generic code parser — delegates to language-specific chunkers."""

from __future__ import annotations

from pathlib import Path

from coral.rag.chunkers.fortran_chunker import FortranChunker
from coral.rag.chunkers.c_chunker import CChunker
from coral.rag.chunkers.markdown_chunker import MarkdownChunker

_FORTRAN_EXTS = {".f90", ".f", ".f77", ".ftn", ".f90", ".f"}
_C_EXTS = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"}

_fortran = FortranChunker()
_c = CChunker()
_markdown = MarkdownChunker()


def parse_code_file(source_code: str, file_path: str) -> list[dict]:
    """Parse a source code file into chunks using the appropriate chunker."""
    ext = Path(file_path).suffix.lower()

    if ext in _FORTRAN_EXTS:
        return _fortran.chunk(source_code, file_path)
    elif ext in _C_EXTS:
        return _c.chunk(source_code, file_path)
    else:
        return _markdown.chunk(source_code, file_path)
