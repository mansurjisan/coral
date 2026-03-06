"""MCP server exposing RAG search over indexed documentation."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coral-rag")

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from coral.rag.retriever import CoralRetriever
        db_path = os.environ.get("CORAL_VECTORDB", "~/.coral/vectordb")
        _retriever = CoralRetriever(db_path=db_path)
    return _retriever


@mcp.tool()
def search_documentation(query: str, top_k: int = 5) -> str:
    """Search indexed NOAA documentation, model source code, and configuration files.

    Use this to find information about:
    - SCHISM/ADCIRC/UFS-Coastal model source code (Fortran subroutines, modules)
    - NOAA technical memorandums and reports
    - Model configuration files and namelists
    - User guides and README documentation

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
    """
    retriever = _get_retriever()
    results = retriever.search(query, top_k=top_k)

    output = []
    for r in results:
        source = r.get("file_path", "unknown")
        line = r.get("start_line", "")
        node_type = r.get("node_type", "")
        name = r.get("name", "")
        text = r.get("text", "")[:2000]

        header = f"--- {source}"
        if line:
            header += f":{line}"
        if node_type:
            header += f" ({node_type}"
            if name:
                header += f" {name}"
            header += ")"
        header += " ---"
        output.append(f"{header}\n{text}")

    return "\n\n".join(output) if output else "No results found in indexed documentation."


if __name__ == "__main__":
    mcp.run()
