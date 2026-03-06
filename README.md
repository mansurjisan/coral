# CORAL — Coastal Ocean Research AI Layer

A self-hosted AI agent for NOAA HPC that gives coastal scientists a ChatGPT-like experience — entirely within NOAA's network.

## What it does

CORAL combines a local LLM (via Ollama) with live ocean data tools, a RAG knowledge base over scientific code and documentation, and HPC workflow integration:

- **Live ocean data** — Query real-time water levels, hurricane tracks, storm surge forecasts, recon flights, and satellite data through [ocean-mcp](https://github.com/mansurjisan) servers
- **RAG over documentation** — Search SCHISM/ADCIRC source code, NOAA tech memos, model configs, and namelists
- **Local file interaction** — Inspect NetCDF model outputs, parse Slurm logs, monitor ecFlow workflows
- **Code execution** — Generate and run Python analysis scripts (xarray, matplotlib, cartopy)
- **CLI + Web UI** — Interactive terminal chat or Gradio web interface

All running on Ollama with open-weight LLMs. No external APIs, no commercial licenses.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Pull a model
ollama pull qwen3:8b

# Chat
coral chat --model qwen3:8b

# Index documentation into RAG
coral index /path/to/schism/src

# Launch web UI
coral serve --model qwen3:8b --port 7860

# List available tools
coral tools
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CORAL                                                  │
│                                                         │
│  User ──► CLI / Gradio UI                               │
│               │                                         │
│           CoralAgent (agentic loop)                      │
│               │                                         │
│           Ollama (qwen3:8b/32b)                         │
│               │                                         │
│           MCPBridge ──► MCP Servers                      │
│               ├── coops-mcp (tides/water levels)        │
│               ├── nhc-mcp (hurricanes)                  │
│               ├── stofs-mcp (storm surge)               │
│               ├── recon-mcp (recon flights)             │
│               ├── erddap-mcp (satellite data)           │
│               ├── ofs-mcp (forecast models)             │
│               ├── coral-rag (documentation search)      │
│               ├── coral-netcdf (local NetCDF files)     │
│               ├── coral-slurm (job management)          │
│               ├── coral-ecflow (workflow status)        │
│               └── coral-viz (code execution)            │
└─────────────────────────────────────────────────────────┘
```

## Example Queries

```
What is the current water level at The Battery, NYC?
Are there any active hurricanes in the Atlantic?
Inspect the NetCDF file /scratch/stofs/output/stofs_2d.nc
What does the subroutine schism_init do?
Show me my last failed Slurm jobs and diagnose what went wrong
Plot water levels from stofs_2d.nc at lat=40.7, lon=-74.0
```

## Project Structure

```
coral/
├── pyproject.toml              # Package config + dependencies
├── coral_config.json           # MCP server configuration
├── src/coral/
│   ├── agent.py                # Agentic loop with tool routing
│   ├── mcp_bridge.py           # MCP server connection manager
│   ├── prompts.py              # System prompt
│   ├── cli.py                  # Typer CLI (chat, serve, index, tools)
│   ├── web_ui.py               # Gradio interface
│   ├── rag/
│   │   ├── indexer.py          # LanceDB + Ollama embeddings
│   │   ├── retriever.py        # Hybrid vector + BM25 search
│   │   └── chunkers/           # Fortran, C, namelist, markdown, ecFlow
│   └── servers/
│       ├── netcdf_server.py    # Read/query NetCDF files
│       ├── slurm_server.py     # Slurm job management
│       ├── ecflow_server.py    # ecFlow suite monitoring
│       ├── rag_server.py       # RAG search over docs
│       └── viz_server.py       # Python code execution
└── tests/                      # 68 tests
```

## Optional Dependencies

```bash
# RAG pipeline (LanceDB, tree-sitter, docling, f90nml)
pip install -e ".[rag]"

# Scientific tools (xarray, netCDF4, matplotlib, cartopy)
pip install -e ".[science]"

# Everything
pip install -e ".[all]"
```

## HPC Deployment (NOAA Ursa)

See `slurm/` for Slurm job scripts that run Ollama on GPU nodes and CORAL on service nodes.

## License

Apache-2.0
