# CORAL Architecture

## Overview

CORAL runs as two Slurm jobs on NOAA Ursa HPC:

1. **GPU node** (`u1-h100`): Ollama serves the LLM (qwen3:32b) and embedding model (nomic-embed-text)
2. **Service node** (`u1-service`): CORAL agent, MCP servers, RAG pipeline, and web UI

The service node has internet access for NOAA API calls. Users connect via SSH tunnel.

## Components

### Orchestrator Layer (`src/coral/agents/orchestrator.py`)

CORAL V2 uses an orchestrator plus three focused sections:
1. **Data**: live NOAA data and NetCDF inspection
2. **Code**: indexed documentation, source code explanation, and plotting
3. **Workflow**: Slurm and ecFlow diagnosis

The orchestrator classifies the query, routes it to one or more sections, and synthesizes a single response when multiple sections contribute.

### Shared Agent Runtime (`src/coral/agents/base.py`, `src/coral/agent.py`)

- `BaseAgent` provides the shared tool-calling loop for the V2 sections.
- `agent.py` remains as the fallback single-agent path.
- The shared runtime handles history, truncation, tool execution, and error handling.

### MCP Bridge (`src/coral/mcp_bridge.py`)

Manages connections to all MCP servers:
- Each server runs as a subprocess (stdio transport)
- Discovers tools from each server and builds a unified tool registry
- Routes tool calls to the correct server session
- Rejects duplicate tool names so routing stays deterministic

### MCP Servers

**Data section servers:**
- `coops-mcp` — CO-OPS tide stations, water levels
- `nhc-mcp` — NHC hurricane tracks, forecasts
- `stofs-mcp` — STOFS storm surge forecasts
- `recon-mcp` — Hurricane Hunter recon flights
- `erddap-mcp` — NOAA CoastWatch satellite data
- `ofs-mcp` — Operational forecast systems
- `adcirc-mcp` — ADCIRC model config parsing
- `schism-mcp` — SCHISM model config parsing
- `goes-mcp` — GOES satellite imagery
- `usgs-mcp` — USGS streamflow and flood data
- `winds-mcp` — Weather station wind observations
- `ww3-mcp` — WaveWatch III wave forecasts
- `netcdf_server.py` — Read/query NetCDF files via xarray

**Workflow section servers:**
- `slurm_server.py` — Parse Slurm jobs and logs
- `ecflow_server.py` — ecFlow suite status

**Code section servers:**
- `rag_server.py` — Search indexed documentation
- `viz_server.py` — Execute Python scripts for analysis/plots

On Ursa, `viz_server.py` is expected to run through Apptainer using `containers/coral_sandbox.sif`. The service-node launch script sets `CORAL_REQUIRE_SANDBOX=1`, so host-side fallback is disabled there.

### RAG Pipeline (`src/coral/rag/`)

- **Chunkers**: Split files by type (Fortran subroutines, C functions, namelist groups, etc.)
- **Indexer**: Embeds chunks via Ollama and stores in LanceDB
- **Retriever**: Hybrid search — vector similarity + BM25 full-text, merged with Reciprocal Rank Fusion

### Web UI (`src/coral/web_ui.py`)

Gradio ChatInterface on port 7860, accessed via SSH tunnel.

## Data Flow

```text
User query
  → Orchestrator classifies intent
  → One or more section agents run with section-specific tool filters
  → MCP Bridge routes tool calls to correct server
  → Server executes (API call, file read, subprocess, etc.)
  → Results returned to the section agent for next iteration
  → Orchestrator synthesizes a final answer if needed
  → Final text response to user
```

## Tool Routing

Routing is now two-stage:

1. The orchestrator decides which section or sections should handle the query.
2. Each section only sees the MCP tools assigned to that domain.

This keeps tool choice narrower and reduces confusion on ambiguous model names like `stofs`, `schism`, and `adcirc`.
