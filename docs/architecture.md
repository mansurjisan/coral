# CORAL Architecture

## Overview

CORAL runs as two Slurm jobs on NOAA Ursa HPC:

1. **GPU node** (`u1-h100`): Ollama serves the LLM (qwen3:32b) and embedding model (nomic-embed-text)
2. **Service node** (`u1-service`): CORAL agent, MCP servers, RAG pipeline, and web UI

The service node has internet access for NOAA API calls. Users connect via SSH tunnel.

## Components

### Agent Layer (`src/coral/agent.py`)

The core chat loop:
1. User sends a message
2. `_select_tools()` filters the 80+ available tools to a relevant subset based on keywords
3. Message + filtered tools are sent to Ollama
4. If the LLM calls tools, they're executed via MCP and results fed back
5. Loop continues until the LLM produces a text response (max 10 iterations)

### MCP Bridge (`src/coral/mcp_bridge.py`)

Manages connections to all MCP servers:
- Starts all servers in parallel via `asyncio.gather`
- Each server runs as a subprocess (stdio transport)
- Discovers tools from each server and builds a unified tool registry
- Routes tool calls to the correct server session

### MCP Servers

**External (ocean-mcp via uvx):**
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

**Custom (built-in):**
- `netcdf_server.py` — Read/query NetCDF files via xarray
- `slurm_server.py` — Parse Slurm jobs and logs
- `ecflow_server.py` — ecFlow suite status
- `viz_server.py` — Execute Python scripts for analysis/plots
- `rag_server.py` — Search indexed documentation

### RAG Pipeline (`src/coral/rag/`)

- **Chunkers**: Split files by type (Fortran subroutines, C functions, namelist groups, etc.)
- **Indexer**: Embeds chunks via Ollama and stores in LanceDB
- **Retriever**: Hybrid search — vector similarity + BM25 full-text, merged with Reciprocal Rank Fusion

### Web UI (`src/coral/web_ui.py`)

Gradio ChatInterface on port 7860, accessed via SSH tunnel.

## Data Flow

```
User query
  → _select_tools() filters relevant tools
  → Ollama generates response (may include tool calls)
  → MCP Bridge routes tool calls to correct server
  → Server executes (API call, file read, subprocess, etc.)
  → Results returned to Ollama for next iteration
  → Final text response to user
```

## Tool Routing

Small models (8B) can't handle 80+ tool definitions. `_select_tools()` uses keyword matching to narrow tools before sending to the LLM. For example, "water level" queries only see CO-OPS tools; "hurricane" queries only see NHC tools. Unknown queries get all tools.
