# CORAL — Coastal Ocean Research AI Layer

**A self-hosted AI agent for NOAA HPC that connects local LLMs to ocean data, scientific documentation, and HPC workflows — entirely within NOAA's network.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-68%20passing-brightgreen.svg)]()

## What It Does

CORAL runs a local LLM ([Ollama](https://ollama.com) + [Qwen3](https://huggingface.co/Qwen)) and connects it to real data and tools through the [Model Context Protocol](https://modelcontextprotocol.io):

- **Live ocean data** — Water levels, hurricane tracks, storm surge forecasts, recon flights, satellite imagery via [ocean-mcp](https://github.com/mansurjisan/ocean-mcp)
- **RAG over documentation** — Search SCHISM/ADCIRC source code, NOAA tech memos, namelists, ecFlow suites
- **HPC tools** — Inspect NetCDF outputs, diagnose Slurm failures, monitor ecFlow workflows
- **Code execution** — Generate and run Python analysis scripts (xarray, matplotlib, cartopy)

No data leaves NOAA's network. No API keys. No per-token costs.

```text
You: What is the current water level at The Battery, NYC?

CORAL: [calls coops-mcp → get_water_levels]
  The current water level at The Battery (station 8518750) is 0.42m
  above MLLW as of 2025-03-05 14:30 UTC.

You: My last STOFS run on Ursa failed. What happened?

CORAL: [calls slurm-mcp → diagnose_job_failure]
  Job 4521893 failed with exit code 137 (SIGKILL) after 2h15m.
  MaxRSS was 180GB against 192GB limit — likely OOM. The log shows
  "CFL violation at timestep 4521" before termination.

You: What does the SCHISM parameter IHFSKIP control?

CORAL: [calls rag-mcp → search_documentation]
  From schism_init.F90 (line 342): IHFSKIP controls the output skip
  count for hotstart files. Setting ihfskip=36 with dt=120s writes
  a hotstart file every 4320 seconds.
```

## Data Sources

| Server | Source | Data |
|--------|--------|------|
| `coops-mcp` | CO-OPS | Water levels, tide predictions, met data from 200+ stations |
| `nhc-mcp` | NHC | Active storms, forecast tracks, surge warnings |
| `stofs-mcp` | STOFS | Storm surge forecasts |
| `recon-mcp` | Hurricane Hunters | Flight-level recon, vortex messages |
| `erddap-mcp` | CoastWatch ERDDAP | Satellite SST, ocean color |
| `ofs-mcp` | OFS | Regional nowcast/forecast guidance |

## Quick Start

```bash
git clone https://github.com/mansurjisan/coral.git
cd coral
pip install -e .

ollama pull qwen3:8b

coral chat --model qwen3:8b
```

### Other Commands

```bash
coral serve --model qwen3:8b --port 7860   # Web UI
coral index /path/to/schism/src             # Index docs for RAG
coral tools                                 # List available tools
```

## Architecture

```
┌──────────────────────────────────────────┐
│            CORAL Agent                   │
│        (Ollama + MCP bridge)             │
├──────────┬───────────┬───────────────────┤
│ Ocean    │ HPC       │ Knowledge Base    │
│ Data     │ Tools     │ (RAG)             │
│          │           │                   │
│ coops    │ netcdf    │ LanceDB           │
│ nhc      │ slurm     │ nomic-embed-text  │
│ stofs    │ ecflow    │ Fortran/C/PDF     │
│ recon    │ viz       │ chunkers          │
│ erddap   │           │                   │
│ ofs      │           │                   │
└──────────┴───────────┴───────────────────┘
```

## HPC Deployment

CORAL runs on NOAA HPC with Ollama on GPU nodes and the agent on service nodes. See `slurm/` for job scripts.

## Status

- [x] Phase 1: MCP agent + CLI + Web UI
- [x] Phase 2: RAG pipeline (Fortran, namelist, PDF, ecFlow chunkers)
- [x] Phase 3: HPC MCP servers (NetCDF, Slurm, ecFlow, code execution)
- [ ] Phase 4: Apptainer sandbox for safe code execution
- [ ] Phase 5: HPC deployment and testing

## Related

- [ocean-mcp](https://github.com/mansurjisan/ocean-mcp) — MCP servers for NOAA ocean data
- [Ollama](https://ollama.com) — Local LLM inference
- [Model Context Protocol](https://modelcontextprotocol.io) — Tool integration standard

## Citation

```bibtex
@software{jisan2025coral,
  author = {Jisan, Mansur},
  title = {CORAL: Coastal Ocean Research AI Layer},
  year = {2025},
  url = {https://github.com/mansurjisan/coral}
}
```

## Author

**Mansur Jisan** — NOAA National Ocean Service

## License

Apache 2.0
