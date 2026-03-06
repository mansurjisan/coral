# 🪸 CORAL — Coastal Ocean Research AI Layer

**A self-hosted AI agent for NOAA HPC that connects local LLMs to ocean data, scientific documentation, and HPC workflows — entirely within NOAA's network.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-68%20passing-brightgreen.svg)]()

## What It Does

CORAL combines a local LLM (via Ollama) with live ocean data tools, a RAG knowledge base over scientific code and documentation, and HPC workflow integration:

- **Live ocean data** — Query real-time water levels, hurricane tracks, storm surge forecasts, recon flights, and satellite data through [ocean-mcp](https://github.com/mansurjisan/ocean-mcp) servers
- **RAG over documentation** — Search SCHISM/ADCIRC source code, NOAA tech memos, model configs, and namelists
- **Local file interaction** — Inspect NetCDF model outputs, parse Slurm logs, monitor ecFlow workflows
- **Code execution** — Generate and run Python analysis scripts (xarray, matplotlib, cartopy)
- **CLI + Web UI** — Interactive terminal chat or Gradio web interface

All running on Ollama with open-weight LLMs. No external APIs, no commercial licenses.

```text
You: What is the current water level at The Battery, NYC?

🪸 CORAL: [calls coops-mcp → get_water_levels]
  The current water level at The Battery (station 8518750) is 0.42m
  above MLLW as of 2025-03-05 14:30 UTC. The predicted tide was 0.38m,
  so the surge residual is +0.04m. No flood thresholds exceeded.

You: Compare that with the STOFS forecast for the next 48 hours.

🪸 CORAL: [calls stofs-mcp → get_stofs_forecast]
  STOFS-3D-Atlantic shows water levels peaking at 0.67m MLLW tomorrow
  at 03:00 UTC — 0.12m above predicted tide. Below minor flood
  threshold (0.83m). No flooding expected.

You: My last STOFS run on Ursa failed. What happened?

🪸 CORAL: [calls slurm-mcp → diagnose_job_failure]
  Job 4521893 failed with exit code 137 (SIGKILL) after 2h15m.
  MaxRSS was 180GB against 192GB limit — likely OOM. The log shows
  "CFL violation at timestep 4521" before termination. Consider
  reducing dt or increasing --mem allocation.

You: What does the SCHISM parameter IHFSKIP control?

🪸 CORAL: [calls rag-mcp → search_documentation]
  From schism_init.F90 (line 342): IHFSKIP controls the output skip
  count for hotstart files. Setting ihfskip=36 with dt=120s writes
  a hotstart file every 36×120 = 4320 seconds (1.2 hours).
```

## Data Sources

| Server | Source | Data |
|--------|--------|------|
| `coops-mcp` | CO-OPS | Water levels, tide predictions, met data from 200+ stations |
| `nhc-mcp` | NHC | Active storms, forecast tracks, surge warnings |
| `stofs-mcp` | STOFS | Storm surge forecasts |
| `recon-mcp` | Hurricane Hunters | Flight-level recon, vortex messages |
| `erddap-mcp` | CoastWatch ERDDAP | Satellite SST, ocean color, in-situ data |
| `ofs-mcp` | OFS | Regional nowcast/forecast guidance |

## Quick Start

```bash
# Install
git clone https://github.com/mansurjisan/coral.git
cd coral
pip install -e .

# Pull a model
ollama pull qwen3:8b

# Chat
coral chat --model qwen3:8b

# Web UI
coral serve --model qwen3:8b --port 7860
```

## Architecture

```
┌──────────────────────────────────┐
│         CORAL Agent              │
│     (Ollama + MCP bridge)        │
├──────────────┬───────────────────┤
│  Ocean Data  │  HPC Tools        │
│  ocean-mcp   │  netcdf-mcp       │
│  (6 servers) │  slurm-mcp        │
│              │  ecflow-mcp       │
└──────┬───────┴────────┬──────────┘
   NOAA APIs       Local files
```

## Related

- [ocean-mcp](https://github.com/mansurjisan/ocean-mcp) — MCP servers for NOAA ocean data
- [Ollama](https://ollama.com) — Local LLM inference
- [Model Context Protocol](https://modelcontextprotocol.io) — Tool integration standard

## Citation

If you use CORAL in your research or operations, please cite:

```bibtex
@software{jisan2025coral,
  author = {Jisan, Mansur},
  title = {CORAL: Coastal Ocean Research AI Layer},
  year = {2025},
  url = {https://github.com/mansurjisan/coral},
  note = {A self-hosted AI agent connecting local LLMs to NOAA ocean data via MCP}
}
```

## Author

**Mansur Jisan** — NOAA National Ocean Service

## License

Apache 2.0
