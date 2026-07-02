<h1 align="center">🪸 CORAL — Coastal Ocean Research AI Layer</h1>

<p align="center">
  <b>A domain-specific AI agent for coastal ocean and wave forecasting —<br>connecting open-weight LLMs to NOAA data, model source code, and HPC workflows, entirely within your network.</b>
</p>

<p align="center">
  <a href="https://github.com/mansurjisan/coral/actions/workflows/ci.yml"><img src="https://github.com/mansurjisan/coral/actions/workflows/ci.yml/badge.svg?branch=feature%2Fmulti-agent" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/LLM-Ollama-white.svg" alt="Ollama"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/protocol-MCP-purple.svg" alt="MCP"></a>
</p>

## Domain & Scope

CORAL is built for coastal hazards and forecasting: water levels and tides, storm surge, hurricanes, waves, and river discharge — together with the models that predict them (SCHISM, ADCIRC, WW3, UFS-Coastal) and the operational workflows that run them (ecFlow, Slurm/PBS). Data sources are NOAA-centric: CO-OPS, NHC, STOFS, ERDDAP, GOES, USGS. Atmospheric fields enter as observations and forcing — winds, pressure, hurricane reconnaissance — not as general atmospheric modeling.

Everything runs on open-weight LLMs via Ollama, entirely inside your network: no external LLM APIs, no cloud dependencies. That makes CORAL deployable where operational forecasting actually happens — air-gapped HPC systems, agency networks, and research clusters.

## What It Does

- **Coastal data & analysis** — Real-time water levels, tides, storm surge forecasts, hurricane tracks, wave conditions, and satellite data via [ocean-mcp](https://github.com/mansurjisan/ocean-mcp), plus Python code execution for plotting and analysis
- **Model code & documentation** — RAG search over SCHISM/ADCIRC/UFS-Coastal source code, NOAA tech memos, namelists, and NOS workflow configs
- **Forecast operations** — Slurm and PBS job diagnostics, ecFlow suite monitoring, UFS-Coastal experiment management, disk quotas, FairShare, threshold alerting
- **Persistent & portable** — Memory across sessions, CLI with slash commands, web UI; deployed on NOAA Ursa and TACC Vista

## Architecture

CORAL routes each query to one of three specialized sections — **Data**, **Code**, and **Workflow** — each of which sees only its own tools, enforced by a policy manifest. This domain-motivated decomposition is what keeps open-weight models reliable across 150+ tools: overloaded coastal terms like `stofs`, `schism`, and `adcirc` appear in data, code, and workflow contexts, so queries are routed by intent, with an LLM classifier as fallback for ambiguous cases. Cross-domain questions chain sections in dependency order (diagnose the run, then explain the code; fetch the data, then plot it). A single-agent mode (`--mode single`) remains available.

## Quick Start

```bash
git clone https://github.com/mansurjisan/coral.git
cd coral
pip install -e .
ollama pull qwen3:32b
coral chat --model qwen3:32b --mode multi
```

For HPC deployment, see [NOAA Ursa setup](docs/setup_ursa.md) or [TACC Vista setup](docs/setup_vista.md).

## Supported Models

CORAL works with any model available in [Ollama](https://ollama.com/library). Pull the model and pass it with `--model`:

```bash
ollama pull gemma4
coral chat --model gemma4 --mode multi
```

You can also switch models mid-session using the `@` prefix:

```
You: @gemma4 What is the current water level at The Battery?
```

| Model | Size | Notes |
|-------|------|-------|
| `qwen3:32b` | 20 GB | Default, strong tool calling |
| `gemma4` | 10 GB | Google's latest, good reasoning |
| `llama3.3:70b` | 40 GB | Largest open model, needs >48 GB VRAM |
| `qwen3:8b` | 5 GB | Lightweight, fast, good for testing |

## MCP Servers

CORAL connects to **23 MCP servers** providing 150+ tools, allowlisted per section by the policy manifest:

- **Data (14 servers)** — CO-OPS, NHC, STOFS, Hurricane Recon, ERDDAP, OFS, ADCIRC, GOES, SCHISM, USGS, Winds, WW3, VDatum via [ocean-mcp](https://github.com/mansurjisan/ocean-mcp), plus local NetCDF queries
- **Code (2 servers)** — RAG documentation search, sandboxed Python execution
- **Workflow (7 servers)** — Slurm, PBS (WCOSS2), ecFlow, UFS experiment runner, HPC system info, NOS workflow configs, threshold alerting

## CLI Features

```
╭──────────────────────── CORAL v0.1.0 ────────────────────────╮
│  🪸 Model  qwen3:32b  │  Tools  150  │  Mode  multi-agent   │
│  🖥️  User   mansurjisan  │  Host   login2.vista.tacc.utexas.edu │
│  🧠 3 memories loaded                                        │
╰──── Coastal Ocean Research AI Layer · /help for commands ────╯
```

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/tools` | Tool count per section (DATA/CODE/WORKFLOW) |
| `/status` | Quick dashboard: Ollama health, running jobs, disk usage |
| `/memory` | Show saved memories |
| `/remember` | Save a preference (e.g. `/remember account = coastal`) |
| `/forget` | Remove a memory |
| `/save` | Export conversation to markdown |
| `/report` | Auto-generate HPC status report |
| `/techmemo` | Auto-generate NOAA tech memo draft |
| `/alert` | Set threshold alert (e.g. `/alert 8518750 > 1.5`) |
| `/watch` | Monitor a job (e.g. `/watch 9848988`) |
| `/branch` | Save conversation, start fresh |
| `/audit` | Show tool call history and stats |
| `/clear` | Clear conversation history |
| `@model` | Override model for one query (e.g. `@qwen3:8b What is SCHISM?`) |

## Audit & Policy

- **Audit logging** — Every query and tool call recorded with timestamps, routing decisions, tool timing, confidence scores, and sandbox usage
- **Policy manifest** (`src/coral/policy_manifest.json`) — Defines which MCP servers each agent section can access, trust classes, and per-environment sandbox requirements
- **Sandbox enforcement** — Set `CORAL_REQUIRE_SANDBOX=1` to block host-side Python execution
- **Tool caching** — 5-minute TTL cache for read-only tools (HPC status, configs)
- **Retry with backoff** — Automatic retry on transient Ollama connection failures

## Related

- [ocean-mcp](https://github.com/mansurjisan/ocean-mcp) — MCP servers for NOAA ocean data (19 servers)
- [nos-workflow](https://github.com/mansurjisan/nos-workflow) — NOS Unified Operational Forecast System workflow
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
  note = {A domain-specific AI agent for coastal ocean and wave forecasting, self-hosted on open-weight LLMs}
}
```

## Author

**Mansur Jisan** — NOAA National Ocean Service

## License

Apache 2.0
