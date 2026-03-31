# 🪸 CORAL — Coastal Ocean Research AI Layer

**A self-hosted AI agent for NOAA HPC that connects local LLMs to ocean data, scientific documentation, and HPC workflows entirely within NOAA's network.**

<p align="center">
  <a href="https://github.com/mansurjisan/coral/actions/workflows/ci.yml"><img src="https://github.com/mansurjisan/coral/actions/workflows/ci.yml/badge.svg?branch=feature%2Fmulti-agent" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/LLM-Ollama-white.svg" alt="Ollama"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/protocol-MCP-purple.svg" alt="MCP"></a>
  <a href="https://github.com/mansurjisan/ocean-mcp"><img src="https://img.shields.io/badge/tools-150+-green.svg" alt="Tools"></a>
</p>

## What It Does

CORAL combines a local LLM (via Ollama) with 150+ tools across 22 MCP servers for ocean data retrieval, code analysis, HPC workflow management, and operational forecast system support:

- **Live ocean data** — Query real-time water levels, hurricane tracks, storm surge forecasts, satellite data through [ocean-mcp](https://github.com/mansurjisan/ocean-mcp) servers
- **RAG over documentation** — Search SCHISM/ADCIRC source code, NOAA tech memos, model configs, and namelists
- **HPC system management** — Disk quotas, FairShare, allocations, modules, partitions on Slurm and PBS systems
- **UFS experiment lifecycle** — Create, validate, submit, monitor, and collect outputs from UFS-Coastal experiments
- **NOS workflow support** — Read/compare OFS configs (SECOFS, STOFS-3D-ATL), diagnose failures, anomaly detection, skill assessment
- **MCP-mediated alerting** — Threshold monitoring for CO-OPS stations with policy controls and audit trail
- **Code execution** — Generate and run Python analysis scripts with Apptainer-backed sandboxing
- **Persistent memory** — Remembers user preferences, accounts, and paths across sessions
- **CLI + Web UI** — Interactive terminal chat with slash commands or Gradio web interface

All running on Ollama with open-weight LLMs. No external APIs, no commercial licenses.

## Quick Start

```bash
git clone https://github.com/mansurjisan/coral.git
cd coral
pip install -e .
ollama pull qwen3:32b
coral chat --model qwen3:32b --mode multi
```

For HPC deployment, see [NOAA Ursa setup](docs/setup_ursa.md) or [TACC Vista setup](docs/setup_vista.md).

## MCP Servers

CORAL connects to **22 MCP servers** providing 150+ tools across three categories:

- **Ocean data** (12 servers) — CO-OPS, NHC, STOFS, ERDDAP, OFS, GOES, USGS, NDBC, WW3, ADCIRC, SCHISM, Hurricane Recon via [ocean-mcp](https://github.com/mansurjisan/ocean-mcp)
- **HPC & workflow** (6 servers) — Slurm, PBS (WCOSS2), ecFlow, UFS experiment runner, HPC system admin, NOS workflow configs
- **Local tools** (4 servers) — NetCDF queries, RAG documentation search, Python execution (sandboxed), threshold alerting

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

- [ocean-mcp](https://github.com/mansurjisan/ocean-mcp) — MCP servers for NOAA ocean data (18 servers)
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
  note = {A self-hosted AI agent connecting local LLMs to NOAA ocean data via MCP}
}
```

## Author

**Mansur Jisan** — NOAA National Ocean Service

## License

Apache 2.0
