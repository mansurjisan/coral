# Local Development Setup

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed
- Node.js 18+ (for `uvx`/`npx` MCP servers)
- ~8GB RAM minimum (for qwen3:8b)

## 1. Clone and install

```bash
git clone https://github.com/mansurjisan/coral.git
cd coral
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## 2. Pull models

```bash
ollama pull qwen3:8b          # Small model for local testing
ollama pull nomic-embed-text  # Embedding model for RAG
```

## 3. Run

```bash
# CLI chat
coral chat --model qwen3:8b

# List available tools
coral tools

# Web UI
coral serve --model qwen3:8b --port 7860
```

## 4. Run tests

```bash
pytest tests/ -v -m "not integration"
```

## Notes

- Local testing uses qwen3:8b which has limited tool-calling ability. Production uses qwen3:32b on H100 GPUs.
- Some MCP servers (slurm, ecflow) require HPC-specific commands and won't work locally — they'll log warnings on startup and CORAL will skip them.
- The RAG pipeline requires `nomic-embed-text` running in Ollama for indexing and search.
