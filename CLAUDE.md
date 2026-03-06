# CLAUDE.md — CORAL (Coastal Ocean Research AI Layer)

## What is CORAL?

CORAL is a self-hosted AI agent for NOAA HPC that gives coastal scientists a ChatGPT-like experience — entirely within NOAA's network. It combines:

1. **Live ocean data** via ocean-mcp servers (CO-OPS tides, NHC hurricanes, STOFS surge, recon flights, ERDDAP satellite)
2. **RAG over documentation** — answer questions from SCHISM/ADCIRC source code, NOAA tech memos, model configs, user guides
3. **Local file interaction** — read NetCDF model outputs, parse Slurm logs, analyze ecFlow workflows, modify namelists
4. **Code execution** — generate and run Python analysis scripts (xarray, matplotlib, cartopy) in a sandboxed environment
5. **CLI coding assistant** — Claude Code compatible via Ollama's Anthropic API for agentic coding on HPC

All running on Ollama with open-weight LLMs on NVIDIA H100 GPUs. No external APIs, no commercial licenses, no root access required.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CORAL on NOAA Ursa HPC                                          │
│                                                                  │
│  ┌──────────────────┐     ┌────────────────────────────────────┐ │
│  │  u1-h100 (GPU)   │     │  u1-service (has internet)         │ │
│  │                   │     │                                    │ │
│  │  ┌──────────────┐ │     │  ┌──────────────────────────────┐  │ │
│  │  │ Ollama       │ │     │  │  MCP Servers                 │  │ │
│  │  │ Qwen3-32B   │◄├─────┤─►│  ocean-mcp (6 servers)       │──┼─► NOAA APIs
│  │  │ + nomic-    │ │     │  │  filesystem-mcp (official)   │  │ │
│  │  │   embed-text│ │     │  │  mcp-shell-server            │  │ │
│  │  │ port 11434  │ │     │  │  coral-netcdf-mcp (custom)   │  │ │
│  │  └──────────────┘ │     │  │  coral-rag-mcp (custom)      │  │ │
│  │  2x H100 NVL     │     │  └──────────────┬───────────────┘  │ │
│  └──────────────────┘     │                 │                   │ │
│                            │  ┌──────────────▼───────────────┐  │ │
│                            │  │  Agent Layer                 │  │ │
│                            │  │  smolagents CodeAgent        │  │ │
│                            │  │  + MCPHost (CLI bridge)      │  │ │
│                            │  └──────────────┬───────────────┘  │ │
│                            │                 │                   │ │
│                            │  ┌──────────────▼───────────────┐  │ │
│                            │  │  RAG Pipeline                │  │ │
│                            │  │  LlamaIndex + LanceDB        │  │ │
│                            │  │  tree-sitter (Fortran/C)     │  │ │
│                            │  │  Docling (PDF parsing)       │  │ │
│                            │  └──────────────────────────────┘  │ │
│                            │                                    │ │
│                            │  ┌──────────────────────────────┐  │ │
│                            │  │  Web UI (Gradio)             │  │ │
│                            │  │  port 7860                   │──┼─► SSH tunnel
│                            │  └──────────────────────────────┘  │ │
│                            └────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
coral/
├── CLAUDE.md
├── README.md
├── LICENSE                             # Apache-2.0
├── pyproject.toml
├── coral_config.json                   # MCP server configuration
│
├── src/
│   └── coral/
│       ├── __init__.py
│       │
│       ├── # ── Core Agent ──
│       ├── agent.py                    # Main agent loop: smolagents CodeAgent
│       ├── mcp_bridge.py              # Connect to MCP servers, discover tools
│       ├── ollama_client.py           # Ollama API wrapper with tool-calling
│       ├── config.py                  # Load configs, Ollama host, model settings
│       ├── prompts.py                 # System prompts for CORAL persona
│       │
│       ├── # ── RAG Pipeline ──
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── indexer.py             # Index documents into LanceDB
│       │   ├── retriever.py           # Hybrid BM25 + semantic search
│       │   ├── chunkers/
│       │   │   ├── __init__.py
│       │   │   ├── fortran_chunker.py # tree-sitter AST chunking for Fortran
│       │   │   ├── c_chunker.py       # tree-sitter AST chunking for C
│       │   │   ├── namelist_chunker.py# f90nml-based namelist parsing
│       │   │   ├── ecflow_chunker.py  # ecFlow suite definition parser
│       │   │   └── markdown_chunker.py# Markdown/RST header-based splitting
│       │   └── parsers/
│       │       ├── __init__.py
│       │       ├── pdf_parser.py      # Docling wrapper for NOAA tech memos
│       │       └── code_parser.py     # tree-sitter wrapper with context
│       │
│       ├── # ── Custom MCP Servers ──
│       ├── servers/
│       │   ├── __init__.py
│       │   ├── netcdf_server.py       # MCP server: read/query NetCDF files
│       │   ├── slurm_server.py        # MCP server: parse Slurm logs/jobs
│       │   ├── ecflow_server.py       # MCP server: ecFlow suite status/logs
│       │   ├── rag_server.py          # MCP server: RAG query interface
│       │   └── viz_server.py          # MCP server: generate plots from data
│       │
│       ├── # ── Interfaces ──
│       ├── web_ui.py                  # Gradio chat interface
│       └── cli.py                     # Typer CLI: coral chat, coral index, coral serve
│
├── slurm/
│   ├── start_ollama.sh                # Slurm job for GPU node
│   ├── start_coral.sh                 # Slurm job for service node
│   └── start_all.sh                   # Launch both with dependency
│
├── containers/
│   └── coral_sandbox.def              # Apptainer definition for safe code execution
│
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_rag_indexer.py
│   ├── test_rag_retriever.py
│   ├── test_fortran_chunker.py
│   ├── test_netcdf_server.py
│   ├── test_slurm_server.py
│   └── fixtures/
│       ├── sample_fortran.f90
│       ├── sample_namelist.nml
│       ├── sample_netcdf.nc           # Small test NetCDF
│       ├── sample_slurm_log.txt
│       └── mock_mcp_responses.json
│
├── docs/
│   ├── setup_ursa.md
│   ├── setup_local.md                 # Laptop development setup
│   ├── adding_documents.md            # How to index new docs into RAG
│   ├── adding_mcp_servers.md          # How to add new MCP servers
│   └── architecture.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

## Dependencies

```toml
[project]
name = "coral-ocean-ai"
version = "0.1.0"
description = "Coastal Ocean Research AI Layer — self-hosted AI agent for NOAA HPC"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
authors = [{name = "Mansur Jisan", email = "mansur.jisan@gmail.com"}]
keywords = ["noaa", "ocean", "ai", "mcp", "ollama", "hurricane", "storm-surge", "rag", "hpc"]

dependencies = [
    # Core agent
    "ollama>=0.4",
    "mcp>=1.0",
    "smolagents>=1.0",
    "typer>=0.12",
    "rich>=13.0",
    "gradio>=5.0",
    "httpx>=0.27",

    # RAG pipeline
    "llama-index-core>=0.12",
    "llama-index-llms-ollama>=0.4",
    "llama-index-embeddings-ollama>=0.4",
    "llama-index-vector-stores-lancedb>=0.4",
    "lancedb>=0.15",
    "tree-sitter>=0.23",
    "tree-sitter-languages>=1.10",
    "docling>=2.0",
    "f90nml>=1.4",

    # Scientific data tools
    "xarray>=2024.1",
    "netCDF4>=1.7",
    "numpy>=1.26",
    "pandas>=2.1",
    "matplotlib>=3.8",
    "cartopy>=0.22",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]

[project.scripts]
coral = "coral.cli:app"
```

## Implementation Plan — Build Order

**IMPORTANT: Build iteratively. Get each phase working before starting the next. Test on your laptop first, deploy to Ursa after Phase 3.**

---

### Phase 1: MCP Agent Foundation (Days 1-3)

**Goal:** A working chat agent that connects to ocean-mcp servers via MCP and answers ocean data questions. This is the MVP.

#### 1a. Project scaffold

Create the project structure, pyproject.toml, empty modules. Install dependencies in a venv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

#### 1b. MCP configuration

Create `coral_config.json`:

```json
{
  "mcpServers": {
    "coops": {
      "command": "uvx",
      "args": ["coops-mcp"]
    },
    "nhc": {
      "command": "uvx",
      "args": ["nhc-mcp"]
    },
    "stofs": {
      "command": "uvx",
      "args": ["stofs-mcp"]
    },
    "recon": {
      "command": "uvx",
      "args": ["recon-mcp"]
    },
    "erddap": {
      "command": "uvx",
      "args": ["erddap-mcp"]
    },
    "ofs": {
      "command": "uvx",
      "args": ["ofs-mcp"]
    }
  }
}
```

#### 1c. MCP bridge (`src/coral/mcp_bridge.py`)

Connect to MCP servers and auto-discover tools:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

class MCPBridge:
    """Connects to multiple MCP servers and discovers their tools."""

    def __init__(self, config_path: str):
        self.config = json.loads(open(config_path).read())
        self.sessions = {}      # server_name → ClientSession
        self.tools = []          # All discovered tools (Ollama format)
        self.tool_map = {}       # tool_name → (session, server_name)

    async def connect_all(self):
        """Connect to all configured MCP servers."""
        for name, cfg in self.config["mcpServers"].items():
            try:
                params = StdioServerParameters(command=cfg["command"], args=cfg["args"])
                read, write = await stdio_client(params).__aenter__()
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                self.sessions[name] = session

                # Discover tools
                result = await session.list_tools()
                for tool in result.tools:
                    ollama_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }
                    }
                    self.tools.append(ollama_tool)
                    self.tool_map[tool.name] = (session, name)
            except Exception as e:
                print(f"[CORAL] Warning: Could not connect to {name}: {e}")

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call via MCP."""
        session, server_name = self.tool_map[tool_name]
        result = await session.call_tool(tool_name, arguments)
        return result.content[0].text
```

#### 1d. Agent loop (`src/coral/agent.py`)

The core agent: takes user input, calls Ollama with tools, executes tool calls, returns response:

```python
import ollama
from coral.mcp_bridge import MCPBridge
from coral.prompts import CORAL_SYSTEM_PROMPT

class CoralAgent:
    def __init__(self, model: str, mcp_bridge: MCPBridge):
        self.model = model
        self.mcp = mcp_bridge
        self.history = []

    async def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": CORAL_SYSTEM_PROMPT}] + self.history

        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=self.mcp.tools,
        )

        # Agent loop: keep executing tools until model gives a text response
        max_iterations = 10
        iteration = 0
        while response.message.tool_calls and iteration < max_iterations:
            self.history.append(response.message)

            for tool_call in response.message.tool_calls:
                try:
                    result = await self.mcp.call_tool(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                except Exception as e:
                    result = f"Error calling {tool_call.function.name}: {e}"

                self.history.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_name": tool_call.function.name,
                })

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": CORAL_SYSTEM_PROMPT}] + self.history,
                tools=self.mcp.tools,
            )
            iteration += 1

        self.history.append(response.message)
        return response.message.content
```

#### 1e. CLI (`src/coral/cli.py`)

```python
import typer
import asyncio
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def chat(
    model: str = typer.Option("qwen3:32b", help="Ollama model name"),
    config: str = typer.Option("coral_config.json", help="MCP config path"),
):
    """Interactive chat with CORAL."""
    from coral.agent import CoralAgent
    from coral.mcp_bridge import MCPBridge

    async def run():
        bridge = MCPBridge(config)
        console.print("[bold cyan]🪸 CORAL — Connecting to MCP servers...[/]")
        await bridge.connect_all()
        console.print(f"[green]✓ Connected. {len(bridge.tools)} tools available.[/]")

        agent = CoralAgent(model=model, mcp_bridge=bridge)

        while True:
            try:
                user_input = console.input("[bold]You:[/] ")
                if user_input.lower() in ("exit", "quit"):
                    break
                with console.status("Thinking..."):
                    response = await agent.chat(user_input)
                console.print(f"\n[bold cyan]CORAL:[/] {response}\n")
            except KeyboardInterrupt:
                break

    asyncio.run(run())

@app.command()
def serve(
    model: str = typer.Option("qwen3:32b"),
    config: str = typer.Option("coral_config.json"),
    port: int = typer.Option(7860),
):
    """Launch CORAL web UI."""
    from coral.web_ui import launch
    launch(model=model, config=config, port=port)
```

#### 1f. System prompt (`src/coral/prompts.py`)

```python
CORAL_SYSTEM_PROMPT = """You are CORAL (Coastal Ocean Research AI Layer), an AI assistant
for NOAA coastal ocean scientists and forecasters running on NOAA HPC.

You have access to real-time NOAA ocean data through MCP tool servers, a RAG knowledge base
over scientific documentation and source code, and the ability to read local files on HPC.

CAPABILITIES:
- Query real-time and historical water levels from 200+ CO-OPS tide stations
- Retrieve hurricane tracks, forecasts, and warnings from NHC
- Access storm surge forecasts from NOAA STOFS operational models
- Pull Hurricane Hunter reconnaissance flight-level data
- Search satellite and ocean data from NOAA CoastWatch ERDDAP
- Search indexed documentation: SCHISM/ADCIRC source code, NOAA tech memos, model configs
- Read and analyze NetCDF model output files on the local filesystem
- Parse Slurm job logs and ecFlow workflow status
- Generate Python code for data analysis and visualization

RULES:
- Always use tools to get real data. Never fabricate water levels, positions, or forecasts.
- Include units (meters, knots, mb) and datum (NAVD, MLLW, MSL) in responses.
- Include timestamps with timezone (UTC) for all observations and forecasts.
- If a tool call fails, tell the user clearly and suggest alternatives.
- For multi-step questions, plan which tools to call and in what order.
- When generating code, use xarray for NetCDF, matplotlib/cartopy for plots, f90nml for namelists.
- Be concise but complete. This is for working scientists, not general public.

COMMON STATION IDS:
- 8518750: The Battery, New York
- 8658163: Wrightsville Beach, NC
- 8726520: St. Petersburg, FL
- 8728690: Apalachicola, FL
- 8761724: Grand Isle, LA
- 8443970: Boston, MA
- 8452660: Newport, RI
"""
```

#### 1g. Web UI (`src/coral/web_ui.py`)

```python
import gradio as gr
import asyncio
from coral.agent import CoralAgent
from coral.mcp_bridge import MCPBridge

agent = None

async def _init_agent(model: str, config: str):
    global agent
    bridge = MCPBridge(config)
    await bridge.connect_all()
    agent = CoralAgent(model=model, mcp_bridge=bridge)

async def respond(message, history):
    response = await agent.chat(message)
    return response

def launch(model: str, config: str, port: int):
    asyncio.run(_init_agent(model, config))

    demo = gr.ChatInterface(
        respond,
        title="🪸 CORAL — Coastal Ocean Research AI Layer",
        description="Ask questions about NOAA ocean data, model code, HPC workflows.",
        examples=[
            "What is the current water level at The Battery, NYC?",
            "Are there any active hurricanes in the Atlantic?",
            "Compare STOFS surge forecast vs observations at Tampa Bay",
            "What does the subroutine schism_init do?",
            "Show me my last failed Slurm job and explain the error",
        ],
        theme="soft",
    )
    demo.launch(server_name="0.0.0.0", server_port=port)
```

#### Phase 1 verification:

```bash
# Start Ollama with a small model for testing
ollama pull qwen3:8b

# Run CORAL CLI
coral chat --model qwen3:8b --config coral_config.json

# Test: "What is the current water level at Newport, RI?"
# Expected: calls coops-mcp, returns real data
```

---

### Phase 2: RAG Pipeline (Days 4-7)

**Goal:** CORAL can answer questions from indexed documentation — SCHISM source code, NOAA tech memos, model configs.

#### 2a. Fortran chunker (`src/coral/rag/chunkers/fortran_chunker.py`)

Use tree-sitter to parse Fortran into semantic chunks (subroutines, functions, modules):

```python
from tree_sitter_languages import get_parser

class FortranChunker:
    """Parse Fortran source into subroutine/function/module chunks using tree-sitter AST."""

    def __init__(self):
        self.parser = get_parser('fortran')

    def chunk(self, source_code: str, file_path: str) -> list[dict]:
        """Return list of chunks with metadata."""
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        chunks = []

        # Node types to extract as chunks
        target_types = {
            'subroutine', 'function', 'module', 'program',
            'subroutine_statement', 'function_statement',
        }

        def walk(node, depth=0):
            if node.type in target_types:
                text = source_code[node.start_byte:node.end_byte]
                chunks.append({
                    "text": text,
                    "metadata": {
                        "file_path": file_path,
                        "node_type": node.type,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "language": "fortran",
                    }
                })
            for child in node.children:
                walk(child, depth + 1)

        walk(tree.root_node)

        # If no AST nodes found (e.g., include files), fall back to fixed-size chunks
        if not chunks:
            chunks = self._fallback_chunk(source_code, file_path)

        return chunks

    def _fallback_chunk(self, source_code: str, file_path: str, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
        """Simple line-based fallback for files that don't parse well."""
        lines = source_code.split('\n')
        chunks = []
        current = []
        current_len = 0
        for i, line in enumerate(lines):
            current.append(line)
            current_len += len(line)
            if current_len >= chunk_size:
                chunks.append({
                    "text": '\n'.join(current),
                    "metadata": {"file_path": file_path, "start_line": i - len(current) + 2, "language": "fortran"}
                })
                # Keep overlap
                overlap_lines = current[-3:]
                current = overlap_lines
                current_len = sum(len(l) for l in current)
        if current:
            chunks.append({
                "text": '\n'.join(current),
                "metadata": {"file_path": file_path, "start_line": len(lines) - len(current) + 1, "language": "fortran"}
            })
        return chunks
```

#### 2b. Namelist chunker (`src/coral/rag/chunkers/namelist_chunker.py`)

```python
import f90nml

class NamelistChunker:
    """Parse Fortran namelists into individual namelist group chunks."""

    def chunk(self, content: str, file_path: str) -> list[dict]:
        chunks = []
        try:
            nml = f90nml.reads(content)
            for group_name, group_data in nml.items():
                text = f"&{group_name}\n"
                for key, value in group_data.items():
                    text += f"  {key} = {value}\n"
                text += "/"
                chunks.append({
                    "text": text,
                    "metadata": {
                        "file_path": file_path,
                        "namelist_group": group_name,
                        "type": "namelist",
                    }
                })
        except Exception:
            # Fall back to treating as plain text
            chunks.append({"text": content, "metadata": {"file_path": file_path, "type": "namelist"}})
        return chunks
```

#### 2c. PDF parser (`src/coral/rag/parsers/pdf_parser.py`)

Use Docling for NOAA technical memorandums:

```python
from docling.document_converter import DocumentConverter

class PDFParser:
    """Parse scientific PDFs using Docling (handles tables, equations, figures)."""

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, pdf_path: str) -> list[dict]:
        result = self.converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()

        # Split on headers
        chunks = []
        current_section = ""
        current_text = ""
        for line in markdown.split('\n'):
            if line.startswith('#'):
                if current_text.strip():
                    chunks.append({
                        "text": current_text.strip(),
                        "metadata": {"file_path": pdf_path, "section": current_section, "type": "pdf"}
                    })
                current_section = line.strip('# ').strip()
                current_text = line + '\n'
            else:
                current_text += line + '\n'
        if current_text.strip():
            chunks.append({
                "text": current_text.strip(),
                "metadata": {"file_path": pdf_path, "section": current_section, "type": "pdf"}
            })
        return chunks
```

#### 2d. Indexer (`src/coral/rag/indexer.py`)

```python
import lancedb
import ollama
from pathlib import Path
from coral.rag.chunkers.fortran_chunker import FortranChunker
from coral.rag.chunkers.namelist_chunker import NamelistChunker
from coral.rag.chunkers.markdown_chunker import MarkdownChunker
from coral.rag.parsers.pdf_parser import PDFParser

EMBED_MODEL = "nomic-embed-text"

class CoralIndexer:
    """Index documents into LanceDB with appropriate chunking per file type."""

    def __init__(self, db_path: str = "~/.coral/vectordb"):
        self.db = lancedb.connect(str(Path(db_path).expanduser()))
        self.fortran_chunker = FortranChunker()
        self.namelist_chunker = NamelistChunker()
        self.markdown_chunker = MarkdownChunker()
        self.pdf_parser = PDFParser()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama."""
        results = []
        for text in texts:
            resp = ollama.embed(model=EMBED_MODEL, input=text)
            results.append(resp["embeddings"][0])
        return results

    def _get_chunker(self, file_path: str):
        """Select chunker based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext in ('.f90', '.f', '.f77', '.ftn', '.F90', '.F'):
            return self.fortran_chunker
        elif ext in ('.c', '.h', '.cpp'):
            # Use tree-sitter C chunker (same pattern as Fortran)
            from coral.rag.chunkers.c_chunker import CChunker
            return CChunker()
        elif ext in ('.nml', '.namelist'):
            return self.namelist_chunker
        elif ext in ('.md', '.rst', '.txt'):
            return self.markdown_chunker
        elif ext == '.pdf':
            return None  # Use PDF parser instead
        else:
            return self.markdown_chunker  # Default fallback

    def index_file(self, file_path: str):
        """Index a single file."""
        file_path = str(Path(file_path).resolve())
        content = open(file_path).read()

        if file_path.endswith('.pdf'):
            chunks = self.pdf_parser.parse(file_path)
        else:
            chunker = self._get_chunker(file_path)
            chunks = chunker.chunk(content, file_path)

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "text": chunk["text"],
                "vector": embedding,
                **chunk["metadata"],
            })

        # Create or append to table
        table_name = "coral_docs"
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(records)
        else:
            self.db.create_table(table_name, records)

        return len(records)

    def index_directory(self, dir_path: str, extensions: list[str] = None):
        """Recursively index all matching files in a directory."""
        if extensions is None:
            extensions = ['.f90', '.f', '.F90', '.c', '.h', '.py', '.md', '.rst', '.txt', '.nml', '.pdf', '.yaml', '.yml']

        dir_path = Path(dir_path)
        total = 0
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    count = self.index_file(str(file_path))
                    total += count
                    print(f"  Indexed {file_path.name}: {count} chunks")
                except Exception as e:
                    print(f"  Error indexing {file_path.name}: {e}")
        return total
```

#### 2e. Retriever with hybrid search (`src/coral/rag/retriever.py`)

```python
import lancedb
import ollama
from pathlib import Path

EMBED_MODEL = "nomic-embed-text"

class CoralRetriever:
    """Hybrid BM25 + semantic search over indexed documents."""

    def __init__(self, db_path: str = "~/.coral/vectordb"):
        self.db = lancedb.connect(str(Path(db_path).expanduser()))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: combine semantic vector search with full-text BM25."""
        table = self.db.open_table("coral_docs")

        # Generate query embedding
        resp = ollama.embed(model=EMBED_MODEL, input=query)
        query_vector = resp["embeddings"][0]

        # Vector search
        vector_results = (
            table.search(query_vector)
            .limit(top_k)
            .to_list()
        )

        # Full-text search (LanceDB supports this natively)
        try:
            fts_results = (
                table.search(query, query_type="fts")
                .limit(top_k)
                .to_list()
            )
        except Exception:
            fts_results = []

        # Reciprocal Rank Fusion
        return self._rrf_merge(vector_results, fts_results, top_k)

    def _rrf_merge(self, vector_results, fts_results, top_k, k=60):
        """Reciprocal Rank Fusion to combine vector and BM25 results."""
        scores = {}
        for rank, r in enumerate(vector_results):
            key = r.get("file_path", "") + str(r.get("start_line", ""))
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            scores[key + "_data"] = r
        for rank, r in enumerate(fts_results):
            key = r.get("file_path", "") + str(r.get("start_line", ""))
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key + "_data" not in scores:
                scores[key + "_data"] = r

        # Sort by combined score
        sorted_keys = sorted(
            [k for k in scores if not k.endswith("_data")],
            key=lambda k: scores[k],
            reverse=True
        )[:top_k]

        return [scores[k + "_data"] for k in sorted_keys if k + "_data" in scores]
```

#### 2f. RAG MCP server (`src/coral/servers/rag_server.py`)

Expose RAG as an MCP server so the LLM can search documentation:

```python
from mcp.server.fastmcp import FastMCP
from coral.rag.retriever import CoralRetriever

mcp = FastMCP("coral-rag")
retriever = CoralRetriever()

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
    results = retriever.search(query, top_k=top_k)
    output = []
    for r in results:
        source = r.get("file_path", "unknown")
        line = r.get("start_line", "")
        node_type = r.get("node_type", "")
        text = r.get("text", "")[:2000]  # Truncate long chunks
        header = f"--- {source}"
        if line:
            header += f":{line}"
        if node_type:
            header += f" ({node_type})"
        header += " ---"
        output.append(f"{header}\n{text}")
    return "\n\n".join(output) if output else "No results found."

if __name__ == "__main__":
    mcp.run()
```

#### 2g. CLI index command

Add to `cli.py`:

```python
@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    db_path: str = typer.Option("~/.coral/vectordb", help="Vector DB path"),
):
    """Index documents into CORAL's RAG knowledge base."""
    from coral.rag.indexer import CoralIndexer
    from pathlib import Path

    indexer = CoralIndexer(db_path=db_path)
    p = Path(path)

    if p.is_file():
        count = indexer.index_file(str(p))
        console.print(f"[green]Indexed {p.name}: {count} chunks[/]")
    elif p.is_dir():
        count = indexer.index_directory(str(p))
        console.print(f"[green]Indexed {p}: {count} total chunks[/]")
    else:
        console.print(f"[red]Path not found: {path}[/]")
```

#### 2h. Add RAG server to config

Update `coral_config.json` to include the RAG server:

```json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["-m", "coral.servers.rag_server"]
    }
  }
}
```

#### Phase 2 verification:

```bash
# Index SCHISM source code
coral index /path/to/schism/src

# Index some NOAA tech memos
coral index /path/to/noaa_tech_memos/

# Index model namelists
coral index /path/to/param.nml

# Test RAG
coral chat --model qwen3:8b
> What does the subroutine schism_init do?
# Expected: retrieves relevant Fortran code chunks and explains
```

---

### Phase 3: Custom MCP Servers for HPC (Days 8-12)

**Goal:** CORAL can read NetCDF files, parse Slurm logs, and check ecFlow status.

#### 3a. NetCDF MCP server (`src/coral/servers/netcdf_server.py`)

```python
from mcp.server.fastmcp import FastMCP
import xarray as xr
import json
import numpy as np

mcp = FastMCP("coral-netcdf")

@mcp.tool()
def inspect_netcdf(file_path: str) -> str:
    """Inspect a NetCDF file: show dimensions, variables, global attributes.

    Args:
        file_path: Path to the NetCDF file on the filesystem.
    """
    ds = xr.open_dataset(file_path)
    info = {
        "dimensions": {k: v for k, v in ds.dims.items()},
        "variables": {},
        "global_attrs": dict(ds.attrs),
    }
    for var_name, var in ds.data_vars.items():
        info["variables"][var_name] = {
            "dims": list(var.dims),
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "units": var.attrs.get("units", ""),
            "long_name": var.attrs.get("long_name", ""),
        }
    ds.close()
    return json.dumps(info, indent=2, default=str)

@mcp.tool()
def query_netcdf(file_path: str, variable: str, lat: float = None, lon: float = None,
                 time: str = None, time_index: int = None) -> str:
    """Query a specific variable from a NetCDF file at a given location and/or time.

    Args:
        file_path: Path to the NetCDF file.
        variable: Variable name to query (e.g., 'zeta', 'temp', 'elev').
        lat: Latitude for nearest-neighbor selection.
        lon: Longitude for nearest-neighbor selection.
        time: Time string for selection (e.g., '2024-10-09T12:00:00').
        time_index: Integer time index (alternative to time string).
    """
    ds = xr.open_dataset(file_path)
    da = ds[variable]

    sel = {}
    if time is not None:
        sel["time"] = time
    elif time_index is not None:
        da = da.isel(time=time_index)
    if lat is not None and "lat" in da.dims:
        sel["lat"] = lat
    if lon is not None and "lon" in da.dims:
        sel["lon"] = lon

    if sel:
        da = da.sel(**sel, method="nearest")

    values = da.values
    units = da.attrs.get("units", "")

    # Summarize
    if values.size == 1:
        result = f"{variable} = {float(values)} {units}"
    elif values.size < 20:
        result = f"{variable} = {values.tolist()} {units}"
    else:
        result = (
            f"{variable}: shape={values.shape}, "
            f"min={float(np.nanmin(values)):.4f}, max={float(np.nanmax(values)):.4f}, "
            f"mean={float(np.nanmean(values)):.4f} {units}"
        )

    ds.close()
    return result

@mcp.tool()
def get_netcdf_timeseries(file_path: str, variable: str, lat: float, lon: float) -> str:
    """Extract a full time series of a variable at a given lat/lon from a NetCDF file.

    Returns JSON array of {time, value} pairs.

    Args:
        file_path: Path to the NetCDF file.
        variable: Variable name.
        lat: Latitude.
        lon: Longitude.
    """
    ds = xr.open_dataset(file_path)
    da = ds[variable].sel(lat=lat, lon=lon, method="nearest")

    records = []
    for t in da.time.values:
        val = float(da.sel(time=t).values)
        records.append({"time": str(t), "value": val})

    ds.close()
    return json.dumps(records[:500])  # Cap at 500 points

if __name__ == "__main__":
    mcp.run()
```

#### 3b. Slurm MCP server (`src/coral/servers/slurm_server.py`)

```python
from mcp.server.fastmcp import FastMCP
import subprocess
import re

mcp = FastMCP("coral-slurm")

@mcp.tool()
def get_my_jobs(state: str = "all") -> str:
    """Get current user's Slurm jobs.

    Args:
        state: Job state filter: 'all', 'running', 'pending', 'failed', 'completed'
    """
    state_map = {"all": "", "running": "-t RUNNING", "pending": "-t PENDING",
                 "failed": "-t FAILED", "completed": "-t COMPLETED"}
    flag = state_map.get(state, "")
    cmd = f"sacct -X --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,MaxRSS,NodeList -n {flag}".split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else "No jobs found."
    except Exception as e:
        return f"Error running sacct: {e}"

@mcp.tool()
def get_job_details(job_id: str) -> str:
    """Get detailed information about a specific Slurm job.

    Args:
        job_id: Slurm job ID.
    """
    cmd = f"scontrol show job {job_id}".split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else f"Job {job_id} not found."
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def read_job_log(job_id: str, tail_lines: int = 100) -> str:
    """Read the output/error log for a Slurm job.

    Searches for common log patterns: slurm-{job_id}.out, logs/*_{job_id}.log

    Args:
        job_id: Slurm job ID.
        tail_lines: Number of lines from end of log to return (default 100).
    """
    import glob
    patterns = [
        f"slurm-{job_id}.out",
        f"logs/*_{job_id}.log",
        f"logs/*_{job_id}.out",
        f"**/slurm-{job_id}.out",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            log_path = matches[0]
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                return f"Log: {log_path}\n" + "".join(lines[-tail_lines:])
            except Exception as e:
                return f"Error reading {log_path}: {e}"
    return f"No log file found for job {job_id}."

@mcp.tool()
def diagnose_job_failure(job_id: str) -> str:
    """Analyze a failed Slurm job: get exit code, resource usage, and common error patterns.

    Args:
        job_id: Slurm job ID.
    """
    # Get job info
    info_cmd = f"sacct -j {job_id} --format=JobID,State,ExitCode,MaxRSS,MaxVMSize,Elapsed,TimelimitRaw,ReqMem,ReqCPUS -n -X".split()
    info = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)

    # Get log
    log = read_job_log(job_id, tail_lines=50)

    # Detect common patterns
    diagnosis = []
    if "TIMEOUT" in info.stdout:
        diagnosis.append("JOB TIMED OUT: Wall time limit exceeded. Consider increasing --time.")
    if "OUT_OF_ME" in info.stdout or "oom" in log.lower():
        diagnosis.append("OUT OF MEMORY: Job exceeded memory limit. Increase --mem or reduce problem size.")
    if "SIGKILL" in log or "signal 9" in log:
        diagnosis.append("KILLED BY SIGNAL 9: Likely OOM killer. Check memory usage.")
    if "MPI_ABORT" in log or "mpirun" in log.lower():
        diagnosis.append("MPI ERROR: Check for MPI rank failures, missing libraries, or node communication issues.")

    output = f"=== Job {job_id} Summary ===\n{info.stdout}\n"
    if diagnosis:
        output += "\n=== Diagnosis ===\n" + "\n".join(diagnosis) + "\n"
    output += f"\n=== Log (last 50 lines) ===\n{log}"
    return output

if __name__ == "__main__":
    mcp.run()
```

#### 3c. ecFlow MCP server (`src/coral/servers/ecflow_server.py`)

```python
from mcp.server.fastmcp import FastMCP
import subprocess
import re

mcp = FastMCP("coral-ecflow")

@mcp.tool()
def get_suite_status(suite_name: str = "") -> str:
    """Get ecFlow suite status. Shows task states (complete, active, aborted, queued).

    Args:
        suite_name: Name of the ecFlow suite. If empty, shows all suites.
    """
    cmd = ["ecflow_client", "--get_state"]
    if suite_name:
        cmd.append(f"/{suite_name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout[:5000] if result.stdout else "No suite found or ecflow_client not available."
    except FileNotFoundError:
        return "ecflow_client not found. Make sure ecFlow module is loaded."
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def get_aborted_tasks(suite_name: str) -> str:
    """Find all aborted tasks in an ecFlow suite.

    Args:
        suite_name: Name of the ecFlow suite.
    """
    cmd = ["ecflow_client", "--get_state", f"/{suite_name}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        aborted = [line for line in result.stdout.split('\n') if 'state:aborted' in line]
        if aborted:
            return f"Found {len(aborted)} aborted tasks:\n" + "\n".join(aborted[:50])
        return f"No aborted tasks in /{suite_name}."
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def read_ecflow_job_output(task_path: str) -> str:
    """Read the job output (.1 file) for an ecFlow task.

    Args:
        task_path: Full ecFlow task path, e.g., /stofs/forecast/run_model
    """
    # ecFlow stores output in ECF_OUT directory
    cmd = ["ecflow_client", "--file", task_path, "jobout"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if len(output) > 5000:
            return output[:2000] + "\n...[truncated]...\n" + output[-2000:]
        return output if output else "No output found."
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()
```

#### 3d. Update coral_config.json with all servers

```json
{
  "mcpServers": {
    "coops": {"command": "uvx", "args": ["coops-mcp"]},
    "nhc": {"command": "uvx", "args": ["nhc-mcp"]},
    "stofs": {"command": "uvx", "args": ["stofs-mcp"]},
    "recon": {"command": "uvx", "args": ["recon-mcp"]},
    "erddap": {"command": "uvx", "args": ["erddap-mcp"]},
    "ofs": {"command": "uvx", "args": ["ofs-mcp"]},
    "rag": {"command": "python", "args": ["-m", "coral.servers.rag_server"]},
    "netcdf": {"command": "python", "args": ["-m", "coral.servers.netcdf_server"]},
    "slurm": {"command": "python", "args": ["-m", "coral.servers.slurm_server"]},
    "ecflow": {"command": "python", "args": ["-m", "coral.servers.ecflow_server"]},
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/scratch", "/home"]
    }
  }
}
```

#### Phase 3 verification:

```bash
coral chat --model qwen3:8b
> Inspect the NetCDF file /scratch/stofs/output/stofs_2d_glo.t00z.nc
> What was the maximum water elevation in that file?
> Show me my recent failed Slurm jobs
> What's the status of the stofs suite in ecFlow?
```

---

### Phase 4: Code Execution & Visualization (Days 13-16)

**Goal:** CORAL can generate and execute Python analysis scripts, producing plots.

#### 4a. Visualization MCP server (`src/coral/servers/viz_server.py`)

```python
from mcp.server.fastmcp import FastMCP
import subprocess
import tempfile
import os

mcp = FastMCP("coral-viz")

@mcp.tool()
def execute_python(code: str, description: str = "") -> str:
    """Execute a Python script for data analysis or visualization.

    The script has access to: xarray, netCDF4, matplotlib, cartopy, numpy, pandas, f90nml.
    If the script creates a figure, save it to '/tmp/coral_plot.png'.

    Args:
        code: Python code to execute.
        description: Brief description of what the code does.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Prepend common imports
        header = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
"""
        f.write(header + code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True,
            timeout=120,
            cwd="/tmp",
        )
        output = ""
        if result.stdout:
            output += result.stdout[:3000]
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr[:2000]}"
        if result.returncode != 0:
            output = f"Script failed (exit code {result.returncode}):\n{output}"
        elif os.path.exists("/tmp/coral_plot.png"):
            output += "\n[Plot saved to /tmp/coral_plot.png]"
        return output if output else "Script completed successfully (no output)."
    except subprocess.TimeoutExpired:
        return "Script timed out after 120 seconds."
    finally:
        os.unlink(script_path)

if __name__ == "__main__":
    mcp.run()
```

**Note on sandboxing for HPC:** For production deployment on Ursa, wrap execution in Apptainer:

```bash
# containers/coral_sandbox.def
Bootstrap: docker
From: python:3.11-slim

%post
    pip install --no-cache-dir xarray netcdf4 matplotlib cartopy numpy pandas f90nml scipy

%runscript
    python3 "$@"
```

Build: `apptainer build coral_sandbox.sif containers/coral_sandbox.def`

Then change the `execute_python` tool to run:
`apptainer exec --nv --bind /scratch coral_sandbox.sif python script.py`

#### Phase 4 verification:

```bash
coral chat --model qwen3:32b
> Plot the water level time series from /scratch/stofs/output/stofs.nc for the past 24 hours at lat=40.7, lon=-74.0. Save to /tmp/coral_plot.png.
# Expected: generates matplotlib code, executes it, tells user plot is saved
```

---

### Phase 5: Slurm Deployment on Ursa (Days 17-20)

#### 5a. Install Ollama without root

```bash
# On a login node
curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -C ~/.local -xzf -
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
# Add to ~/.bashrc
```

#### 5b. Download models (from service node — has internet)

```bash
salloc -A your_project -p u1-service -q batch -t 2:00:00

export OLLAMA_MODELS=/scratch5/purged/$USER/ollama_models
mkdir -p $OLLAMA_MODELS

# Pull models
ollama pull qwen3:32b            # Main agent model (~22GB)
ollama pull nomic-embed-text     # Embedding model (~275MB)
ollama pull qwen3:8b             # Lightweight fallback
```

#### 5c. Slurm job: Ollama on GPU

```bash
#!/bin/bash
# slurm/start_ollama.sh
#SBATCH --job-name=coral-ollama
#SBATCH --account=your_project
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/ollama_%j.log

export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/scratch5/purged/$USER/ollama_models
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

echo "OLLAMA_NODE=$(hostname)" > /scratch5/purged/$USER/coral_host.env
echo "Ollama starting on $(hostname):11434"

ollama serve
```

#### 5d. Slurm job: CORAL agent on service node

```bash
#!/bin/bash
# slurm/start_coral.sh
#SBATCH --job-name=coral-agent
#SBATCH --account=your_project
#SBATCH --partition=u1-service
#SBATCH --qos=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/coral_%j.log

# Wait for Ollama
while [ ! -f /scratch5/purged/$USER/coral_host.env ]; do
    echo "Waiting for Ollama..."
    sleep 10
done
source /scratch5/purged/$USER/coral_host.env
export OLLAMA_HOST=http://${OLLAMA_NODE}:11434

# Activate environment
source /path/to/coral/.venv/bin/activate

# Start web UI
coral serve --model qwen3:32b --config coral_config.json --port 7860
```

#### 5e. Launch script

```bash
#!/bin/bash
# slurm/start_all.sh
rm -f /scratch5/purged/$USER/coral_host.env

OLLAMA_JOB=$(sbatch --parsable slurm/start_ollama.sh)
echo "Ollama job: $OLLAMA_JOB"

CORAL_JOB=$(sbatch --parsable --dependency=after:$OLLAMA_JOB slurm/start_coral.sh)
echo "CORAL job: $CORAL_JOB"

echo ""
echo "After jobs start, connect with:"
echo "  ssh -L 7860:SERVICE_NODE:7860 \$USER@ursa-bastion"
echo "  Open http://localhost:7860"
```

---

### Phase 6: Index Your Team's Documentation (Days 21-25)

Once CORAL is running, populate the RAG knowledge base:

```bash
# SCHISM source code
coral index /path/to/schism/src --db-path /scratch5/purged/$USER/coral_vectordb

# ADCIRC source code
coral index /path/to/adcirc/src

# UFS-Coastal source
coral index /path/to/ufs-coastal/src

# NOAA technical memorandums (PDFs)
coral index /path/to/noaa_tech_memos/

# Model configuration files
coral index /path/to/model_configs/

# ecFlow suite definitions
coral index /path/to/ecflow_suites/

# Team READMEs and documentation
coral index /path/to/docs/
```

---

## Testing Strategy

### Unit tests (always pass, no network)

```python
# tests/test_fortran_chunker.py
def test_fortran_subroutine_extraction():
    code = """
subroutine test_sub(x, y)
  implicit none
  real, intent(in) :: x
  real, intent(out) :: y
  y = x * 2.0
end subroutine test_sub
"""
    chunker = FortranChunker()
    chunks = chunker.chunk(code, "test.f90")
    assert len(chunks) >= 1
    assert "test_sub" in chunks[0]["text"]
    assert chunks[0]["metadata"]["language"] == "fortran"

# tests/test_namelist_chunker.py
def test_namelist_parsing():
    content = """
&core
  dt = 120.0
  rnday = 5.0
/
&opt
  ihfskip = 36
/
"""
    chunker = NamelistChunker()
    chunks = chunker.chunk(content, "param.nml")
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["namelist_group"] == "core"
```

### Integration tests (need Ollama running)

```python
# tests/test_agent.py
@pytest.mark.integration
async def test_agent_tool_call():
    """Agent should call coops-mcp for water level questions."""
    bridge = MCPBridge("coral_config.json")
    await bridge.connect_all()
    agent = CoralAgent(model="qwen3:8b", mcp_bridge=bridge)
    response = await agent.chat("What is the current water level at station 8518750?")
    assert "water level" in response.lower() or "8518750" in response
```

---

## Example Queries by Capability

**Ocean data (MCP):**
- "What is the current water level at Newport, RI?"
- "Are there any active hurricanes? What's the surge forecast for Tampa Bay?"
- "Get SST from ERDDAP for the Gulf of Maine this week"

**RAG (documentation):**
- "What does the subroutine schism_init do in SCHISM?"
- "How is the NUOPC cap structured in UFS-Coastal?"
- "What parameters control the time step in the STOFS namelist?"
- "Explain the ADCIRC wetting/drying algorithm"

**Local files (NetCDF/logs):**
- "Inspect the file /scratch/stofs/output/stofs_2d.nc and tell me what variables it contains"
- "What was the maximum water elevation in last night's STOFS run?"
- "Show me my last 5 failed Slurm jobs and diagnose what went wrong"
- "What tasks are aborted in the stofs ecFlow suite?"

**Code execution (visualization):**
- "Plot water levels from /scratch/stofs/output/stofs_2d.nc at lat=40.7, lon=-74.0"
- "Read param.nml and show me the time step settings"
- "Generate a map of maximum surge from the maxele.63.nc file using cartopy"

**Cross-capability (combines multiple):**
- "Compare the STOFS forecast in /scratch/stofs/output.nc against real-time CO-OPS observations at The Battery for the last 24 hours. Plot both on the same figure."
- "My last STOFS run failed. Check the Slurm log, and look up in the SCHISM docs what the error message means."
