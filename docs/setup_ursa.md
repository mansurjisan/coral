# Deploying CORAL on NOAA Ursa HPC

Step-by-step guide based on actual deployment on March 6, 2026.

## Prerequisites

- Active Ursa account with GPU allocation (e.g., `gpu-nos-surge`)
- Access to `u1-h100` (GPU) and `u1-service` (internet) partitions
- Python 3.10+ available on front-end nodes

## 1. Install Ollama (no root required)

Ursa home directories have a small quota (~5 GB). Install everything to scratch.

```bash
# Download and extract Ollama to scratch
mkdir -p /scratch5/purged/$USER/ollama_install
curl -fsSL -o /tmp/ollama.tar.zst \
  https://github.com/ollama/ollama/releases/download/v0.17.7/ollama-linux-amd64.tar.zst
tar --zstd -xf /tmp/ollama.tar.zst -C /scratch5/purged/$USER/ollama_install
rm /tmp/ollama.tar.zst

# Symlink the binary to ~/.local/bin (tiny, fits in home quota)
mkdir -p ~/.local/bin
ln -sf /scratch5/purged/$USER/ollama_install/bin/ollama ~/.local/bin/ollama

# Add to PATH (may fail to write .bashrc if home quota is full — that's ok)
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/scratch5/purged/$USER/ollama_install/lib/ollama:${LD_LIBRARY_PATH:-}

# Verify
ollama --version
# Expected: "Warning: could not connect to a running Ollama instance"
# followed by version number — this is normal, server isn't running yet.
```

## 2. Pull models (requires internet — use service node)

```bash
# Get an interactive session on a service node
salloc -A gpu-nos-surge -p u1-service -q batch -n 1 -t 2:00:00

# Set model storage to scratch (models are large)
export OLLAMA_MODELS=/scratch5/purged/$USER/ollama_models
mkdir -p $OLLAMA_MODELS
export LD_LIBRARY_PATH=/scratch5/purged/$USER/ollama_install/lib/ollama:${LD_LIBRARY_PATH:-}

# Start Ollama server temporarily
ollama serve &
sleep 5

# Pull models
ollama pull qwen3:32b           # Main agent model (~22 GB)
ollama pull nomic-embed-text    # Embedding model for RAG (~275 MB)

# Optional: faster MoE variant
# ollama pull qwen3:30b-a3b

# Stop Ollama and exit the session
kill %1
exit
```

## 3. Clone and install CORAL

```bash
cd /scratch5/purged/$USER

git clone https://github.com/mansurjisan/coral.git CORAL
cd CORAL

python -m venv .venv
source .venv/bin/activate

# Redirect pip cache to scratch (home quota too small for build artifacts)
export PIP_CACHE_DIR=/scratch5/purged/$USER/pip_cache
export TMPDIR=/scratch5/purged/$USER/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR

pip install -e ".[all,dev]"
```

## 4. Install ocean-mcp servers

The ocean-mcp packages are pip-installed into the venv (not run via `uvx`, which has home quota issues on Ursa).

```bash
pip install coops-mcp nhc-mcp stofs-mcp recon-mcp erddap-mcp ofs-mcp \
            adcirc-mcp goes-mcp schism-mcp usgs-mcp winds-mcp ww3-mcp \
            vdatum-mcp
```

## 5. Build the Apptainer sandbox for `viz`

CORAL V2 now enforces sandboxed Python execution on Ursa for the `execute_python` tool. The Slurm launch script sets `CORAL_REQUIRE_SANDBOX=1`, so plotting/code-execution requests will refuse to run unless an Apptainer image is available.

Build the sandbox image from the checked-in definition and keep it at `containers/coral_sandbox.sif` inside the repo checkout:

```bash
cd /scratch5/purged/$USER/CORAL

# Build on a service node with internet access
apptainer build containers/coral_sandbox.sif containers/coral_sandbox.def
```

Verify the image before launching CORAL:

```bash
apptainer exec containers/coral_sandbox.sif \
  python3 -c "import xarray, cartopy, netCDF4, f90nml; print('ok')"
```

Expected output:

```text
ok
```

If you want to store the image elsewhere, set `CORAL_SANDBOX_SIF` before launch. The default Ursa path used by `slurm/start_coral.sh` is:

```bash
/scratch5/purged/$USER/CORAL/containers/coral_sandbox.sif
```

## 6. Launch CORAL (Slurm)

The Slurm scripts in `slurm/` are pre-configured for Ursa. They launch two jobs:

1. **coral-ollama** — Ollama LLM server on a GPU node (H100)
2. **coral-agent** — CORAL agent + web UI on a service node (has internet for NOAA APIs)

```bash
cd /scratch5/purged/$USER/CORAL
mkdir -p logs
bash slurm/start_all.sh
```

Monitor:

```bash
squeue -u $USER
```

Expected output:

```
JOBID  PARTITION   NAME          STATE    TIME  NODES  NODELIST
123456 u1-h100     coral-ollama  RUNNING  0:20  1      u23g03
123457 u1-service  coral-agent   RUNNING  0:17  1      ufe08
```

Check logs:

```bash
# Ollama — should show "inference compute" with H100 and ~93 GB VRAM
tail -10 logs/ollama_<JOBID>.log

# CORAL — should show "Connected. 108 tools available."
tail -20 logs/coral_<JOBID>.log
```

## 7. Connect via CLI

From any Ursa front-end node (no SSH tunnel needed):

```bash
cd /scratch5/purged/$USER/CORAL
source .venv/bin/activate

# Set OLLAMA_HOST to the GPU node (check logs/ollama_*.log or coral_host.env)
source /scratch5/purged/$USER/coral_host.env
export OLLAMA_HOST=http://${OLLAMA_NODE}:11434

coral chat --model qwen3:32b
```

Example session:

```
You: What is the current water level at The Battery, NYC?

CORAL: The current water level at The Battery, NYC (Station 8518750) is
0.37 meters above MLLW as of 2026-03-06 20:12 UTC.
```

## 8. Connect via Web UI (SSH tunnel)

The web UI runs on the service node. To access it from your laptop, you need an SSH tunnel through the Ursa bastion.

### Step 1: SSH to Ursa with port forwarding

```bash
ssh -L 7860:localhost:7860 Mansur.Jisan@ursa-rsa.boulder.rdhpcs.noaa.gov
```

When the bastion says "hit ^C within 5 seconds", press **Ctrl+C** and type the front-end node name (e.g., `ufe04`).

### Step 2: Tunnel from the front-end to the service node

Once on the front-end node, set up a hop to the service node where CORAL is running:

```bash
# Replace ufe08 with the actual node from squeue output
ssh -L 7860:localhost:7860 -N ufe08 &
```

Verify:

```bash
curl -s http://localhost:7860 | head -3
# Should return: <!doctype html>
```

### Step 3: Open in browser

Open **http://localhost:7860** on your laptop.

## 9. Index documentation (optional)

To enable RAG search over your team's source code and documentation:

```bash
source .venv/bin/activate
source /scratch5/purged/$USER/coral_host.env
export OLLAMA_HOST=http://${OLLAMA_NODE}:11434

coral index /path/to/schism/src
coral index /path/to/adcirc/src
coral index /path/to/noaa_tech_memos/
coral index /path/to/model_configs/
```

## Troubleshooting

### `Disk quota exceeded` errors

Almost everything should go on scratch. Common fixes:

```bash
export PIP_CACHE_DIR=/scratch5/purged/$USER/pip_cache
export TMPDIR=/scratch5/purged/$USER/tmp
export UV_CACHE_DIR=/scratch5/purged/$USER/uv_cache
export XDG_DATA_HOME=/scratch5/purged/$USER/.local/share
```

### `uvx: No such file or directory`

CORAL uses `python -m` (not `uvx`) to run ocean-mcp servers. If you see this error, make sure you ran `git pull` to get the latest `coral_config.json`.

### `Sandboxed Python execution is required`

This means CORAL received a plotting or Python-analysis request, but `execute_python` could not find a usable Apptainer image while `CORAL_REQUIRE_SANDBOX=1` was set by the Ursa launch script.

Check:

```bash
cd /scratch5/purged/$USER/CORAL
ls -lh containers/coral_sandbox.sif
apptainer exec containers/coral_sandbox.sif python3 -c "print('ok')"
```

If the file is missing, rebuild it:

```bash
apptainer build containers/coral_sandbox.sif containers/coral_sandbox.def
```

### MCP servers fail to connect

Check that the ocean-mcp packages are installed in the venv:

```bash
source .venv/bin/activate
python -m coops_mcp --help
```

If not found, reinstall:

```bash
pip install coops-mcp nhc-mcp stofs-mcp recon-mcp erddap-mcp ofs-mcp \
            adcirc-mcp goes-mcp schism-mcp usgs-mcp winds-mcp ww3-mcp \
            vdatum-mcp
```

### Ollama not found / can't connect

```bash
# Check if Ollama job is running
squeue -u $USER | grep coral-ollama

# Check the host file
cat /scratch5/purged/$USER/coral_host.env

# Test connectivity from front-end
curl http://<OLLAMA_NODE>:11434/api/version
```

### Cancel and restart

```bash
scancel <OLLAMA_JOBID> <CORAL_JOBID>
bash slurm/start_all.sh
```

## Architecture on Ursa

```
┌─────────────────┐     ┌──────────────────────────────────┐
│  u1-h100 (GPU)  │     │  u1-service (internet)           │
│                 │     │                                  │
│  Ollama         │◄───►│  CORAL Agent                     │
│  qwen3:32b      │     │  17 MCP servers (108 tools)      │
│  nomic-embed    │     │  Gradio web UI (:7860)           │
│  93 GB VRAM     │     │                                  │
└─────────────────┘     └──────────────────────────────────┘
                              ▲
                              │ SSH tunnel
                              │
                        ┌─────┴─────┐
                        │  Laptop   │
                        │  :7860    │
                        └───────────┘
```
