# Deploying CORAL on TACC Vista

Step-by-step guide based on actual deployment on March 31, 2026.

## Prerequisites

- Active TACC account with allocation (e.g., `DS-NOAA-AI`)
- Access to `gh-dev` (development) or `gh` (production) partition
- Internet access on login nodes

## System Details

| Component | Detail |
|-----------|--------|
| CPU | NVIDIA Grace (ARM64, 72 cores) |
| GPU | NVIDIA GH200 (96 GB HBM3) |
| Partition | `gh-dev` (2h max), `gh` (production) |
| Python | 3.11.8 via `module load gcc/14.2.0 python3/3.11.8` |
| Ollama | ARM64 binary (`ollama-linux-arm64.tar.zst`) |
| Storage | `$WORK` (1 TB), `$SCRATCH` (temp), `$HOME` (23 GB) |

## Quick Start (Automated)

```bash
module load gcc/14.2.0 python3/3.11.8
cd $WORK
git clone https://github.com/mansurjisan/coral.git
cd coral && git checkout feature/multi-agent
source slurm/vista/setup.sh
```

This runs the full setup: modules, venv, CORAL, ocean-mcp servers, Ollama. Takes ~10 minutes.

## Manual Setup

### 1. Load Modules

Vista's default Python is 3.9 (too old). Load 3.11:

```bash
module load gcc/14.2.0 python3/3.11.8
python3 --version
# Expected: Python 3.11.8
```

### 2. Create Virtual Environment

```bash
cd $WORK
mkdir -p coral_install && cd coral_install

python3 -m venv .venv

# ARM64 Python 3.11 needs LD_LIBRARY_PATH for shared lib
echo "export LD_LIBRARY_PATH=/opt/apps/gcc14/cuda12/python3/3.11.8/lib:\$LD_LIBRARY_PATH" >> .venv/bin/activate

source .venv/bin/activate
pip install --upgrade pip
```

### 3. Clone and Install CORAL

```bash
cd $WORK/coral_install
git clone https://github.com/mansurjisan/coral.git
cd coral
git checkout feature/multi-agent
pip install -e .
pip install pytest pytest-asyncio matplotlib
```

### 4. Install Ocean-MCP Servers

```bash
cd $WORK/coral_install
git clone https://github.com/mansurjisan/ocean-mcp.git
cd ocean-mcp/servers

for server in coops-mcp stofs-mcp nhc-mcp recon-mcp erddap-mcp ofs-mcp \
              adcirc-mcp goes-mcp schism-mcp usgs-mcp winds-mcp ww3-mcp \
              ufs-runner-mcp hpc-system-mcp nos-workflow-mcp alert-mcp vdatum-mcp; do
    cd $server && pip install -e . && cd ..
done
```

### 5. Create Python Wrapper

Vista's MCP subprocesses need `LD_LIBRARY_PATH` set explicitly. Create a wrapper:

```bash
INSTALL_DIR=$WORK/coral_install

cat > $INSTALL_DIR/python_wrapper.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/opt/apps/gcc14/cuda12/python3/3.11.8/lib:$LD_LIBRARY_PATH
exec /work/11036/mansurjisan/vista/coral_install/.venv/bin/python "$@"
EOF
chmod +x $INSTALL_DIR/python_wrapper.sh
```

**Important:** Edit the `exec` path to match your `$WORK` directory. Find it with `echo $WORK`.

Then update `coral_config.json` to use the wrapper:

```bash
cd $WORK/coral_install/coral
python3 -c "
import json, os
with open('coral_config.json') as f:
    cfg = json.load(f)
wrapper = os.environ['WORK'] + '/coral_install/python_wrapper.sh'
for srv in cfg['mcpServers'].values():
    srv['command'] = wrapper
with open('coral_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Updated all servers to use {wrapper}')
"
```

### 6. Install Ollama (ARM64)

```bash
cd $WORK/coral_install
curl -fSL https://github.com/ollama/ollama/releases/download/v0.19.0/ollama-linux-arm64.tar.zst -o ollama.tar.zst
zstd -d ollama.tar.zst -o ollama.tar
mkdir -p ollama_install
tar -xf ollama.tar -C ollama_install
rm ollama.tar.zst ollama.tar

export PATH=$WORK/coral_install/ollama_install/bin:$PATH
ollama --version
```

### 7. Run Tests

```bash
cd $WORK/coral_install/coral
python -m pytest tests/ -q --ignore=tests/test_rag_indexer.py --ignore=tests/test_rag_retriever.py
# Expected: 282+ passed
```

## Launch CORAL

### Start Ollama on a GPU Node

```bash
cd $WORK/coral_install/coral
mkdir -p logs
sbatch slurm/vista/start_ollama.sh
```

Check status:

```bash
squeue -u $USER
# JOBID  PARTITION  NAME          STATE    NODES  NODELIST
# 647507 gh-dev     coral-ollama  RUNNING  1      c642-002
```

### Pull the Model (First Time Only)

```bash
export PATH=$WORK/coral_install/ollama_install/bin:$PATH
export OLLAMA_HOST=http://<gpu-node>:11434
ollama pull qwen3:32b
```

### Start CORAL CLI

```bash
source $WORK/coral_install/.venv/bin/activate
module load gcc/14.2.0 python3/3.11.8
export OLLAMA_HOST=http://<gpu-node>:11434
cd $WORK/coral_install/coral
coral chat --model qwen3:32b --mode multi
```

Expected output:

```
   ██████╗ ██████╗ ██████╗  █████╗ ██╗
  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ╚██████╗╚██████╔╝██║  ██║██║  ██║███████╗
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

╭──────────────────────── CORAL v0.1.0 ────────────────────────╮
│  🪸 Model  qwen3:32b  │  Tools  144  │  Mode  multi-agent   │
│  🖥️  User   mansurjisan  │  Host   login2.vista.tacc.utexas.edu │
╰──────── Coastal Ocean Research AI Layer · /help for commands ╯
```

## Optional: NOS Workflow Configs

To enable NOS OFS config queries (SECOFS, STOFS-3D-ATL, etc.):

```bash
cd $WORK/coral_install
git clone --branch feature/unified-nowcast-forecast https://github.com/mansurjisan/nos-workflow.git
```

Add to `python_wrapper.sh`:

```bash
export NOS_WORKFLOW_DIR=$WORK/coral_install/nos-workflow
```

Then queries like "Compare SECOFS and STOFS-3D-ATL forcing configurations" will work.

## Troubleshooting

### `libpython3.11.so.1.0: cannot open shared object file`

The Python module isn't loaded or `LD_LIBRARY_PATH` isn't set:

```bash
module load gcc/14.2.0 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc14/cuda12/python3/3.11.8/lib:$LD_LIBRARY_PATH
```

For MCP subprocesses, ensure `python_wrapper.sh` includes the `LD_LIBRARY_PATH` export.

### `QOSMaxWallDurationPerJobLimit`

The `gh-dev` partition has a 2-hour max wall time:

```bash
#SBATCH --time=02:00:00
```

For longer sessions, use the `gh` production partition (if your allocation allows).

### `No module named recon_mcp`

The `recon-mcp` server may fail to install in the batch loop. Install manually:

```bash
cd $WORK/coral_install/ocean-mcp/servers/recon-mcp
pip install -e .
```

### `coral_config.json not found`

Run `coral chat` from the CORAL repo directory:

```bash
cd $WORK/coral_install/coral
coral chat --model qwen3:32b --mode multi
```

### GPU node not reachable

Verify the Ollama job is running and test connectivity:

```bash
squeue -u $USER
cat $WORK/coral_install/coral_host.env
curl http://<gpu-node>:11434/api/version
```

## Architecture on Vista

```
┌─────────────────────┐
│  gh-dev (GH200 GPU) │
│                     │
│  Ollama             │
│  qwen3:32b          │
│  96 GB HBM3         │
│  ARM64              │
└────────┬────────────┘
         │ HTTP :11434
         ▼
┌─────────────────────┐
│  login node         │
│                     │
│  CORAL CLI          │
│  22 MCP servers     │
│  144 tools          │
│  ARM64              │
└─────────────────────┘
```

Note: Unlike Ursa's two-job model (GPU + service node), Vista runs CORAL directly on the login node since login nodes have internet access. Only Ollama needs a GPU node.
