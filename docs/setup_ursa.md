# Setting Up CORAL on NOAA Ursa HPC

## Prerequisites

- Active account on Ursa with a compute allocation
- Access to `u1-h100` (GPU) and `u1-service` (internet) partitions
- Python 3.10+ available via module or conda

## 1. Install Ollama (no root required)

```bash
curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -C ~/.local -xzf -

# Add to ~/.bashrc
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc
```

## 2. Download models (from service node)

Service nodes have internet access. Request an interactive session:

```bash
salloc -A your_project -p u1-service -q batch -t 2:00:00

export OLLAMA_MODELS=/scratch5/purged/$USER/ollama_models
mkdir -p $OLLAMA_MODELS

ollama pull qwen3:32b           # Main agent model (~22GB)
ollama pull nomic-embed-text    # Embedding model for RAG (~275MB)
ollama pull qwen3:8b            # Lightweight fallback
```

## 3. Clone and install CORAL

```bash
git clone https://github.com/mansurjisan/coral.git
cd coral
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## 4. Configure

Edit the Slurm scripts to match your allocation:

```bash
# In slurm/start_ollama.sh and slurm/start_coral.sh:
#SBATCH --account=your_project   # ← your allocation

# In slurm/start_coral.sh:
CORAL_DIR="/path/to/coral"       # ← your coral clone path
```

## 5. Build the sandbox container (optional)

For safe Python code execution:

```bash
apptainer build coral_sandbox.sif containers/coral_sandbox.def
export CORAL_SANDBOX_SIF=$(pwd)/coral_sandbox.sif
```

## 6. Launch

```bash
bash slurm/start_all.sh
```

Monitor:

```bash
squeue -u $USER
```

Once both jobs are RUNNING, connect from your laptop:

```bash
ssh -L 7860:<SERVICE_NODE>:7860 $USER@ursa-bastion
# Open http://localhost:7860
```

## 7. Index documentation (optional)

```bash
# From a service node session
source .venv/bin/activate

coral index /path/to/schism/src
coral index /path/to/adcirc/src
coral index /path/to/noaa_tech_memos/
coral index /path/to/model_configs/
```

## Troubleshooting

**Ollama not found**: Ensure `~/.local/bin` is in PATH.

**Model download fails**: You must be on a service node (has internet). GPU nodes don't have internet.

**CORAL can't reach Ollama**: Check that `coral_host.env` was written by the Ollama job. Verify with `cat /scratch5/purged/$USER/coral_host.env`.

**Port 7860 not reachable**: Make sure the SSH tunnel targets the correct service node hostname.
