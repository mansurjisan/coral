# CORAL on TACC Vista

TACC Vista deployment for CORAL using NVIDIA Grace Hopper GH200 (ARM64).

## Quick Start

```bash
# One-time setup (from login node)
source slurm/vista/setup.sh

# Start Ollama on a GPU node
sbatch slurm/vista/start_ollama.sh
squeue -u $USER  # wait for RUNNING, note the node name

# Pull the model (first time only)
export OLLAMA_HOST=http://<gpu-node>:11434
$WORK/coral_install/ollama_install/bin/ollama pull qwen3:32b

# Start CORAL
source $WORK/coral_install/.venv/bin/activate
module load gcc/14.2.0 python3/3.11.8
export OLLAMA_HOST=http://<gpu-node>:11434
cd $WORK/coral_install/coral
coral chat --model qwen3:32b --mode multi
```

## System Details

| Component | Detail |
|-----------|--------|
| CPU | NVIDIA Grace (ARM64) |
| GPU | NVIDIA GH200 (96GB HBM3) |
| Partition | `gh-dev` (dev), `gh` (production) |
| Max wall time | 2 hours (gh-dev) |
| Python | 3.11.8 via `module load gcc/14.2.0 python3/3.11.8` |
| Ollama binary | `ollama-linux-arm64` |
| Storage | `$WORK` for install, `$SCRATCH` for temp |

## Notes

- Grace Hopper nodes include GPUs by default — no `--gres` needed
- Python 3.11 requires `LD_LIBRARY_PATH` set for MCP subprocesses (handled by `python_wrapper.sh`)
- The `recon-mcp` server may need manual install if it fails during batch setup
- Vista uses Slurm (not PBS), so all HPC system tools work natively
