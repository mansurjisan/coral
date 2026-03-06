#!/bin/bash
#SBATCH --job-name=coral-ollama
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/ollama_%j.log
#
# Start Ollama LLM server on a GPU node.
# Edit --account to match your project allocation.

set -euo pipefail

# --- Paths (edit if your Ollama install differs) ---
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/scratch5/purged/$USER/ollama_install/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=/scratch5/purged/$USER/ollama_models

# --- Server config ---
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

# Write hostname so the agent job can find us
HOST_FILE=/scratch5/purged/$USER/coral_host.env
echo "OLLAMA_NODE=$(hostname)" > "$HOST_FILE"
echo "Ollama starting on $(hostname):11434 at $(date)"

# Run (blocks until job killed/timeout)
ollama serve
