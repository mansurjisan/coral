#!/bin/bash
#SBATCH --job-name=coral-ollama
#SBATCH --account=DS-NOAA-AI
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ollama_%j.log
#
# Start Ollama LLM server on a TACC Vista Grace Hopper node.
# Edit --account to match your project allocation.
# Vista GH200 nodes include GPUs by default (no --gres needed).

set -euo pipefail

# --- Paths ---
export PATH=$WORK/coral_install/ollama_install/bin:$PATH
export OLLAMA_MODELS=$WORK/coral_install/ollama_models

# --- Server config ---
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

# Write hostname so the CORAL CLI can auto-detect
HOST_FILE=$WORK/coral_install/coral_host.env
echo "OLLAMA_NODE=$(hostname)" > "$HOST_FILE"
echo "Ollama starting on $(hostname):11434 at $(date)"

# Run (blocks until job killed/timeout)
ollama serve
