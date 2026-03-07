#!/bin/bash
#SBATCH --job-name=coral-agent
#SBATCH --account=gpu-nos-surge
#SBATCH --partition=u1-service
#SBATCH --qos=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/coral_%j.log
#
# Start CORAL agent + web UI on a service node (has internet for NOAA APIs).
# Edit --account and CORAL_DIR to match your setup.

set -euo pipefail

CORAL_DIR="${CORAL_DIR:-/scratch5/purged/$USER/CORAL}"
HOST_FILE=/scratch5/purged/$USER/coral_host.env

# --- Wait for Ollama GPU job to start ---
echo "Waiting for Ollama to start..."
WAIT_SECONDS=0
while [ ! -f "$HOST_FILE" ]; do
    sleep 10
    WAIT_SECONDS=$((WAIT_SECONDS + 10))
    if [ $WAIT_SECONDS -ge 600 ]; then
        echo "ERROR: Ollama did not start within 10 minutes. Exiting."
        exit 1
    fi
done

source "$HOST_FILE"
export OLLAMA_HOST=http://${OLLAMA_NODE}:11434
echo "Ollama found at $OLLAMA_HOST"

# --- Verify Ollama is responding ---
for i in $(seq 1 30); do
    if curl -sf "$OLLAMA_HOST/api/version" > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    echo "  Waiting for Ollama API... ($i/30)"
    sleep 5
done

# --- Paths for uv/uvx (MCP servers) and Ollama ---
export PATH=/scratch5/purged/$USER/uv:$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/scratch5/purged/$USER/uv_cache
export UV_PYTHON_INSTALL_DIR=/scratch5/purged/$USER/uv_python
export XDG_DATA_HOME=/scratch5/purged/$USER/.local/share

# --- Activate environment and start ---
source "$CORAL_DIR/.venv/bin/activate"
cd "$CORAL_DIR"

# Require sandboxed viz execution on Ursa. If the image is missing, execute_python
# will refuse to run instead of falling back to host-side Python.
export CORAL_ENV=ursa
export CORAL_REQUIRE_SANDBOX=1
if [ -f "$CORAL_DIR/containers/coral_sandbox.sif" ]; then
    export CORAL_SANDBOX_SIF="$CORAL_DIR/containers/coral_sandbox.sif"
fi

echo "Starting CORAL web UI on port 7860 at $(date)"
coral serve --model qwen3:32b --config coral_config.json --port 7860
