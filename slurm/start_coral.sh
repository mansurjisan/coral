#!/bin/bash
#SBATCH --job-name=coral-agent
#SBATCH --account=your_project
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

CORAL_DIR="${CORAL_DIR:-/path/to/coral}"
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

# --- Activate environment and start ---
source "$CORAL_DIR/.venv/bin/activate"
cd "$CORAL_DIR"

echo "Starting CORAL web UI on port 7860 at $(date)"
coral serve --model qwen3:32b --config coral_config.json --port 7860
